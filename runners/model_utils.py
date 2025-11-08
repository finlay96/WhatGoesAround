import inspect

from accelerate.utils import is_compiled_module
from diffusers import AutoencoderKLTemporalDecoder
import numpy as np
import torch

from argus_cotracker.components.pipelines.sam2.sam2_pipeline import get_sam2_predictor
from argus_cotracker.components.argus_cotracker_approach import ArgusCoTracker
from argus_cotracker.components.pipelines.argus.utils import get_models_custom
from argus_cotracker.components.pipelines.sam2.sam2_pipeline import sam_preprocessing
from exploration.cotracker_utils import CoTracker
from exploration.argus_src.inference import SVDArguments


def get_models(sam_checkpoint, accelerator, device, debug_skip_vae=False):
    vae = None
    if not debug_skip_vae:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            'stabilityai/stable-video-diffusion-img2vid', subfolder="vae", revision=None)
        vae = accelerator.unwrap_model(vae)
        vae = vae.to(device).eval()

    sam_pred_video = get_sam2_predictor(single_image=False, checkpoint=sam_checkpoint, device=device)
    sam_pred_image = get_sam2_predictor(single_image=True, checkpoint=sam_checkpoint, device=device)
    cotracker = CoTracker()

    return vae, sam_pred_video, sam_pred_image, cotracker


def decode_latents(vae, latents: torch.Tensor, num_frames: int, decode_chunk_size: int = 25,
                   extended_decoding: bool = False, blend_decoding_ratio: int = 4) -> torch.Tensor:
    # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
    # if blend rotation, rotate the latents and blend the decoded frames
    decode_chunk_size = num_frames if decode_chunk_size is None else decode_chunk_size
    latents = latents.flatten(0, 1)

    latents = latents / vae.config.scaling_factor

    forward_vae_fn = vae._orig_mod.forward if is_compiled_module(vae) else vae.forward
    accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

    # decode decode_chunk_size frames at a time to avoid OOM
    frames = []
    for i in range(0, latents.shape[0], decode_chunk_size):
        num_frames_in = latents[i: i + decode_chunk_size].shape[0]
        decode_kwargs = {}
        if accepts_num_frames:
            # we only pass num_frames_in if it's expected
            decode_kwargs["num_frames"] = num_frames_in

        if extended_decoding:
            latent_width = latents.shape[-1]
            frame_left = vae.decode(latents[i: i + decode_chunk_size], **decode_kwargs).sample
            frame_right = vae.decode(torch.roll(latents[i: i + decode_chunk_size], latent_width // 2, dims=-1),
                                     **decode_kwargs).sample

            frame_width = frame_left.shape[-1]
            frame_right = torch.roll(frame_right, -frame_width // 2, dims=-1)

            blend_count = int(frame_width // blend_decoding_ratio)
            same_width = (frame_width // 2 - blend_count) // 2
            weight_left_half = torch.cat(
                [torch.zeros(same_width), torch.linspace(0, 1, frame_width // 2 - same_width * 2),
                 torch.ones(same_width)]).to(frame_left.device)
            weight_left = torch.cat([weight_left_half, weight_left_half.flip(0)])
            weight_right = 1 - weight_left
            frame = frame_left * weight_left + frame_right * weight_right
        else:
            frame = vae.decode(latents[i: i + decode_chunk_size], **decode_kwargs).sample
        frames.append(frame)
    frames = torch.cat(frames, dim=0)

    # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
    frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    frames = frames.float()
    return frames


class SAMRunner:
    def __init__(self, sam_img_pred=None, sam_video_pred=None):
        self.sam_img_pred = sam_img_pred
        self.sam_video_pred = sam_video_pred

    @staticmethod
    def _preprocess_points(positive_points: list, negative_points: list | None):
        # Convert points to NumPy array
        input_points = np.array(positive_points)
        input_labels = [1] * len(input_points)
        if negative_points is not None:
            input_points = np.concatenate([input_points, np.array(negative_points)], axis=0)
            input_labels += [0] * len(negative_points)

        return input_points, np.array(input_labels)

    def run_through_image(self, img: np.ndarray, bbox_xyxy: list | None = None, positive_points: list | None = None,
                          negative_points: list | None = None):
        # TODO internally allow this to be a torch image
        self.sam_img_pred.set_image(img)
        input_points, input_labels, box = None, None, None
        if positive_points is not None:
            input_points, input_labels = self._preprocess_points(positive_points, negative_points)
        else:
            box = np.array(bbox_xyxy)

        masks, scores, logits = self.sam_img_pred.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=box,
            multimask_output=True
        )

        mask = masks[np.argmax(scores)]

        return mask.astype(np.uint8) * 255

    def run_through_video(self, frames, masks):
        vid_sam_frames, pp_sam_masks = sam_preprocessing(frames, masks,
                                                         resize_to=(self.sam_video_pred.image_size,
                                                                    self.sam_video_pred.image_size))
        assert vid_sam_frames is not None, "Need to pass in all the frames for video sam"
        inference_state = self.sam_video_pred.init_state(vid_sam_frames, frames[0].shape[0],
                                                         frames[0].shape[1], self.sam_video_pred.device)
        for i, mask in enumerate(pp_sam_masks):
            self.sam_video_pred.add_new_mask(inference_state, 0, i + 1, mask)
        # with suppress_output():
        out_logits = [out_mask_logits for _, _, out_mask_logits in
                      self.sam_video_pred.propagate_in_video(inference_state)]
        confidence_scores = [torch.sigmoid(logits) for logits in out_logits]
        assert len(out_logits), "No masks generated"

        return torch.stack(out_logits)[:, :, 0].transpose(1, 0), torch.stack(confidence_scores)[:, :, 0].transpose(1, 0)


def get_360_latents_model(unet_path, accelerator, num_frames, device, weight_dtype=torch.float32):
    args = SVDArguments(unet_path=unet_path, num_inference_steps=50)
    args.blend_frames = 10 #5
    args.num_frames_batch = 25
    args.num_frames = num_frames

    argus_pipeline = get_models_custom(args, accelerator, weight_dtype)
    model = ArgusCoTracker(accelerator, weight_dtype, args, argus_pipeline, None, device=device)

    return model
