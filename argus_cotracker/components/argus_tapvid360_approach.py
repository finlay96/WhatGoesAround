import copy
from pathlib import Path
from tqdm import tqdm

from imageio.v2 import mimsave
import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from argus_cotracker.components.pipelines.argus.utils import get_models_custom
from conversions.mapper import Mappers
from data.data_utils import collate_fn
from data.dataloader import TAPVid360Dataset
from exploration.argus_src.src import pers2equi_batch
from exploration.argus_src.inference import SVDArguments


class ArgusCoTrackerTAP360Approach(torch.nn.Module):
    def __init__(self, accelerator, weight_dtype, argus_config, argus_pipeline, cotracker_pipeline,
                 device=torch.device("cuda:0")):
        super().__init__()
        self.device = device
        self.accelerator = accelerator
        self.weight_dtype = weight_dtype
        self.argus_config = argus_config
        self.argus_pipeline = argus_pipeline
        if cotracker_pipeline is not None:
            self.cotracker_pipeline = cotracker_pipeline.to(device)

    def forward_cotracker(self, queries, argus_latents, is_train=False):
        self.cotracker_pipeline(queries, argus_latents, is_train=is_train, device=self.device)

    @torch.no_grad()
    def debug_decode_gen_videos(self, latents):
        with torch.autocast(str(self.accelerator.device).replace(":0", ""),
                            enabled=self.accelerator.mixed_precision != 'no',
                            dtype=self.weight_dtype):
            generated_frames = self.argus_pipeline.decode_latents(latents, self.argus_config.num_frames,
                                                                  decode_chunk_size=self.argus_config.decode_chunk_size,
                                                                  extended_decoding=self.argus_config.extended_decoding,
                                                                  blend_decoding_ratio=self.argus_config.blend_decoding_ratio)
        return generated_frames

    def save_debug_360_videos(self, conditional_video_equi, generated_frames, out_path, fps=3):
        out_path.mkdir(exist_ok=True, parents=True)
        conditional_video_equi = ((conditional_video_equi.clamp(-1, 1) + 1) * 127.5).cpu().to(
            torch.float32).numpy().astype(
            np.uint8)
        conditional_video_equi = conditional_video_equi.transpose(0, 2, 3, 1)  # (T, H, W, C)
        mimsave(out_path / "conditional_equi.mp4", conditional_video_equi, fps=fps)

        # save the generated video, RGB -> BGR
        generated_frames = ((generated_frames.clamp(-1, 1) + 1) * 127.5).cpu().to(torch.float32).numpy().astype(
            np.uint8)
        generated_frames = generated_frames[0].transpose(1, 2, 3, 0)  # (T, H, W, C)
        mimsave(out_path / "generated_frames.mp4", generated_frames, fps=fps)

        # save the input video and the generated video side-by-side
        concatenated_video = np.concatenate([conditional_video_equi, generated_frames], axis=1)  # (T, 3H, W, C)
        mimsave(out_path / "combined.mp4", concatenated_video, fps=fps)

    @torch.no_grad()
    def forward_argus(self, input_video, eq_video, eq_mask, noise_conditioning=False):
        conditional_video_pers = copy.deepcopy(input_video)
        if eq_video is not None:
            conditional_video_equi = eq_video.to(self.weight_dtype)
            mask = eq_mask
        else:
            # TODO for the sake of experiments maybe I could just feed these in!!! or actually make my own with our logic
            conditional_video_equi, mask = pers2equi_batch(input_video.to(torch.float32), fov_x=fov_x,
                                                           roll=roll, pitch=pitch, yaw=yaw,
                                                           width=width, height=height, device=self.accelerator.device,
                                                           return_mask=True)  # (T, C, H, W)
            conditional_video_equi = conditional_video_equi.to(self.weight_dtype)
        # conditional_video_equi = conditional_video_equi + torch.randn_like(conditional_video_equi, device=accelerator.device) * noise_aug_strength
        # conditional_video_equi = conditional_video_equi + torch.randn_like(conditional_video_equi, device=accelerator.device) * noise_aug_strength * mask
        if noise_conditioning:
            conditional_video_equi = torch.where(mask == 1, input_video,
                                                 torch.randn_like(input_video, device=self.accelerator.device))
        conditional_video_equi = conditional_video_equi.unsqueeze(0)
        hw_ratio = conditional_video_pers.shape[-2] / conditional_video_pers.shape[-1]
        mask = mask.unsqueeze(0)
        conditional_video_pers = conditional_video_pers.unsqueeze(0)

        with torch.autocast(str(self.accelerator.device).replace(":0", ""),
                            enabled=self.accelerator.mixed_precision != 'no',
                            dtype=self.weight_dtype):
            num_frames_remaining = self.argus_config.num_frames
            num_frames_batch = self.argus_config.num_frames_batch if (
                    hasattr(self.argus_config,
                            'num_frames_batch') and self.argus_config.num_frames_batch is not None) else self.argus_config.num_frames
            generated_latents = None
            generated_latents_this = None
            generated_frames_this = None
            num_frames_processed = 0
            blend_frames = self.argus_config.blend_frames if hasattr(self.argus_config, 'blend_frames') else 0
            round = 0

            # TODO the frame length should be greater than 25 if possible
            num_frames_to_process = min(num_frames_batch, num_frames_remaining)
            conditional_video_input = conditional_video_equi[:, :num_frames_to_process] if round == 0 else \
                torch.cat([generated_frames_this[:, :, -blend_frames:].permute(0, 2, 1, 3, 4),
                           conditional_video_equi[
                               :, num_frames_processed + blend_frames: num_frames_processed + num_frames_to_process]],
                          dim=1)
            mask_this = mask[:, num_frames_processed: num_frames_processed + num_frames_to_process]
            conditional_video_pers_this = conditional_video_pers[
                :, num_frames_processed: num_frames_processed + num_frames_to_process]
            conditional_video_input = conditional_video_input + torch.randn_like(conditional_video_input,
                                                                                 device=self.accelerator.device) * self.argus_config.noise_aug_strength * mask_this

            fps = 3.0  # TODO hardcoded for now

            # TODO need to get the unet outputs - now we also need to get the eq frames stuff in there from the other bit too
            generated_latents_this, _ = self.argus_pipeline(
                conditional_video_input,  # (1, T, C, H, W)
                conditional_images=conditional_video_pers_this,  # (1, T, C, H, W)
                height=self.argus_config.height,
                width=self.argus_config.width,
                num_frames=num_frames_to_process,
                decode_chunk_size=self.argus_config.decode_chunk_size,
                motion_bucket_id=127,
                fps=fps,
                num_inference_steps=self.argus_config.num_inference_steps,
                # num_inference_steps=1,  # 50 # 25 # TODO need to change this ideally to 50
                noise_aug_strength=self.argus_config.noise_aug_strength,
                min_guidance_scale=self.argus_config.guidance_scale,
                max_guidance_scale=self.argus_config.guidance_scale,
                inference_final_rotation=self.argus_config.inference_final_rotation,
                blend_decoding_ratio=self.argus_config.blend_decoding_ratio,
                extended_decoding=self.argus_config.extended_decoding,
                rotation_during_inference=self.argus_config.rotation_during_inference,
                return_latents=True,
            )  # [B, T, C, H, W]

        return generated_latents_this

    def forward(self, video, queries, iters, conditional_video_equi=None, noise_conditioning=False):
        B, T, C_, H, W = video.shape
        N = queries.shape[2]
        fov_x = 75.18
        roll, pitch, yaw = 0.0, 0.0, 0.0
        width, height = W, H

        generated_latents_this = self.forward_argus(video, conditional_video_equi, eq_mask=None,
                                                    noise_conditioning=noise_conditioning)
        # Decode the latents

        # Run the SAM2 on the eq frames to make perspective images

        # Run Cotracker on the perspective frames
        if self.cotracker_pipeline is not None:
            self.forward_cotracker(queries, all_extracted_latents, is_train=self.training)

        return generated_latents_this, all_extracted_latents


if __name__ == "__main__":
    device = torch.device("cuda:0")
    unet_path = "/home/userfs/f/fgch500/storage/pretrained_models/video_generation/argus"
    ds_root = Path("/home/userfs/f/fgch500/storage/shared/track-everywhere/tapvid360")
    assert ds_root.exists()
    out_root = Path("/home/userfs/f/fgch500/storage/outputs/tracking/whatGoesAround/argus_feats")
    out_root.mkdir(parents=True, exist_ok=True)
    args = SVDArguments(unet_path=unet_path, num_inference_steps=50)
    weight_dtype = torch.float32
    accelerator = Accelerator(mixed_precision='no')

    argus_pipeline = get_models_custom(args, accelerator, weight_dtype)
    model = ArgusCoTrackerTAP360Approach(accelerator, weight_dtype, args, argus_pipeline, None, device=device)

    # TODO get dataloader
    with open(Path(__file__).parent.parent.parent / "data/mini_dataset_names.txt", "r") as f:
        video_names = [line.strip() for line in f.readlines()]

    dataset = TAPVid360Dataset(ds_root / f"TAPVid360-10k", transforms.Compose([transforms.ToTensor()]),
                               num_queries=256, num_frames=25, specific_videos_list=video_names)
    dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True)

    for i, sample in enumerate(tqdm(dl)):
        sample.to_device(device)
        mapper = Mappers(320, 240, 512, fov_x=75.18)  # TODO should use actual data for this
        # TODO currently not batchable
        eq_frames, eq_mask = mapper.image.perspective_image_to_equirectangular(
            sample.video[0].permute(0, 2, 3, 1) * 255,
            sample.rotations[0], to_uint=False)
        video = (sample.video[0] * 255) / 127.5 - 1
        video = video.to(device)

        eq_frames = (eq_frames.permute(0, 3, 1, 2).float()) / 127.5 - 1
        eq_frames = eq_frames.to(device)
        eq_mask = eq_mask.to(device).unsqueeze(1)

        with torch.no_grad():
            latents, last_layer_unet_outs = model.forward_argus(sample.video[0], eq_frames, eq_mask)

        latents= latents.cpu()
        last_layer_unet_outs = {k1: {k2: v2.to('cpu') for k2, v2 in v1.items()}
                                     for k1, v1 in last_layer_unet_outs.items()}
        out_data = {
            "latents": latents,
            "last_layer_unet_outs": last_layer_unet_outs,
        }

        torch.save(out_data, out_root / f"{sample.seq_name[0].replace('/', '-')}.pth")