import copy

from imageio.v2 import mimsave
import numpy as np
import torch

from exploration.argus_src.src import pers2equi_batch


class ArgusCoTracker(torch.nn.Module):
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
    def forward_argus(self, input_video, eq_video, eq_mask, noise_conditioning=False, return_unet_latents=True):
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
        conditional_video_equi = conditional_video_equi.unsqueeze(0).to(device=self.accelerator.device)
        hw_ratio = conditional_video_pers.shape[-2] / conditional_video_pers.shape[-1]
        mask = mask.unsqueeze(0).to(device=self.accelerator.device)
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
            all_extracted_latents = None
            num_frames_processed = 0
            blend_frames = self.argus_config.blend_frames if hasattr(self.argus_config, 'blend_frames') else 0
            round = 0

            while num_frames_remaining > 0:
                # TODO the frame length should be greater than 25 if possible
                num_frames_to_process = min(num_frames_batch, num_frames_remaining)
                conditional_video_input = conditional_video_equi[:, :num_frames_to_process] if round == 0 else \
                    torch.cat([generated_frames_this[:, :, -blend_frames:].permute(0, 2, 1, 3, 4),
                               conditional_video_equi[
                                   :, num_frames_processed + blend_frames: num_frames_processed + num_frames_to_process]],
                              dim=1)
                if round > 0:
                    generated_frames_this = generated_frames_this.cpu()
                    del generated_frames_this

                mask_this = mask[:, num_frames_processed: num_frames_processed + num_frames_to_process]
                conditional_video_pers_this = conditional_video_pers[
                    :, num_frames_processed: num_frames_processed + num_frames_to_process]
                conditional_video_input = conditional_video_input.to() + torch.randn_like(conditional_video_input,
                                                                                     device=self.accelerator.device) * self.argus_config.noise_aug_strength * mask_this

                fps = 3.0  # TODO hardcoded for now

                # TODO need to get the unet outputs - now we also need to get the eq frames stuff in there from the other bit too

                generated_latents_this, extracted_latents_this = self.argus_pipeline(
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
                    get_final_unet_latents=return_unet_latents
                )  # [B, T, C, H, W]
                if blend_frames != 0:  # save current generation results into a file
                    generated_frames_this = self.argus_pipeline.decode_latents(generated_latents_this, num_frames_to_process,
                                                                    decode_chunk_size=self.argus_config.decode_chunk_size,
                                                                    extended_decoding=self.argus_config.extended_decoding,
                                                                    blend_decoding_ratio=self.argus_config.blend_decoding_ratio)
                if generated_latents is None:
                    generated_latents = generated_latents_this
                    all_extracted_latents = extracted_latents_this
                else:
                    blend_weight = torch.linspace(1, 0, blend_frames, device=self.accelerator.device,
                                                  dtype=generated_latents.dtype).view(1, blend_frames, 1, 1, 1)
                    generated_latents = torch.cat([generated_latents[:, :-blend_frames],
                                                   blend_weight * generated_latents[:, -blend_frames:] + (
                                                               1 - blend_weight) * generated_latents_this[
                                                       :, :blend_frames],
                                                   generated_latents_this[:, blend_frames:]], dim=1)
                    # TODO doesnt work for greater than 1 batchsize and untested currently
                    if extracted_latents_this is not None:
                        for k1 in all_extracted_latents.keys():
                            for k2 in all_extracted_latents[k1].keys():
                                all_extracted_latents[k1][k2] = torch.cat(
                                    [all_extracted_latents[k1][k2][:-blend_frames],
                                     blend_weight[0] * all_extracted_latents[k1][k2][-blend_frames:] + (
                                             1 - blend_weight[0]) * extracted_latents_this[k1][k2][:blend_frames],
                                     extracted_latents_this[k1][k2][blend_frames:]], dim=0)

                if num_frames_remaining == num_frames_to_process:
                    break

                num_frames_remaining -= (num_frames_to_process - blend_frames)
                num_frames_processed += (num_frames_to_process - blend_frames)
                round += 1
                del generated_latents_this, extracted_latents_this
                torch.cuda.empty_cache()

        return generated_latents, all_extracted_latents
