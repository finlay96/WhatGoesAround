from diffusers import UNetSpatioTemporalConditionModel, AutoencoderKLTemporalDecoder
import numpy as np
import argparse
import torch
from dataset.video_dataset import VideoDataset
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import os
from accelerate import Accelerator
from src import sample_svd, StableVideoDiffusionPipelineCustom, generate_mask_batch, video_psnr, AverageMeter, get_rpy
from mast3r.model import AsymmetricMASt3R
import random
import math
import cv2
from equilib import equi2pers
import lpips

def format_save_path(args, file_path: str) -> str:

	video_id = file_path.split('/')[-2]
	video_idx = file_path.split('/')[-1].split('.')[0]

	# save_file_path = f'rotate{args.inference_final_rotation}'
	save_file_path = f'{video_id}_{video_idx}'
	save_file_path += '_extended' if args.extended_decoding else ''
	save_file_path += '.mp4'

	return save_file_path

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# load model arguments
	parser.add_argument('--pretrained_model_name_or_path', type=str, default='stabilityai/stable-video-diffusion-img2vid', help='Base pipeline path.')
	parser.add_argument("--revision", type=str, default=None, required=False, help="Revision of pretrained model's checkpoint.")
	parser.add_argument("--val_base_folder", nargs='+', type=str, required=True, help="Path to the valiadtion dataset.")
	parser.add_argument("--val_clip_file", type=str, default=None, help="Path to the validation clip file.")
	parser.add_argument("--calibration_cache_path", type=str, default='../datasets/calibration_files', help="Path to the calibration cache file.")
	parser.add_argument('--val_save_folder', type=str, default='val_results', help='Path to save the generated videos.')
	parser.add_argument("--unet_path", type=str, required=True, help="Path to the pretrained U-Net model.")
	
	# data arguments
	parser.add_argument('--dataset_size', type=int, default=None, help='Number of videos to generate.')
	parser.add_argument("--width", type=int, default=768, help="Width of the generated video.")
	parser.add_argument("--height", type=int, default=384, help="Height of the generated video.")
	parser.add_argument('--equirectangular_input', action='store_true', help='Input is equirectangular.')
	parser.add_argument('--fov_x_min', type=float, default=90., help='Minimum width fov')
	parser.add_argument('--fov_x_max', type=float, default=90., help='Maximum width fov')
	parser.add_argument('--fov_y_min', type=float, default=60., help='Minimum height fov')
	parser.add_argument('--fov_y_max', type=float, default=60., help='Maximum height fov')
	parser.add_argument("--min_frame_rate", type=float, default=5., help="Minimum frame rate.")
	parser.add_argument("--max_frame_rate", type=float, default=5., help="Maximum frame rate.")
	parser.add_argument('--fixed_start_frame', action='store_true', help='for each video, start from the first frame, for debugging')
	parser.add_argument('--mast3r_img_size', type=int, default=None, help='long side of image for MAST3R.')

	# inference arguments
	parser.add_argument('--rotation_during_inference', action='store_true', help='Rotate the video during inference.')
	parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps.')
	parser.add_argument('--inference_final_rotation', type=int, default=0, help='Final rotation during inference.')
	parser.add_argument('--blend_decoding_ratio', type=int, default=4, help='Blend decoding ratio. typically 2 or 4')
	parser.add_argument('--extended_decoding', action='store_true', help='Use extended decoding.')
	parser.add_argument('--noise_conditioning', action='store_true', help='Condition on noise.')
	parser.add_argument('--dense_calibration', action='store_true', help='Dense calibration, i.e., use all frames for calibration.')

	parser.add_argument('--predict_camera_motion', action='store_true', help='Predict camera motion.')

	parser.add_argument("--num_frames", type=int, default=18, help="Number of frames to generate.")
	parser.add_argument("--decode_chunk_size", type=int, default=8, help="Decode chunk size.")
	parser.add_argument("--motion_bucket_id", type=int, default=127, help="Motion bucket ID.")
	parser.add_argument("--noise_aug_strength", type=float, default=0.02, help="Noise augmentation strength.")
	parser.add_argument("--guidance_scale", type=float, default=1., help="Minimum guidance scale.")

	args = parser.parse_args()
	args.calibration_cache_path = args.calibration_cache_path + f'_{args.num_frames}'
	os.makedirs(args.calibration_cache_path, exist_ok=True)

	weight_dtype = torch.float32
	accelerator = Accelerator(mixed_precision='no')

	assert args.equirectangular_input, "Only equirectangular input is supported."

	unet = UNetSpatioTemporalConditionModel.from_pretrained(
		pretrained_model_name_or_path=args.unet_path,
		subfolder="unet",
	)

	val_dataset = VideoDataset(args.val_base_folder, 
								clip_info_path = args.val_clip_file,
								sample_frames=args.num_frames,
							   fixed_start_frame=args.fixed_start_frame,
							   dataset_size=args.dataset_size,
							   width=args.width if args.equirectangular_input else None,
							   height=args.height if args.equirectangular_input else None,
							   mast3r_img_size=args.mast3r_img_size,
							   min_frame_rate=args.min_frame_rate, max_frame_rate=args.max_frame_rate,
							   dense_calibration=args.dense_calibration,)

	val_save_dir = os.path.join(args.unet_path, args.val_save_folder)
	os.makedirs(val_save_dir, exist_ok=True)

	pipeline = StableVideoDiffusionPipelineCustom.from_pretrained(
				args.pretrained_model_name_or_path,
				unet=unet,
				revision=args.revision,
				torch_dtype=weight_dtype,
				).to(accelerator.device)
	
	lpips_fn = lpips.LPIPS(net='vgg').to(accelerator.device)

	psnr_t0_meter, psnr_tn_meter, lpips_t0_meter, lpips_tn_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

	for idx, batch in enumerate(val_dataset):

		frame_rate = batch['frame_rate'] # int
		video = batch["video"].to(weight_dtype).to(accelerator.device, non_blocking=True) # (T, C, H, W)
		path = batch['path']
		
		formatted_path = format_save_path(args, path)
		out_file_path = os.path.join(val_save_dir, formatted_path)
		ext = '.'+path.split('.')[-1]
		if any(x.startswith(formatted_path.replace(ext, '')) for x in os.listdir(val_save_dir)):
			print(f'{out_file_path} already exists. Skipping.')
			continue

		fov_x = random.uniform(args.fov_x_min, args.fov_x_max) # (1,)
		fov_y = random.uniform(args.fov_y_min, args.fov_y_max) # (1,)
		hw_ratio = math.tan(math.radians(fov_y) / 2) / math.tan(math.radians(fov_x) / 2)

		pitches, yaws, rolls = get_rpy(frame_rate, timesteps=args.num_frames)

		mask, generate_frames = sample_svd(args, accelerator, pipeline, weight_dtype, 
											fov_x = fov_x, hw_ratio = hw_ratio,
											roll = rolls, pitch = pitches, yaw = yaws,
											out_file_path=out_file_path,
											conditional_video=video,
											noise_aug_strength=args.noise_aug_strength,
											fps = frame_rate,
											num_inference_steps=args.num_inference_steps,
											width=args.width, height=args.height,
											guidance_scale=args.guidance_scale,
											inference_final_rotation=args.inference_final_rotation,
											blend_decoding_ratio=args.blend_decoding_ratio,
											extended_decoding=args.extended_decoding,
											noise_conditioning=args.noise_conditioning,
											rotation_during_inference=args.rotation_during_inference,
											return_for_metrics=True,
											equirectangular_input=args.equirectangular_input,
											) # (T, 1, H, W), (1, C, T, H, W)
		generate_frames = generate_frames.squeeze(0).permute(1, 0, 2, 3) # (T, C, H, W)
		# mask_stack = (torch.sum(mask, dim=0) > 0).repeat(3, 1, 1) # (C=3, H, W)

		rots = [{'roll': roll, 'pitch': pitch, 'yaw': yaw} for roll, pitch, yaw in zip(rolls, pitches, yaws)]

		ground_truth_t0_multiview = equi2pers(video[0].repeat(args.num_frames, 1, 1, 1), rots = rots,
											  fov_x=fov_x, width=480, height=int(480*hw_ratio), z_down=True)
		generated_frames_t0_multiview = equi2pers(generate_frames[0].repeat(args.num_frames, 1, 1, 1), rots = rots,
												   fov_x=fov_x, width=480, height=int(480*hw_ratio), z_down=True)

		ground_truth_tn_multiview = equi2pers(video[-1].repeat(args.num_frames, 1, 1, 1), rots = rots,
											  fov_x=fov_x, width=480, height=int(480*hw_ratio), z_down=True)
		
		generated_frames_tn_multiview = equi2pers(generate_frames[-1].repeat(args.num_frames, 1, 1, 1), rots = rots,
												   fov_x=fov_x, width=480, height=int(480*hw_ratio), z_down=True)
		

		psnr_t0 = video_psnr((generated_frames_t0_multiview + 1) / 2, (ground_truth_t0_multiview + 1) / 2)
		psnr_tn = video_psnr((generated_frames_tn_multiview + 1) / 2, (ground_truth_tn_multiview + 1) / 2)
		lpips_t0 = lpips_fn((generated_frames_t0_multiview + 1) / 2, (ground_truth_t0_multiview + 1) / 2).mean()
		lpips_tn = lpips_fn((generated_frames_tn_multiview + 1) / 2, (ground_truth_tn_multiview + 1) / 2).mean()

		print(f'PSNR T0: {psnr_t0.item()}')
		print(f'PSNR TN: {psnr_tn.item()}')
		print(f'LPIPS T0: {lpips_t0.item()}')
		print(f'LPIPS TN: {lpips_tn.item()}')

		psnr_t0_meter.update(psnr_t0.item(), 1)
		psnr_tn_meter.update(psnr_tn.item(), 1)
		lpips_t0_meter.update(lpips_t0.item(), 1)
		lpips_tn_meter.update(lpips_tn.item(), 1)

		