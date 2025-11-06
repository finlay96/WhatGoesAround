from PIL import Image
from diffusers.utils import load_image
from urllib.parse import urlparse
import torch
from einops import rearrange
import cv2
import numpy as np
from torch.nn import functional as F
import timm
import gc
import os
from equilib import equi2pers
import math
from tqdm import tqdm
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from typing import Tuple, Optional
from equilib import equi2pers
from typing import Union
from imageio import mimsave

def resize_mask(mask, target_shape, seperate_first_frame=True):
	# mask: B, F, 1, H, W
	# target_shape: (f, h, w)
	# output: B, f, 1, h, w

	mask = mask.permute(0, 2, 1, 3, 4)
	f, h, w = target_shape

	if seperate_first_frame:
		first_frame_resized = F.interpolate(
			mask[:, :, 0, :, :],
			size=(h, w),
			mode='nearest',
			# mode='bilinear',
			# align_corners=False
		).unsqueeze(2) # B, 1, 1, h, w
		
		if f > 1:
			remaining_frames_resized = F.interpolate(
				mask[:, :, 1:, :, :],
				size=(f - 1, h, w),
				mode='nearest',
				# mode='trilinear',
				# align_corners=False
			) # B, 1, f-1, h, w
			resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
		else:
			resized_mask = first_frame_resized
	else:
		resized_mask = F.interpolate(
			mask,
			size=(f, h, w),
			mode='nearest',
			# mode='trilinear',
			# align_corners=False
		)

	return resized_mask.permute(0, 2, 1, 3, 4) # B, f, 1, h, w

def download_image(url):
	original_image = (
		lambda image_url_or_path: load_image(image_url_or_path)
		if urlparse(image_url_or_path).scheme
		else Image.open(image_url_or_path).convert("RGB")
	)(url)
	return original_image

# copy from https://github.com/crowsonkb/k-diffusion.git
def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32, min_val=1e-5):
	"""Draws samples from an lognormal distribution."""
	u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
	return torch.distributions.LogNormal(loc, scale).icdf(u).clamp(min=min_val)

# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
	h, w = input.shape[-2:]
	factors = (h / size[0], w / size[1])

	# First, we have to determine sigma
	# Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
	sigmas = (
		max((factors[0] - 1.0) / 2.0, 0.001),
		max((factors[1] - 1.0) / 2.0, 0.001),
	)

	# Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
	# https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
	# But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
	ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

	# Make sure it is odd
	if (ks[0] % 2) == 0:
		ks = ks[0] + 1, ks[1]

	if (ks[1] % 2) == 0:
		ks = ks[0], ks[1] + 1

	input = _gaussian_blur2d(input, ks, sigmas)

	output = torch.nn.functional.interpolate(
		input, size=size, mode=interpolation, align_corners=align_corners)
	return output


def _compute_padding(kernel_size):
	"""Compute padding tuple."""
	# 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
	# https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
	if len(kernel_size) < 2:
		raise AssertionError(kernel_size)
	computed = [k - 1 for k in kernel_size]

	# for even kernels we need to do asymmetric padding :(
	out_padding = 2 * len(kernel_size) * [0]

	for i in range(len(kernel_size)):
		computed_tmp = computed[-(i + 1)]

		pad_front = computed_tmp // 2
		pad_rear = computed_tmp - pad_front

		out_padding[2 * i + 0] = pad_front
		out_padding[2 * i + 1] = pad_rear

	return out_padding


def _filter2d(input, kernel):
	# prepare kernel
	b, c, h, w = input.shape
	tmp_kernel = kernel[:, None, ...].to(
		device=input.device, dtype=input.dtype)

	tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

	height, width = tmp_kernel.shape[-2:]

	padding_shape: list[int] = _compute_padding([height, width])
	input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

	# kernel and input tensor reshape to align element-wise or batch-wise params
	tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
	input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

	# convolve the tensor with the kernel.
	output = torch.nn.functional.conv2d(
		input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

	out = output.view(b, c, h, w)
	return out


def _gaussian(window_size: int, sigma):
	if isinstance(sigma, float):
		sigma = torch.tensor([[sigma]])

	batch_size = sigma.shape[0]

	x = (torch.arange(window_size, device=sigma.device,
		 dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

	if window_size % 2 == 0:
		x = x + 0.5

	gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

	return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
	if isinstance(sigma, tuple):
		sigma = torch.tensor([sigma], dtype=input.dtype)
	else:
		sigma = sigma.to(dtype=input.dtype)

	ky, kx = int(kernel_size[0]), int(kernel_size[1])
	bs = sigma.shape[0]
	kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
	kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
	out_x = _filter2d(input, kernel_x[..., None, :])
	out = _filter2d(out_x, kernel_y[..., None])

	return out

def read_video(video_path, frame_interval=1, max_frames=None, size=None):
	cap = cv2.VideoCapture(video_path)
	fps = cap.get(cv2.CAP_PROP_FPS)
	frames = []
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		if size is not None:
			frame = cv2.resize(frame, size)
		frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		if max_frames is not None and len(frames) >= max_frames:
			break
		for _ in range(frame_interval - 1):
			cap.grab()
	cap.release()
	frames = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float() / 127.5 - 1
	return frames, fps / frame_interval


def export_to_video(video_frames, output_video_path, fps):
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	h, w, _ = video_frames[0].shape
	video_writer = cv2.VideoWriter(
		output_video_path, fourcc, fps=fps, frameSize=(w, h))
	for i in range(len(video_frames)):
		img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
		video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
	"""
	Export a list of frames to a GIF.

	Args:
	- frames (list): List of frames (as numpy arrays or PIL Image objects).
	- output_gif_path (str): Path to save the output GIF.
	- duration_ms (int): Duration of each frame in milliseconds.

	"""
	# Convert numpy arrays to PIL Images if needed
	pil_frames = [Image.fromarray(frame) if isinstance(
		frame, np.ndarray) else frame for frame in frames]

	pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
					   format='GIF',
					   append_images=pil_frames[1:],
					   save_all=True,
					   duration=500,
					   loop=0)

def tensor_to_vae_latent(t, vae):
	assert len(t.shape) == 5 or len(t.shape) == 4
	is5 = len(t.shape) == 5
	
	if is5:
		video_length = t.shape[1]
		t = rearrange(t, "b f c h w -> (b f) c h w")
	latents = vae.encode(t).latent_dist.sample()
	if is5:
		latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
	latents = latents * vae.config.scaling_factor

	return latents

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def _map_layer_to_idx(backbone, layers, offset=0):
	"""Maps set of layer names to indices of model. Ported from anomalib

	Returns:
		Feature map extracted from the CNN
	"""
	idx = []
	features = timm.create_model(
		backbone,
		pretrained=False,
		features_only=False,
		exportable=True,
	)
	for i in layers:
		try:
			idx.append(list(dict(features.named_children()).keys()).index(i) - offset)
		except ValueError:
			raise ValueError(
				f"Layer {i} not found in model {backbone}. Select layer from {list(dict(features.named_children()).keys())}. The network architecture is {features}"
			)
	return idx


def get_perceptual_loss(pixel_values, fmap, timm_model, timm_model_resolution, timm_model_normalization):
	img_timm_model_input = timm_model_normalization(F.interpolate(pixel_values, timm_model_resolution))
	fmap_timm_model_input = timm_model_normalization(F.interpolate(fmap, timm_model_resolution))

	if pixel_values.shape[1] == 1:
		# handle grayscale for timm_model
		img_timm_model_input, fmap_timm_model_input = (
			t.repeat(1, 3, 1, 1) for t in (img_timm_model_input, fmap_timm_model_input)
		)

	img_timm_model_feats = timm_model(img_timm_model_input)
	recon_timm_model_feats = timm_model(fmap_timm_model_input)
	perceptual_loss = F.mse_loss(img_timm_model_feats[0], recon_timm_model_feats[0])
	for i in range(1, len(img_timm_model_feats)):
		perceptual_loss += F.mse_loss(img_timm_model_feats[i], recon_timm_model_feats[i])
	perceptual_loss /= len(img_timm_model_feats)
	return perceptual_loss


def grad_layer_wrt_loss(loss, layer):
	return torch.autograd.grad(
		outputs=loss,
		inputs=layer,
		grad_outputs=torch.ones_like(loss),
		retain_graph=True,
	)[0].detach()


def gradient_penalty(images, output, weight=10):
	gradients = torch.autograd.grad(
		outputs=output,
		inputs=images,
		grad_outputs=torch.ones(output.size(), device=images.device),
		create_graph=True,
		retain_graph=True,
		only_inputs=True,
	)[0]
	bsz = gradients.shape[0]
	gradients = torch.reshape(gradients, (bsz, -1))
	return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def log_grad_norm(model, accelerator, global_step):
	for name, param in model.named_parameters():
		if param.grad is not None:
			grads = param.grad.detach().data
			grad_norm = (grads.norm(p=2) / grads.numel()).item()
			accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)

def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
	tw = tgt_width
	th = tgt_height
	h, w = src
	r = h / w
	if r > (th / tw):
		resize_height = th
		resize_width = int(round(th / h * w))
	else:
		resize_width = tw
		resize_height = int(round(tw / w * h))

	crop_top = int(round((th - resize_height) / 2.0))
	crop_left = int(round((tw - resize_width) / 2.0))

	return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def get_3d_rotary_pos_embed(
	embed_dim,
	crops_coords,
	grid_size,
	temporal_size,
	theta: int = 10000,
	use_real: bool = True,
	grid_type: str = "linspace", # linspace or slice or equirectangular
	max_size: Optional[Tuple[int, int]] = None,
	width_rotation_degree=0,
	device: Optional[torch.device] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
	"""
	RoPE for video tokens with 3D structure.

	Args:
	embed_dim: (`int`):
		The embedding dimension size, corresponding to hidden_size_head.
	crops_coords (`Tuple[int]`):
		The top-left and bottom-right coordinates of the crop.
	grid_size (`Tuple[int]`):
		The grid size of the spatial positional embedding (height, width).
	temporal_size (`int`):
		The size of the temporal dimension.
	theta (`float`):
		Scaling factor for frequency computation.
	grid_type (`str`):
		Whether to use "linspace" or "slice" or "equirectangular" to compute grids. 

	Returns:
		`torch.Tensor`: positional embedding with shape `(temporal_size * grid_size[0] * grid_size[1], embed_dim/2)`.
	"""
	if use_real is not True:
		raise ValueError(" `use_real = False` is not currently supported for get_3d_rotary_pos_embed")

	if grid_type == "linspace":
		start, stop = crops_coords
		grid_size_h, grid_size_w = grid_size
		grid_h = torch.linspace(
			start[0], stop[0] * (grid_size_h - 1) / grid_size_h, grid_size_h, device=device, dtype=torch.float32
		)
		grid_w = torch.linspace(
			start[1], stop[1] * (grid_size_w - 1) / grid_size_w, grid_size_w, device=device, dtype=torch.float32
		)
		grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32)
		grid_t = torch.linspace(
			0, temporal_size * (temporal_size - 1) / temporal_size, temporal_size, device=device, dtype=torch.float32
		)
	elif grid_type == "slice":
		max_h, max_w = max_size
		grid_size_h, grid_size_w = grid_size
		grid_h = torch.arange(max_h, device=device, dtype=torch.float32)
		grid_w = torch.arange(max_w, device=device, dtype=torch.float32)
		grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32)
	elif grid_type == "equirectangular":
		grid_size_h, grid_size_w = grid_size
		grid_h = torch.linspace(-np.pi / 2, np.pi / 2, grid_size_h, device=device, dtype=torch.float32) # -pi/2 to pi/2
		grid_w = torch.linspace(-np.pi, np.pi, grid_size_w, device=device, dtype=torch.float32) # -pi to pi
		grid_h = (torch.sin(grid_h) + 1) / 2 # 0 -> 1 (from north pole to south pole)
		grid_w = (torch.sin(grid_w) + 1) / 2 # 1 -> 0 -> 1 (left & right are the same)
		grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32)
	else:
		raise ValueError("Invalid value passed for `grid_type`.")

	# Compute dimensions for each axis
	dim_t = embed_dim // 4
	dim_h = embed_dim // 8 * 3
	dim_w = embed_dim // 8 * 3

	# Temporal frequencies
	freqs_t = get_1d_rotary_pos_embed(dim_t, grid_t, theta=theta, use_real=True)
	# Spatial frequencies for height and width
	freqs_h = get_1d_rotary_pos_embed(dim_h, grid_h, theta=theta, use_real=True)
	freqs_w = get_1d_rotary_pos_embed(dim_w, grid_w, theta=theta, use_real=True)

	# BroadCast and concatenate temporal and spaial frequencie (height and width) into a 3d tensor
	def combine_time_height_width(freqs_t, freqs_h, freqs_w):
		freqs_t = freqs_t[:, None, None, :].expand(
			-1, grid_size_h, grid_size_w, -1
		)  # temporal_size, grid_size_h, grid_size_w, dim_t
		freqs_h = freqs_h[None, :, None, :].expand(
			temporal_size, -1, grid_size_w, -1
		)  # temporal_size, grid_size_h, grid_size_2, dim_h
		freqs_w = freqs_w[None, None, :, :].expand(
			temporal_size, grid_size_h, -1, -1
		)  # temporal_size, grid_size_h, grid_size_2, dim_w

		freqs = torch.cat(
			[freqs_t, freqs_h, freqs_w], dim=-1
		)  # temporal_size, grid_size_h, grid_size_w, (dim_t + dim_h + dim_w)
		freqs = freqs.view(
			temporal_size * grid_size_h * grid_size_w, -1
		)  # (temporal_size * grid_size_h * grid_size_w), (dim_t + dim_h + dim_w)
		return freqs

	t_cos, t_sin = freqs_t  # both t_cos and t_sin has shape: temporal_size, dim_t
	h_cos, h_sin = freqs_h  # both h_cos and h_sin has shape: grid_size_h, dim_h
	w_cos, w_sin = freqs_w  # both w_cos and w_sin has shape: grid_size_w, dim_w

	if grid_type == "slice":
		t_cos, t_sin = t_cos[:temporal_size], t_sin[:temporal_size]
		h_cos, h_sin = h_cos[:grid_size_h], h_sin[:grid_size_h]
		w_cos, w_sin = w_cos[:grid_size_w], w_sin[:grid_size_w]

	cos = combine_time_height_width(t_cos, h_cos, w_cos)
	sin = combine_time_height_width(t_sin, h_sin, w_sin)
	return cos, sin


def prepare_rotary_positional_embeddings(
	height: int,
	width: int,
	num_frames: int,
	vae_scale_factor_spatial: int = 8,
	patch_size: int = 2,
	patch_size_t: int = 1,
	attention_head_dim: int = 64,
	sample_width: int = 300,
	sample_height: int = 300,
	device: Optional[torch.device] = None,
	grid_type: str = "slice", # linspace or slice or equirectangular, currently only slice and equirectangular are supported
) -> Tuple[torch.Tensor, torch.Tensor]:
	
	grid_height = height // (vae_scale_factor_spatial * patch_size)
	grid_width = width // (vae_scale_factor_spatial * patch_size)
	base_size_width = sample_height // patch_size
	base_size_height = sample_width // patch_size

	p_t = patch_size_t
	base_num_frames = (num_frames + p_t - 1) // p_t

	if grid_type == "slice":
		freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
			embed_dim=attention_head_dim,
			crops_coords=None,
			grid_size=(grid_height, grid_width),
			temporal_size=base_num_frames,
			grid_type='slice',
			max_size=(base_size_height, base_size_width),
		)
	elif grid_type == "equirectangular":
		freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
			embed_dim=attention_head_dim,
			crops_coords=None,
			grid_size=(grid_height, grid_width),
			temporal_size=base_num_frames,
			grid_type='equirectangular', # omits max_size
		)
	else:
		raise ValueError("Invalid value passed for `grid_type`.")

	freqs_cos = freqs_cos.to(device=device)
	freqs_sin = freqs_sin.to(device=device)
	return freqs_cos, freqs_sin

def reset_memory(device: str | torch.device) -> None:
	gc.collect()
	torch.cuda.empty_cache()
	torch.cuda.reset_peak_memory_stats(device)
	torch.cuda.reset_accumulated_memory_stats(device)


def print_memory(device: str | torch.device) -> None:
	memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
	max_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
	max_memory_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
	print(f"{memory_allocated=:.3f} GB")
	print(f"{max_memory_allocated=:.3f} GB")
	print(f"{max_memory_reserved=:.3f} GB")

def focal2fov(focal_length, width):
	"""
	Convert focal length to field of view in degrees.
	"""
	return 2 * np.arctan(width / (2 * focal_length)) * 180 / np.pi

def get_rotating_demo(video_path, 
					  output_paths: list[str], 
					  fov: float = 120., 
					  rotation_angles: list[float] = [360.], 
					  width=640, height=480,
					  device="cuda",):
	
	cap = cv2.VideoCapture(video_path)
	num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = cap.get(cv2.CAP_PROP_FPS)
	frames = []
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		frames.append(frame)
	cap.release()
	frames = torch.from_numpy(np.array(frames)).to(device).permute(0, 3, 1, 2).float() / 255

	for (output_path, rotation_angle) in zip(output_paths, rotation_angles):

		degrees_per_frame = rotation_angle / (num_frames - 1)
		rots = [{'roll': 0, 'pitch': 0, 'yaw': math.radians(i) * degrees_per_frame} for i in range(num_frames)]
		
		pers_frames = equi2pers(frames, rots=rots, fov_x=fov, height=height, width=width)
		pers_frames = (pers_frames * 255).cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
		pers_frames = pers_frames[:, :, :, ::-1] # BGR -> RGB

		mimsave(output_path, pers_frames, fps=fps)

def rotation_matrix_to_euler(R, z_down=True):
	"""
	Convert a rotation matrix to Euler angles (roll, pitch, yaw).
	
	Parameters:
	R : ndarray
		A 3x3 rotation matrix.
		
	Returns:
	roll, pitch, yaw : tuple of float
		The Euler angles in radians.
	"""
	assert R.shape == (3, 3), "Input rotation matrix must be 3x3"

	# Check for gimbal lock
	if np.isclose(R[2, 0], -1.0):
		pitch = np.pi / 2
		yaw = 0
		roll = np.arctan2(R[0, 1], R[0, 2])
	elif np.isclose(R[2, 0], 1.0):
		pitch = -np.pi / 2
		yaw = 0
		roll = np.arctan2(-R[0, 1], -R[0, 2])
	else:
		pitch = np.arcsin(-R[2, 0])
		roll = np.arctan2(R[2, 1], R[2, 2])
		yaw = np.arctan2(R[1, 0], R[0, 0])

	if z_down:
		yaw = -yaw
		pitch = -pitch

	return roll, pitch, yaw

def get_rotation_from_intrinsics(poses, intrinsics):

	convention_rotation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
	convention_inverse = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

	rolls, pitches, yaws = np.zeros(len(poses)), np.zeros(len(poses)), np.zeros(len(poses))
	R1 = poses[0, :3, :3].cpu().numpy()

	for i in range(1, len(poses)):
		R2 = poses[i, :3, :3].cpu().numpy()
		roll, pitch, yaw = rotation_matrix_to_euler(convention_inverse @ R2.T @ R1 @ convention_rotation, z_down=True) # rotation matrix are camera-to-world, cam1 --> cam2 is R2.T @ R1
		rolls[i] = -roll
		pitches[i] = pitch
		yaws[i] = yaw

	focal_length = intrinsics[0, 0, 0].cpu().item() # focal length in pixels
	fov_x = torch.tensor(focal2fov(focal_length, 512), dtype=torch.float32).item()

	return rolls, pitches, yaws, fov_x

def get_rotating_demo_from_rpy(video_path, 
								output_root: str,
								rolls, pitches, yaws, fov_x,
								height: int, width: int,
								first_frame_only=False,
								device="cuda",):

	os.makedirs(output_root, exist_ok=True)
	
	cap = cv2.VideoCapture(video_path)
	fps = cap.get(cv2.CAP_PROP_FPS)
	frames = []

	while True:
		ret, frame = cap.read()
		if not ret:
			break
		if first_frame_only:
			frames = [frame] * len(rolls)
			break
		frames.append(frame)

	if len(frames) >= len(rolls):
		rolls = torch.cat([rolls, torch.tensor([rolls[-1]] * (len(frames) - len(rolls)))])
		pitches = torch.cat([pitches, torch.tensor([pitches[-1]] * (len(frames) - len(pitches)))])
		yaws = torch.cat([yaws, torch.tensor([yaws[-1]] * (len(frames) - len(yaws)))])
	else:
		assert False
 		
	cap.release()
	frames = torch.from_numpy(np.array(frames)).to(device).permute(0, 3, 1, 2).float() / 255

	print(rolls, pitches, yaws, fov_x)

	rots = [{'roll': roll, 'pitch': pitch, 'yaw': yaw} for roll, pitch, yaw in zip(rolls, pitches, yaws)]
	# rots = rots = [{'roll': 0, 'pitch': 0, 'yaw': 0} for i in range(len(poses))]

	perspective_frames = equi2pers(equi=frames, rots=rots, fov_x=fov_x, height=height, width=width, z_down=True)

	# save each frame
	for i, frame in enumerate(perspective_frames):
		cv2.imwrite(os.path.join(output_root, f'{i}.jpg'), (frame * 255).cpu().permute(1, 2, 0).numpy().astype(np.uint8))

	writer = cv2.VideoWriter(os.path.join(output_root, 'output.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
	for frame in perspective_frames:
		writer.write((frame * 255).cpu().permute(1, 2, 0).numpy().astype(np.uint8))
	writer.release()

def rpy_to_rotation_matrix(roll, pitch, yaw, device='cpu'):
	# yaw is clock wise rotation around z-axis
	# pitch is clock wise rotation around y-axis
	# roll is counter clock wise rotation around x-axis
	
	yaw = -yaw
	pitch = -pitch

	yaw_rotation = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0],
								[torch.sin(yaw), torch.cos(yaw), 0],
								[0, 0, 1]], device=device)
	
	pitch_rotation = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)],
								 [0, 1, 0],
								 [-torch.sin(pitch), 0, torch.cos(pitch)]], device=device)

	roll_rotation = torch.tensor([[1, 0, 0],    
							  [0, torch.cos(roll), -torch.sin(roll)],
							  [0, torch.sin(roll), torch.cos(roll)]], device=device)
	
	return yaw_rotation @ pitch_rotation @ roll_rotation

	# return np.dot(np.dot(yaw_rotation, pitch_rotation), roll_rotation)

def rotation_matrix_to_rpy(rotation_matrix):

	pitch = torch.arcsin(-rotation_matrix[:, 2, 0])
	roll = torch.arctan2(rotation_matrix[:, 2, 1] / torch.cos(pitch), rotation_matrix[:, 2, 2] / torch.cos(pitch))
	yaw = torch.arctan2(rotation_matrix[:, 1, 0] / torch.cos(pitch), rotation_matrix[:, 0, 0] / torch.cos(pitch))

	pitch = -pitch
	yaw = -yaw
	
	return roll, pitch, yaw

def get_six_view_angles(roll, pitch, yaw, device='cpu'):

	roll, pitch, yaw = torch.tensor(roll, device=device).float(), torch.tensor(pitch, device=device).float(), torch.tensor(yaw, device=device).float()
	stack_rotation_matrix = torch.tensor([[[1, 0, 0],
											[0, 1, 0],
											[0, 0, 1]],
											[[-1, 0, 0],
											[0, -1, 0],
											[0, 0, 1]],
											[[0, -1, 0],
											[1, 0, 0],
											[0, 0, 1]],
											[[0, 1, 0],
											[-1, 0, 0],
											[0, 0, 1]],
											[[0, 0, -1],
											[0, 1, 0],
											[1, 0, 0]],
											[[0, 0, 1],
											[0, 1, 0],
											[-1, 0, 0]]], device=device).float() # (6, 3, 3)

	rotation_matrix_positive_x = rpy_to_rotation_matrix(roll, pitch, yaw, device=device).unsqueeze(0)  # (1, 3, 3)
	rotation_matrix_stack_other = torch.einsum('ijk,ikl->ijl', rotation_matrix_positive_x, stack_rotation_matrix) # (6, 3, 3)
	rolls, pitches, yaws = rotation_matrix_to_rpy(rotation_matrix_stack_other) # (6,)

	return rolls, pitches, yaws

if __name__ == "__main__":
	from torchvision.transforms.functional import to_tensor, to_pil_image
	image_path = '/home/jovyan/shared/rl897/equi.jpg'
	# 512 x 1024
	image = Image.open(image_path).resize((512, 1024))
	image = to_tensor(image).to(device="cuda")

	roll, pitch, yaw = math.radians(25), math.radians(10), math.radians(30)
	# get six views
	rolls, pitches, yaws = get_six_view_angles(roll, pitch, yaw, device="cuda")

	# get six perspective views
	rots = [{'roll': roll.item(), 'pitch': pitch.item(), 'yaw': yaw.item()} for roll, pitch, yaw in zip(rolls, pitches, yaws)]
	pers_frames = equi2pers(image.unsqueeze(0).repeat(6, 1, 1, 1), rots=rots, fov_x=90, height=480, width=480, z_down=True)

	for (rot, frame) in zip(rots, pers_frames):
		to_pil_image(frame.cpu()).save(f'pers_roll{math.degrees(rot["roll"]):.0f}_pitch{math.degrees(rot["pitch"]):.0f}_yaw{math.degrees(rot["yaw"]):.0f}.jpg')

	# do pers to equi
	import sys
	sys.path.append('.')
	from src import pers2equi_batch

	masked_equi_frames, masks = pers2equi_batch(pers_frames, fov_x=90, 
												roll=rolls, pitch=pitches, yaw=yaws,
												height=512, width=1024, device="cuda", return_mask=True)
	
	canvas = torch.zeros((3, 512, 1024), device="cuda")
	for i, (masked_equi_frame, mask) in enumerate(zip(masked_equi_frames, masks)):
		canvas = canvas + masked_equi_frame * mask
	to_pil_image(canvas.cpu()).save('canvas.jpg')

	# save the masked equi frames
	for i, (masked_equi_frame, mask, roll, pitch, yaw) in enumerate(zip(masked_equi_frames, masks, rolls, pitches, yaws)):
		to_pil_image(masked_equi_frame).save(f'equi_mask_roll{math.degrees(roll):.0f}_pitch{math.degrees(pitch):.0f}_yaw{math.degrees(yaw):.0f}.jpg')

	# video_root = '/home/rl897/360VideoGeneration/experiments-svd/1025-node2-proc8-acc4-filtered-HQFT/checkpoint-29500/unet/real-world-stablization'
	# output_root = '/home/rl897/360VideoGeneration/experiments-svd/1025-node2-proc8-acc4-filtered-HQFT/checkpoint-29500/unet/real-world-stablization-pers'
	# os.makedirs(output_root, exist_ok=True)

	# rotation_angles =  [360, -360, 0]

	# for video_name in tqdm([x for x in os.listdir(video_root) if 'output' in x]):

	# 	video_path = os.path.join(video_root, video_name)
	# 	outfile_name = os.path.join(output_root, video_name)

	# 	output_names = ['clockwise', 'counter_clockwise', 'front']
	# 	output_paths = [outfile_name.replace('.mp4', f'_{output_name}.mp4') for output_name in output_names]

	# 	hw_ratio = float(video_name.split('hw')[1].split('_')[0].split('.mp4')[0])
	# 	height, width = 640, int(640 / hw_ratio)
	# 	fov = float(video_name.split('fov')[1].split('_')[0])

	# 	get_rotating_demo(video_path, output_paths, fov=fov, rotation_angles=rotation_angles, 
	# 					width=width, height=height, device="cuda")

			