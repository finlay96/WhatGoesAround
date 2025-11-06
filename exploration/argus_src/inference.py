from dataclasses import dataclass, field
from natsort import natsorted
from pathlib import Path
from typing import List, Optional

from accelerate import Accelerator
import cv2
from diffusers import UNetSpatioTemporalConditionModel, AutoencoderKLTemporalDecoder
import numpy as np
from PIL import Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from conversions.mapper import Mappers
from exploration.argus_src.sampling_svd import sample_svd
from exploration.argus_src.src import StableVideoDiffusionPipelineCustom

# unet_path = "/media/finlay/BigDaddyDrive/PretrainedModels/video_generation/argus"
unet_path = "/home/userfs/f/fgch500/storage/pretrained_models/video_generation/argus"


def get_models(args, accelerator, weight_dtype):
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        pretrained_model_name_or_path=unet_path,
        subfolder="unet",
    )

    # feature_extractor = CLIPImageProcessor.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    # )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    pipeline = StableVideoDiffusionPipelineCustom.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=accelerator.unwrap_model(unet),
        image_encoder=accelerator.unwrap_model(image_encoder),
        vae=accelerator.unwrap_model(vae),
        revision=args.revision,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)

    return pipeline


@dataclass
class SVDArguments:
    """
    A class to hold all configuration arguments for the SVD sampling script.
    """

    # --- Model and Path Arguments ---
    # These arguments specify the paths to models, data, and output folders.
    unet_path: str  # This is a required path to the pretrained U-Net model.
    pretrained_model_name_or_path: str = 'stabilityai/stable-video-diffusion-img2vid'
    revision: Optional[str] = None
    val_base_folder: List[str] = field(default_factory=list)
    val_clip_file: Optional[str] = None
    calibration_cache_path: Optional[str] = None
    cached_motion_npz_path: Optional[str] = None
    val_save_folder: str = 'results'
    cached_motion_path: Optional[str] = None

    # --- Data Arguments ---
    # These arguments control the input data's properties, such as size and format.
    dataset_size: Optional[int] = None
    width: int = 1024
    height: int = 512
    equirectangular_input: bool = False
    crop_center_then_calibrate: bool = False
    narrow: bool = False
    width_narrow: int = 512
    height_narrow: int = 512

    # --- Camera and FOV Arguments ---
    # These arguments relate to the virtual camera's field of view (FOV) and orientation.
    fov_x_min: float = 90.0
    fov_x_max: float = 90.0
    fov_y_min: float = 90.0
    fov_y_max: float = 90.0
    fixed_fov: Optional[float] = None
    fixed_rpy: bool = False
    yaw_start: float = 0.0
    noisy_rpy: float = 0.0

    # --- Sampling and Frame Arguments ---
    # These arguments define the video's temporal properties, like frame count and rate.
    frame_rate: Optional[int] = None
    frame_interval: int = 6
    fixed_start_frame: bool = False
    full_sampling: bool = False
    num_frames: int = 25
    num_frames_batch: int = 25
    blend_frames: int = 0

    # --- Inference Arguments ---
    # Core parameters for the diffusion model's generation process.
    num_inference_steps: int = 50
    motion_bucket_id: int = 127
    noise_aug_strength: float = 0.02
    guidance_scale: float = 1.0
    decode_chunk_size: int = 10
    noise_conditioning: bool = False

    # --- Advanced Generation Arguments ---
    # Special techniques used during or after inference.
    rotation_during_inference: bool = False
    post_rotation: bool = False
    inference_final_rotation: int = 0
    blend_decoding_ratio: int = 4
    extended_decoding: bool = False
    replacement_sampling: bool = False
    calibration_cache_path_add_prefix: bool = False

    # --- Calibration Arguments ---
    # Settings for the camera motion prediction and calibration process.
    predict_camera_motion: bool = False
    dense_calibration: bool = False
    calibration_img_size: int = 512


def rotation_matrix_to_euler_xyz(R: torch.Tensor, degrees: bool = True) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts Roll, Pitch, and Yaw angles from a 3x3 rotation matrix
    created with an R_x @ R_y @ R_z convention.

    Args:
        R (torch.Tensor): The 3x3 rotation matrix.
        degrees (bool): If True, returns angles in degrees. Otherwise, in radians.

    Returns:
        A tuple containing (roll, pitch, yaw).
    """
    # Epsilon for float comparisons
    epsilon = 1e-6

    # Check for the gimbal lock case
    # This happens when R[0, 2] is very close to +1 or -1
    if torch.abs(R[0, 2]) > 1.0 - epsilon:
        # Gimbal lock: Pitch is at +/- 90 degrees
        pitch_rad = torch.asin(R[0, 2])

        # We can't distinguish roll and yaw. By convention, set roll to 0.
        roll_rad = torch.tensor(0.0, device=R.device, dtype=R.dtype)

        # Calculate yaw from the other elements
        yaw_rad = torch.atan2(R[1, 0], R[1, 1])

    else:
        # The general case
        # Pitch (around Y-axis) can be uniquely determined from R[0, 2] = sin(pitch)
        pitch_rad = torch.asin(R[0, 2])

        # Roll (around X-axis) from R[1, 2] = -sin(roll)*cos(pitch) and R[2, 2] = cos(roll)*cos(pitch)
        roll_rad = torch.atan2(-R[1, 2], R[2, 2])

        # Yaw (around Z-axis) from R[0, 1] = -cos(pitch)*sin(yaw) and R[0, 0] = cos(pitch)*cos(yaw)
        yaw_rad = torch.atan2(-R[0, 1], R[0, 0])

    if degrees:
        # Convert radians to degrees for the final output
        roll = torch.rad2deg(roll_rad)
        pitch = torch.rad2deg(pitch_rad)
        yaw = torch.rad2deg(yaw_rad)
    else:
        roll, pitch, yaw = roll_rad, pitch_rad, yaw_rad

    return roll, pitch, yaw


def resize(frame, calibration_img_size=512):
    if calibration_img_size is not None:  # return the longest side to be calibration_img_size
        h, w = frame.shape[:2]
        if h > w:
            new_h = calibration_img_size
            new_w = int(w * calibration_img_size / h)
        else:
            new_w = calibration_img_size
            new_h = int(h * calibration_img_size / w)
        return cv2.resize(frame, (new_w, new_h))
    else:
        return frame


def main(device):
    max_frames = 25
    args = SVDArguments(unet_path=unet_path)
    weight_dtype = torch.float32
    accelerator = Accelerator(mixed_precision='no')
    pipeline = get_models(args, accelerator, weight_dtype)

    root_dir = Path("/home/userfs/f/fgch500/tracking/WhatGoesAround/exploration")
    # root_dir = Path("/home/finlay/part_time_PHD_stuff/tracking/WhatGoesAround/exploration")
    vid_name = "5CQoP-G5fx8_clip_0008_0"
    frames = [Image.open(fp) for fp in
              natsorted(list((root_dir / f"tapvid360/{vid_name[:-2]}/{vid_name[-1]}/perspective_frames").glob("*")))]
    frames = frames[:max_frames]
    frames = np.array(frames)

    # vid_loader = DecordVideoReader(root_dir / f"tapvid360/{vid_name[:-2]}/{vid_name[-1]}/data.pt")
    # frames = vid_loader.load_frames()
    tap_data = torch.load(root_dir / f"tapvid360/{vid_name[:-2]}/{vid_name[-1]}/data.pt")
    gt_uv_coords = tap_data["coords"]
    gt_rot_mats = tap_data["rot_matrices"][:len(frames)]

    mapper = Mappers(320, 240, 1024, fov_x=75.18)
    eq_frames, eq_mask = mapper.image.perspective_image_to_equirectangular(torch.from_numpy(frames), gt_rot_mats)
    eq_frames = eq_frames.numpy()
    # rolls, pitches, yaws = [], [], []
    # for gt_rot_mat in gt_rot_mats:
    #     r, p, y = rotation_matrix_to_euler_xyz(gt_rot_mat)
    #     rolls.append(r.item())
    #     pitches.append(p.item())
    #     yaws.append(y.item())
    # rolls = np.stack(rolls)[:len(frames)]
    # pitches = np.stack(pitches)[:len(frames)]
    # yaws = np.stack(yaws)[:len(frames)]

    # TODO these rolls picth yaw are wrong somehow makes for some bad outputs

    frame_rate = 3.0
    out_dir_path = root_dir / "argus_outputs"
    out_dir_path.mkdir(exist_ok=True)
    out_file_path = out_dir_path / "try_tap_on_360_output.mp4"
    fov_x = 75.18

    rz_frames = np.stack([resize(f) for f in frames])
    video = torch.from_numpy(rz_frames).permute(0, 3, 1, 2).float() / 127.5 - 1
    video = video.to(device)

    eq_frames = torch.from_numpy(eq_frames).permute(0, 3, 1, 2).float() / 127.5 - 1
    eq_frames = eq_frames.to(device)

    eq_mask = eq_mask.to(device).unsqueeze(1)

    # TODO DEBUG
    # eq_mask = None
    # eq_frames = None
    sample_svd(args, accelerator, pipeline, weight_dtype,
               fov_x=fov_x, hw_ratio=None,
               eq_frames=eq_frames,
               eq_mask=eq_mask,
               # roll=rolls, pitch=pitches, yaw=yaws,
               out_file_path=str(out_file_path),
               conditional_video=video,
               noise_aug_strength=args.noise_aug_strength,
               fps=frame_rate,
               decode_chunk_size=args.decode_chunk_size,
               num_inference_steps=args.num_inference_steps,
               width=args.width, height=args.height,
               guidance_scale=args.guidance_scale,
               inference_final_rotation=args.inference_final_rotation,
               blend_decoding_ratio=args.blend_decoding_ratio,
               extended_decoding=args.extended_decoding,
               noise_conditioning=args.noise_conditioning,
               rotation_during_inference=args.rotation_during_inference,
               equirectangular_input=args.equirectangular_input,
               post_rotation=args.post_rotation,
               replacement_sampling=args.replacement_sampling,
               narrow=args.narrow,
               width_narrow=args.width_narrow, height_narrow=args.height_narrow,
               )


if __name__ == "__main__":
    main(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
