import argparse
import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator
from tqdm import tqdm

from conversions.mapper import Mappers
from conversions.rotations import rot, rot_to_euler_xyz
from data.dataloader import TAPVid360Dataset
from data.dataloader_lasot import LaSOTDataset
from exploration.argus_src.src import focal2fov, rotation_matrix_to_euler, pers2equi_batch
from metrics.tapvid360_metrics import compute_metrics
from runners.run_argus_cotracker import _get_latents, _decode_pred_eq_frames, _resize_eq_frames
from runners.vis_utils import overlay_mask_over_image
from runners.model_utils import get_models, decode_latents, get_360_latents_model, SAMRunner
from runners.utils import get_object_centred_persp_imgs, get_dataset, overlay_orig_persp_on_pred_eq
from runners.bbox_utils import xywh_to_xyxy, get_bbox_xyxy_from_points, bbox_iou

# TODO
# make proper output format
# allow to work for bbox or points according to dataste
# run with sam2 and see what ious that gets
# make the original perspective image overlay on the equirectangular


JUST_GET_LATENTS = False
JUST_GET_PRED_EQ_FRAMES = False
USE_GT_POSES = True
SKIP_IF_EXISTS = False
ONLY_IF_LATENTS_EXIST = True

EQ_MAKE_METHOD = "standard"


@dataclasses.dataclass
class Settings:
    ds_name = "tapvid360-10k"
    # specific_video_names = "-73Nyd_QcNc_clip_0003/0"
    specific_video_names = ["AfXjCJcexSI_clip_0010/0"]

    unet_path = "/home/userfs/f/fgch500/storage/pretrained_models/video_generation/argus"
    # CSGPU
    ds_root = Path("/home/userfs/f/fgch500/storage/shared/TAPVid360/data")
    sam_checkpoint = Path(
        "/shared/storage/cs/staffstore/fgch500/pretrained_models/segmentation") / "sam2" / "sam2_hiera_large.pt"
    out_root = Path("/home/userfs/f/fgch500/storage/shared/WhatGoesAround")

    force_get_latents = True
    eq_height = 512  # TODO watchout this is currently hardcoded
    weight_dtype = torch.float32
    decode_chunk_size = 5  # Should be able to be bigger for larger gpu's
    offload_to_cpu = True


@torch.no_grad()
def main(device, settings):
    out_root = settings.out_root / "argus_cotracker_outputs" / settings.ds_name
    if settings.ds_name == "tapvid360-10k":
        out_root = out_root / f"gt_poses-{USE_GT_POSES}"
    out_root.mkdir(exist_ok=True, parents=True)
    results_dir = out_root / "results"
    results_dir.mkdir(exist_ok=True)
    metrics_dir = out_root / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    dataset, dl = get_dataset(settings.ds_root, settings.ds_name, settings.specific_video_names)

    accelerator = Accelerator(mixed_precision='no')
    if not JUST_GET_LATENTS:
        vae, sam_pred_video, sam_pred_image, cotracker = get_models(settings.sam_checkpoint, accelerator,
                                                                    accelerator.device, debug_skip_vae=False)
        vae = vae.eval()
    for data in tqdm(dl):
        if SKIP_IF_EXISTS and (results_dir / f"{data.seq_name[0].replace('/', '-')}.pth").exists():
            print(f"Skipping {data.seq_name[0]} as results already exist")
            continue
        data.video = data.video.to(accelerator.device, non_blocking=True)
        normed_vid = (data.video.float() * 2) - 1
        R_w2c, conditional_video_equi, fov_x, mask = get_poses_and_equirectangular_inputs(accelerator, data, normed_vid,
                                                                                          settings)
        latents_dir = settings.out_root / "argus_feats" / settings.ds_name / data.seq_name[
            0] / f"gt_poses-{USE_GT_POSES}" / f"fov_x-{fov_x:.2f}" / "latents.pth"
        if ONLY_IF_LATENTS_EXIST and not (latents_dir.exists()):
            print(f"Skipping {data.seq_name[0]} as latents dont exist")
            continue
        argus_latents = _get_latents(normed_vid[0].clone(), conditional_video_equi, mask, accelerator, settings,
                                     latents_dir)
        if JUST_GET_LATENTS:
            continue
        # TODO the larger i make this the better quality image that gets produced but obviously slower
        # settings.eq_height = 512 #2048
        mapper = Mappers(data.video.shape[-1], data.video.shape[-2], settings.eq_height, fov_x=fov_x)
        with torch.no_grad():
            orig_pred_eq_frames_torch = _decode_pred_eq_frames(argus_latents, settings, vae)
        orig_pred_eq_frames_torch = _resize_eq_frames(orig_pred_eq_frames_torch,
                                                      new_shape=(settings.eq_height, settings.eq_height * 2))

        vid_out_dir = out_root / "debugs" / data.seq_name[0]
        vid_out_dir.mkdir(exist_ok=True, parents=True)
        pred_eq_frames_torch = overlay_orig_persp_on_pred_eq(data, mapper, orig_pred_eq_frames_torch, R_w2c)
        pred_eq_frames_out_dir = vid_out_dir / "pred_eq_frames"
        pred_eq_frames_out_dir.mkdir(exist_ok=True)
        for i, pred_eq_frame_torch in enumerate(pred_eq_frames_torch):
            Image.fromarray(pred_eq_frame_torch.cpu().numpy().astype(np.uint8)).save(
                pred_eq_frames_out_dir / f"{i}.jpg")

        if settings.offload_to_cpu:
            orig_pred_eq_frames_torch = orig_pred_eq_frames_torch.cpu()

        del normed_vid
        del conditional_video_equi
        del mask
        del argus_latents
        del mapper
        del orig_pred_eq_frames_torch
        del pred_eq_frames_torch
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_poses_and_equirectangular_inputs(accelerator, data, normed_vid, settings):
    if USE_GT_POSES:
        assert settings.ds_name == "tapvid360-10k", "GT poses only supported for tapvid360 for now"
        settings.eq_height = data.equirectangular_height[0]
        fov_x = data.fov_x[0]
        mapper = Mappers(data.video.shape[-1], data.video.shape[-2], settings.eq_height, fov_x=fov_x)
        conditional_video_equi, mask = mapper.image.perspective_image_to_equirectangular(
            data.video[0].permute(0, 2, 3, 1).cpu(), data.rotations[0], to_uint=False)
        conditional_video_equi = conditional_video_equi.clip(0, 1)
        conditional_video_equi = conditional_video_equi * 2 - 1
        conditional_video_equi = conditional_video_equi.permute(0, 3, 1, 2)
        mask = mask.unsqueeze(1)
        # Argus needs of shape 512, 1024
        conditional_video_equi = torch.nn.functional.interpolate(conditional_video_equi, size=(512, 1024),
                                                                 mode='bilinear', align_corners=False)
        mask = torch.nn.functional.interpolate(mask.float(), size=(512, 1024), mode='nearest').bool()
        R_w2c = data.rotations[0]
    else:
        # gt_roll, gt_pitch, gt_yaw = rot_to_euler_xyz(data.rotations[0], degrees=True)
        fov_x, pitches, rolls, yaws = _get_poses(data, settings.poses_root / "estimated_poses" / settings.ds_name)
        conditional_video_equi, mask = pers2equi_batch(normed_vid[0].to(torch.float32), fov_x=fov_x,
                                                       roll=rolls, pitch=pitches, yaw=yaws,
                                                       width=settings.eq_height * 2, height=settings.eq_height,
                                                       device=accelerator.device,
                                                       return_mask=True)  # (T, C, H, W)
        R_w2c = rot(alpha=torch.from_numpy(rolls), beta=torch.from_numpy(pitches), gamma=torch.from_numpy(yaws),
                    degrees=True)
    return R_w2c, conditional_video_equi, fov_x, mask


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    settings = Settings()
    main(device, settings)
