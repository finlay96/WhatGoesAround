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
from data.dataloader_lasot import LaSOTDataset
from exploration.argus_src.src import focal2fov, rotation_matrix_to_euler, pers2equi_batch
from runners.vis_utils import overlay_mask_over_image
from runners.model_utils import get_models, decode_latents, get_360_latents_model, SAMRunner
from runners.utils import get_object_centred_persp_imgs, get_dataset, overlay_orig_persp_on_pred_eq
from runners.bbox_utils import xywh_to_xyxy, get_bbox_xyxy_from_points, bbox_iou

# TODO
# make proper output format
# allow to work for bbox or points according to dataste
# run with sam2 and see what ious that gets
# make the original perspective image overlay on the equirectangular


JUST_GET_LATENTS = True
JUST_GET_PRED_EQ_FRAMES = False
USE_GT_POSES = True


def _get_args():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        '--split_file',  # The name of the flag
        type=str,  # The type of value to expect
        default=None,  # The default value if the flag is not provided
        help="Optional: Path to the input file."  # Help message
    )

    return parser.parse_args()


@dataclasses.dataclass
class Settings:
    # ds_root = Path("/media/finlay/BigDaddyDrive/Outputs/tracker/tapvid360")
    ds_name = "tapvid360-10k"
    # specific_video_names = "-73Nyd_QcNc_clip_0003/0"
    specific_video_names = None
    ds_root = Path("/mnt/scratch/projects/cs-dclabs-2019/tapvid360/outputs")

    # ds_name = "lasot_oov"
    # # specific_video_names = "mouse-15/clip_000"
    # # specific_video_names = "kite-5/clip_000"
    # specific_video_names = None
    # ds_root = Path("/home/userfs/f/fgch500/storage/datasets/tracking/object_tracking/LaSOT/custom_out_of_frame_clips")

    out_root = Path("/mnt/scratch/projects/cs-dclabs-2019/WhatGoesAround")

    # ds_root = Path("/media/finlay/BigDaddyDrive/Datasets/tracking/object-tracking/LaSOT/custom_out_of_frame_clips")
    # poses_root = Path("/media/finlay/BigDaddyDrive/Outputs/tracker/whatGoesAround/estimated_poses")
    # unet_path = "/home/userfs/f/fgch500/storage/pretrained_models/video_generation/argus"
    # latents_root = Path("/media/finlay/BigDaddyDrive/Outputs/tracker/whatGoesAround/argus_feats")
    # sam_checkpoint = Path("/home/finlay/Shared/pretrained_models/segmentation") / "sam2" / "sam2_hiera_large.pt"

    poses_root = Path("/mnt/scratch/projects/cs-dclabs-2019/WhatGoesAround/estimated_poses")
    unet_path = "/mnt/scratch/projects/cs-dclabs-2019/WhatGoesAround/pretrained_models/argus"
    latents_root = Path("/mnt/scratch/projects/cs-dclabs-2019/WhatGoesAround/argus_feats")
    sam_checkpoint = Path(
        "/home/userfs/f/fgch500/storage/pretrained_models/segmentation") / "sam2" / "sam2_hiera_large.pt"
    force_get_latents = False
    eq_height = 512  # TODO watchout this is currently hardcoded
    weight_dtype = torch.float32
    decode_chunk_size = 5  # Should be able to be bigger for larger gpu's


def _decode_pred_eq_frames(argus_latents, settings, vae):
    pred_eq_frames = decode_latents(vae.eval(), argus_latents[None], argus_latents.shape[0],
                                    settings.decode_chunk_size, False, 4)
    pred_eq_frames_torch = ((pred_eq_frames[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    # TODO temp example data
    # pred_eq_frames_torch = torch.from_numpy(np.array([Image.open(fn) for fn in natsorted(
    #     list((Path(__file__).parent.parent / "data/eq_frame_examples").glob("*.png")))])).to(device)
    return pred_eq_frames_torch


def _get_poses(data, poses_dir):
    pose_data = torch.load(str(poses_dir / data.seq_name[0]) + ".pth")
    intrinsics, poses, width_resized = pose_data["intrinsics"], pose_data["poses"], pose_data["width_resized"]
    focal_length = intrinsics[0, 0, 0].cpu().item()  # focal length in pixels
    fov_x = torch.tensor(focal2fov(focal_length, width_resized), dtype=torch.float32).item()
    # poses = poses.to(weight_dtype).to(accelerator.device, non_blocking=True)
    convention_rotation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    convention_inverse = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    rolls, pitches, yaws = np.zeros(len(poses)), np.zeros(len(poses)), np.zeros(len(poses))
    R1 = poses[0, :3, :3].cpu().numpy()
    for i in range(1, len(poses)):
        R2 = poses[i, :3, :3].cpu().numpy()
        roll, pitch, yaw = rotation_matrix_to_euler(convention_inverse @ R2.T @ R1 @ convention_rotation,
                                                    z_down=True)  # rotation matrix are camera-to-world, cam1 --> cam2 is R2.T @ R1
        rolls[i] = -roll
        pitches[i] = pitch
        yaws[i] = yaw
    return fov_x, pitches, rolls, yaws


def _get_latents(condition_video_persp, conditional_video_equi, mask, accelerator, settings, latents_dir):
    if not latents_dir.exists() or settings.force_get_latents:
        # normed_vid = (data.video.float() * 2) - 1
        # conditional_video_pers = copy.deepcopy(conditional_video)
        # TODO this could be in the argus pipeline
        # conditional_video_equi, mask = pers2equi_batch(normed_vid[0].to(torch.float32), fov_x=fov_x,
        #                                                roll=rolls, pitch=pitches, yaw=yaws,
        #                                                width=settings.eq_height * 2, height=settings.eq_height,
        #                                                device=accelerator.device,
        #                                                return_mask=True)  # (T, C, H, W)
        conditional_video_equi = conditional_video_equi.to(settings.weight_dtype)

        latents_model = get_360_latents_model(settings.unet_path, accelerator, len(conditional_video_equi),
                                              accelerator.device, settings.weight_dtype)
        if len(condition_video_persp) > latents_model.argus_config.num_frames:
            print("WARNING: video longer than max frames for argus latents model, updating max frames")
            latents_model.argus_config.num_frames = len(condition_video_persp)
        argus_latents, _ = latents_model.forward_argus(condition_video_persp, conditional_video_equi, mask)
        argus_latents = argus_latents[0]
        latents_dir.parent.mkdir(exist_ok=True, parents=True)
        torch.save({"latents": argus_latents.cpu()}, latents_dir)
    else:
        latents_data = torch.load(latents_dir)
        argus_latents = latents_data["latents"].to(accelerator.device)

    return argus_latents


def _resize_eq_frames(eq_frames, new_shape=(1024, 2048)):
    return torch.nn.functional.interpolate(eq_frames.permute(0, 3, 1, 2).float(),
                                           new_shape, mode='bicubic',
                                           align_corners=True).permute(0, 2, 3, 1).clamp(0, 255).byte()


@torch.no_grad()
def main(device, settings):
    out_root = settings.out_root / "argus_cotracker_outputs" / settings.ds_name
    if settings.ds_name == "tapvid360-10k":
        out_root = out_root / f"gt_poses-{USE_GT_POSES}"
    out_root.mkdir(exist_ok=True, parents=True)
    args = _get_args()
    if args.split_filename is not None:
        with open(args.split_file, "r") as f:
            settings.specific_video_names = json.load(f)
    dataset, dl = get_dataset(settings.ds_root, settings.ds_name, settings.specific_video_names)

    accelerator = Accelerator(mixed_precision='no')
    vae, sam_pred_video, sam_pred_image, cotracker = get_models(settings.sam_checkpoint, accelerator,
                                                                accelerator.device, debug_skip_vae=False)
    sam_runner = SAMRunner(sam_img_pred=sam_pred_image, sam_video_pred=sam_pred_video)
    for data in tqdm(dl):
        vid_out_dir = out_root / data.seq_name[0]
        vid_out_dir.mkdir(exist_ok=True, parents=True)
        data.video = data.video.to(accelerator.device, non_blocking=True)
        normed_vid = (data.video.float() * 2) - 1
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
        else:
            # gt_roll, gt_pitch, gt_yaw = rot_to_euler_xyz(data.rotations[0], degrees=True)
            fov_x, pitches, rolls, yaws = _get_poses(data, settings.poses_root / settings.ds_name)
            conditional_video_equi, mask = pers2equi_batch(normed_vid[0].to(torch.float32), fov_x=fov_x,
                                                           roll=rolls, pitch=pitches, yaw=yaws,
                                                           width=settings.eq_height * 2, height=settings.eq_height,
                                                           device=accelerator.device,
                                                           return_mask=True)  # (T, C, H, W)
        latents_dir = settings.latents_root / settings.ds_name / data.seq_name[
            0] / f"gt_poses-{USE_GT_POSES}" / f"fov_x-{fov_x:.2f}" / "latents.pth"
        argus_latents = _get_latents(normed_vid[0].clone(), conditional_video_equi, mask, accelerator, settings,
                                     latents_dir)
        if JUST_GET_LATENTS:
            continue
        # TODO the larger i make this the better quality image that gets produced but obviously slower
        # settings.eq_height = 512 #2048
        mapper = Mappers(data.video.shape[-1], data.video.shape[-2], settings.eq_height, fov_x=fov_x)
        orig_pred_eq_frames_torch = _decode_pred_eq_frames(argus_latents, settings, vae)
        orig_pred_eq_frames_torch = _resize_eq_frames(orig_pred_eq_frames_torch,
                                                      new_shape=(settings.eq_height, settings.eq_height * 2))

        # PUT ORIG PERSP BACK IN THE IMAGE
        if USE_GT_POSES:
            R_w2c = data.rotations[0]
        else:
            R_w2c = rot(alpha=torch.from_numpy(rolls), beta=torch.from_numpy(pitches), gamma=torch.from_numpy(yaws),
                        degrees=True)
        pred_eq_frames_torch = overlay_orig_persp_on_pred_eq(data, mapper, orig_pred_eq_frames_torch, R_w2c)
        pred_eq_frames_out_dir = vid_out_dir / "pred_eq_frames"
        pred_eq_frames_out_dir.mkdir(exist_ok=True)
        for i, pred_eq_frame_torch in enumerate(pred_eq_frames_torch):
            Image.fromarray(pred_eq_frame_torch.cpu().numpy().astype(np.uint8)).save(
                pred_eq_frames_out_dir / f"{i}.jpg")

        if JUST_GET_PRED_EQ_FRAMES:
            continue

        first_mask = get_inital_frame_mask(data, dataset, sam_runner)
        first_mask_eq = project_perspective_mask_to_equirectangular_mask(first_mask, mapper, pitches, rolls, settings,
                                                                         yaws)
        vid_seg_logits, vid_seg_confs = sam_runner.run_through_video(pred_eq_frames_torch,
                                                                     masks=first_mask_eq[None][None])
        vid_seg_mean_above_conf = vid_seg_confs > 0.1

        # If no segmentation gets found we just have to estimate it to being all 1
        is_all_zero_slice = (vid_seg_mean_above_conf.sum(dim=(-2, -1)) == 0)
        vid_seg_mean_above_conf[is_all_zero_slice] = 1

        debug_seg_masks = []
        for i in range(vid_seg_mean_above_conf.shape[1]):
            debug_seg_mask = overlay_mask_over_image(pred_eq_frames_torch[i].cpu().numpy(),
                                                     vid_seg_mean_above_conf[0, i].cpu().numpy())
            debug_seg_masks.append(debug_seg_mask)

        obj_cnt_persp_imgs, obj_cnt_rot_matrices = get_object_centred_persp_imgs(pred_eq_frames_torch,
                                                                                 vid_seg_mean_above_conf,
                                                                                 mapper, batchsize=1)
        print("")

        # So now if its a bbox dataset we can run something like sam2 to get the bbox or if its point tracks now run cotracker
        # TODO should put this through the sam2 again on the object centred images to get better masks but skip for now
        # TODO can we use original perspective image for frame 0, or can we use this to correct drift in roll pitch yaw anyway?

        pred_bboxes = []
        debug_out_persp_imgs = []
        for i, pred_mask in enumerate(vid_seg_mean_above_conf[0]):
            pred_eq_pnts = torch.vstack(torch.where(pred_mask)).permute(1, 0)
            rot_mat = rot(torch.Tensor([rolls[i]]), torch.Tensor([pitches[i]]), torch.Tensor([yaws[i]]), degrees=True)
            pred_persp_points = mapper.point.cr.to_ij(pred_eq_pnts.flip(-1), rot_mat.to(pred_eq_pnts.device))
            pred_bbox = get_bbox_xyxy_from_points(pred_persp_points)
            pred_bboxes.append(pred_bbox)
            # debug_vis_img = (data.video[0, i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
            # for pnt in pred_persp_points:
            #     cv2.circle(debug_vis_img, (int(pnt[0].item()), int(pnt[1].item())), 3, (0, 0, 255), -1)
            # debug_out_persp_imgs.append(debug_vis_img)

        pred_bboxes = torch.stack(pred_bboxes).cpu()
        pred_bboxes_clipped = pred_bboxes.clone()
        pred_bboxes_clipped[:, 0::2] = pred_bboxes_clipped[:, 0::2].clamp(0, data.video.shape[-1] - 1)
        pred_bboxes_clipped[:, 1::2] = pred_bboxes_clipped[:, 1::2].clamp(0, data.video.shape[-2] - 1)

        gt_bboxes = data.bboxes_xyxy[0]

        iou_scores = bbox_iou(gt_bboxes, pred_bboxes_clipped).diag()
        vis_iou_scores = iou_scores[data.visibility[0].to(iou_scores.device).bool()]

        # TODO NOW MAKE USEFUL VIS
        gt_persp_frames = (data.video[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        gt_vis_bboxes = gt_bboxes.cpu().numpy().astype(int)
        pred_vis_bboxes = pred_bboxes_clipped.cpu().numpy().astype(int)
        for f_idx, frame in enumerate(gt_persp_frames):
            vis_frame = frame.copy()
            cv2.rectangle(vis_frame, gt_vis_bboxes[f_idx][:2], gt_vis_bboxes[f_idx][2:], (0, 255, 0), 2)
            cv2.rectangle(vis_frame, pred_vis_bboxes[f_idx][:2], pred_vis_bboxes[f_idx][2:], (255, 0, 0), 2)
            print("")


def project_perspective_mask_to_equirectangular_mask(first_mask, mapper, pitches, rolls, settings, yaws):
    first_rot_mat = rot(torch.Tensor([rolls[0]]), torch.Tensor([pitches[0]]), torch.Tensor([yaws[0]]), degrees=True)
    first_mask_eq_pnts = mapper.point.ij.to_cr(torch.from_numpy(np.column_stack(np.where(first_mask))).flip(-1),
                                               first_rot_mat)
    first_mask_eq = torch.zeros(settings.eq_height, settings.eq_height * 2, dtype=torch.bool)
    indices = first_mask_eq_pnts.long()
    indices_x, indices_y = indices[:, 0], indices[:, 1]
    valid_mask = (indices_x >= 0) & (indices_x < settings.eq_height * 2) & \
                 (indices_y >= 0) & (indices_y < settings.eq_height)
    indices_x_valid = indices_x[valid_mask]
    indices_y_valid = indices_y[valid_mask]
    # IMPORTANT: Tensor indexing is [row, col], so you use [indices_y, indices_x]
    first_mask_eq[indices_y_valid, indices_x_valid] = True
    return first_mask_eq


def get_inital_frame_mask(data, dataset, sam_runner):
    if isinstance(dataset, LaSOTDataset):
        pos_points, neg_points = None, None
        bbox_xyxy = data.bboxes_xyxy[:, 0].cpu().tolist()
    else:
        bbox_xyxy, neg_points = None, None
        raise NotImplementedError("FIGURE HOW TO GET THE POINTS FOR TAPVID360 DATASET")
    first_mask = sam_runner.run_through_image(
        (data.video[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
        bbox_xyxy=bbox_xyxy, positive_points=pos_points, negative_points=neg_points)
    return first_mask


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    settings = Settings()
    main(device, settings)
