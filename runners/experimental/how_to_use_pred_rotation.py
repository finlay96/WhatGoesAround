import dataclasses
import json
from pathlib import Path

from accelerate import Accelerator
from diffusers import AutoencoderKLTemporalDecoder
from torchvision import transforms

from conversions.mapper import Mappers
from conversions.mapper_atomics import compute_vertical_fov, get_pixel_focal_length
from data.dataloader import TAPVid360Dataset
from metrics.tapvid360_metrics import compute_metrics
from runners.model_utils import get_models, SAMRunner
from runners.pose_utils import VGGTPoseRunner
from runners.run_argus_cotracker import _get_latents, _decode_pred_eq_frames, _resize_eq_frames, run_through_cotracker

import cv2
import numpy as np
from PIL import Image
import torch

from runners.utils import get_object_centred_persp_imgs
from runners.vis_utils import overlay_mask_over_image


def get_pred_rots(gt_rots, pred_rots):
    """
        Calculates corrected prediction rotations in PyTorch.

        Args:
            gt_rots (torch.Tensor): Ground truth rotation tensor (e.g., shape [N, 3, 3]).
            pred_rots (torch.Tensor): Predicted rotation tensor (e.g., shape [N, 3, 3]).
                                      Must be on the same device or gt_rots will be moved.

        Returns:
            torch.Tensor: The aligned predicted rotations.
        """
    # Ensure gt_rots[0] is on the same device as pred_rots
    gt_rots_0 = gt_rots[0].to(device=pred_rots.device, dtype=pred_rots.dtype)

    # Calculate the inverse of the first predicted rotation
    # Replaces np.linalg.inv(pred_rots[0].cpu().numpy())
    inv_pred_rots_0 = torch.linalg.inv(pred_rots[0])

    # Calculate the correction matrix
    R_correction = gt_rots_0 @ inv_pred_rots_0

    # Apply the correction to all predicted rotations
    # This uses broadcasting: (3, 3) @ (N, 3, 3) -> (N, 3, 3)
    pred_rots_aligned = R_correction @ pred_rots

    return pred_rots_aligned


# --- Define constant matrices outside the function for efficiency ---
# R_libworld_to_vggtworld = torch.tensor([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
# inv_R_lib = torch.linalg.inv(R_libworld_to_vggtworld)
# (Note: This matrix is its own inverse)
INV_R_LIB = torch.tensor(
    [[0., -1., 0.],
     [-1., 0., 0.],
     [0., 0., -1.]]
)

# R_align_cam = torch.tensor([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
# inv_R_align = torch.linalg.inv(R_align_cam)
# (Note: The inverse is the transpose)
INV_R_ALIGN = torch.tensor(
    [[0., -1., 0.],
     [0., 0., 1.],
     [-1., 0., 0.]]
)

def through_conversion(c2w_vggt):
    """
    Applies a coordinate space transformation in PyTorch.

    Args:
        c2w_vggt (torch.Tensor): Camera-to-world matrices, shape (3, 3) or (N, 3, 3).

    Returns:
        torch.Tensor: The transformed matrices.
    """
    # Move constant tensors to the same device and dtype as the input
    inv_R_lib_d = INV_R_LIB.to(device=c2w_vggt.device, dtype=c2w_vggt.dtype)
    inv_R_align_d = INV_R_ALIGN.to(device=c2w_vggt.device, dtype=c2w_vggt.dtype)

    # Apply the transformation using torch matrix multiplication
    # Broadcasting handles (N, 3, 3) or (3, 3) inputs
    c2w_tapvid360 = inv_R_lib_d @ c2w_vggt @ inv_R_align_d

    return c2w_tapvid360


@dataclasses.dataclass
class Settings:
    ds_name = "tapvid360-10k"
    specific_video_names = ["-73Nyd_QcNc_clip_0003/0"]  # "nqR-umO4bvY_clip_0006/0"
    unet_path = "/home/userfs/f/fgch500/storage/pretrained_models/video_generation/argus"
    # CSGPU
    ds_root = Path("/home/userfs/f/fgch500/storage/shared/TAPVid360/data")
    sam_checkpoint = Path(
        "/shared/storage/cs/staffstore/fgch500/pretrained_models/segmentation") / "sam2" / "sam2_hiera_large.pt"
    out_root = Path("/home/userfs/f/fgch500/storage/shared/WhatGoesAround")

    force_get_latents = True
    eq_height = 512  # TODO watchout this is currently hardcoded
    weight_dtype = torch.float32
    decode_chunk_size = 10  # Should be able to be bigger for larger gpu's
    offload_to_cpu = True


def overlay_orig_persp_on_pred_eq(data, mapper, pred_eq_frames_torch, R_w2c):
    warped_gt_persp_batch, valid_mask_batch = mapper.image.perspective_image_to_equirectangular(
        data.video.permute(0, 2, 3, 1) * 255,
        R_w2c.to(data.video.device),
        to_uint=False
    )
    valid_mask_broadcastable = valid_mask_batch.unsqueeze(-1).bool()

    return torch.where(valid_mask_broadcastable.to(pred_eq_frames_torch.device),
                       warped_gt_persp_batch.to(pred_eq_frames_torch.device), pred_eq_frames_torch)


if __name__ == "__main__":
    settings = Settings()
    raw_vggt_extrinsics = "/home/userfs/f/fgch500/storage/shared/WhatGoesAround/vggt_extrinsics/raw_extrinsic_vggt.pt"
    vggt_extrinsics = torch.load(raw_vggt_extrinsics)

    video_names = ["-73Nyd_QcNc_clip_0003/0"]
    dataset = TAPVid360Dataset(settings.ds_root / f"TAPVid360-10k",
                               transforms.Compose([transforms.ToTensor()]),
                               num_queries=256, num_frames=32, specific_videos_list=video_names)
    data = dataset[0]
    eq_height = 1024

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred_rots = through_conversion(vggt_extrinsics[..., :3])[0].to(torch.float32)
    pred_rots = get_pred_rots(data.rotations, pred_rots)
    TESTS = [
        # {"name": "gt", "fov_x": data.fov_x, "rots": data.rotations, "height": 240, "width": 320},
        {"name": "pred_rots", "fov_x": data.fov_x, "rots": pred_rots, "height": 240, "width": 320},
        # {"name": "pred_rots_pred_fov", "fov_x": 71.74, "rots": pred_rots, "height": 240, "width": 320},
        # {"name": "pred_rots_pred_fov_new_size", "fov_x": 71.74, "rots": pred_rots, "height": 392, "width": 518}
    ]
    accelerator = Accelerator(mixed_precision='no')
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        'stabilityai/stable-video-diffusion-img2vid', subfolder="vae", revision=None)
    vae = accelerator.unwrap_model(vae)
    vae = vae.to(device).eval()


    # 518, 392, 71.74
    for test in TESTS:
        print("RUNNING TEST:", test["name"])
        if test["name"] != "gt":
            pose_runner = VGGTPoseRunner(device=device)
            extrinsics, fov_x = pose_runner.run(data.video.permute(0, 2, 3, 1), mode="pad")
            pred_rots = through_conversion(vggt_extrinsics[..., :3])[0].to(torch.float32)
            pred_rots = get_pred_rots(data.rotations, pred_rots).to(device)
            f_x, f_y = get_pixel_focal_length(518, 392, fov_x, fov_y=compute_vertical_fov(fov_x, 518 / 392))
            gt_f_x, gt_f_y = get_pixel_focal_length(320, 240, fov_x, fov_y=compute_vertical_fov(data.fov_x, 320 / 240))

            del pose_runner

        data.video = data.video.to(device, non_blocking=True)

        mapper = Mappers(test["width"], test["height"], eq_height, fov_x=test["fov_x"])
        mapper_gt = Mappers(data.video.shape[-1], data.video.shape[-2], eq_height, fov_x=data.fov_x)
        conditional_video_equi, mask = mapper.image.perspective_image_to_equirectangular(
            data.video.permute(0, 2, 3, 1), test["rots"], to_uint=False)
        conditional_video_equi = conditional_video_equi.clip(0, 1)
        conditional_video_equi = conditional_video_equi * 2 - 1
        conditional_video_equi = conditional_video_equi.permute(0, 3, 1, 2)
        mask = mask.unsqueeze(1)
        # Argus needs of shape 512, 1024
        conditional_video_equi = torch.nn.functional.interpolate(conditional_video_equi, size=(512, 1024),
                                                                 mode='bilinear', align_corners=False)
        mask = torch.nn.functional.interpolate(mask.float(), size=(512, 1024), mode='nearest').bool()

        out_dir = settings.out_root / "pred_rotation_experiment" / settings.ds_name / data.seq_name / test["name"]
        out_dir.mkdir(parents=True, exist_ok=True)

        vid_out_dir = out_dir / "debugs"
        vid_out_dir.mkdir(exist_ok=True, parents=True)
        inputs_out_dir = vid_out_dir / "inputs"
        inputs_out_dir.mkdir(exist_ok=True)
        conditional_video_equi_vis = ((conditional_video_equi.permute(0, 2, 3, 1).clamp(-1, 1) + 1) * 127.5).to(
            torch.uint8)
        for i, frame in enumerate(conditional_video_equi_vis):
            Image.fromarray(frame.cpu().numpy().astype(np.uint8)).save(inputs_out_dir / f"{i}.jpg")

        normed_vid = (data.video.float() * 2) - 1
        latents_path = out_dir / "latents.pth"
        # TODO does normed_vid need cloning???
        argus_latents = _get_latents(normed_vid, conditional_video_equi, mask, accelerator, settings, latents_path)
        with torch.no_grad():
            orig_pred_eq_frames_torch = _decode_pred_eq_frames(argus_latents, settings, vae)
        orig_pred_eq_frames_torch = _resize_eq_frames(orig_pred_eq_frames_torch, new_shape=(1024, 2048))

        pred_eq_frames_torch = overlay_orig_persp_on_pred_eq(data, mapper, orig_pred_eq_frames_torch, test["rots"])
        pred_eq_frames_out_dir = vid_out_dir / "pred_eq_frames"
        pred_eq_frames_out_dir.mkdir(exist_ok=True)
        for i, pred_eq_frame_torch in enumerate(pred_eq_frames_torch):
            Image.fromarray(pred_eq_frame_torch.cpu().numpy().astype(np.uint8)).save(
                pred_eq_frames_out_dir / f"{i}.jpg")
        #####################################################################################################

        _, sam_pred_video, sam_pred_image, cotracker = get_models(settings.sam_checkpoint, accelerator,
                                                                  accelerator.device, debug_skip_vae=True)
        sam_runner = SAMRunner(sam_img_pred=sam_pred_image, sam_video_pred=sam_pred_video)  #
        video = data.video.to(device)[None]
        trajectory = data.trajectory.to(device)[None]
        visibility = data.visibility.to(device)[None]
        # data.rotations = data.rotations.to(device)[None]
        # first_mask = get_inital_frame_mask(data, dataset, sam_runner, mapper)
        bbox_xyxy, neg_points = None, None
        pos_points = mapper_gt.point.vc.to_ij(trajectory)[0, 0].cpu().numpy().astype(int).tolist()
        pos_points = pos_points[::10]
        first_mask = sam_runner.run_through_image(
            (video[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
            bbox_xyxy=bbox_xyxy, positive_points=pos_points, negative_points=neg_points)

        # DEBUGS
        first_mask_pred_out_dir = vid_out_dir / "first_mask_pred"
        first_mask_pred_out_dir.mkdir(exist_ok=True)
        debug_vis = (video[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
        for pnt in pos_points:
            cv2.circle(debug_vis, (pnt[0], pnt[1]), 3, (0, 0, 255), -1)
        debug_vis = overlay_mask_over_image(debug_vis, first_mask)
        Image.fromarray(debug_vis).save(first_mask_pred_out_dir / "debug_vis.jpg")

        first_mask_eq_torch_, _ = mapper.image.perspective_image_to_equirectangular(
            torch.from_numpy(first_mask[None, ..., None]).to(test["rots"].device), test["rots"][0], interp_mode="nearest")
        first_mask_eq = first_mask_eq_torch_[0, ..., 0].to(bool)
        vid_seg_logits, vid_seg_confs = sam_runner.run_through_video(pred_eq_frames_torch,
                                                                     masks=first_mask_eq[None][None])
        vid_seg_mean_above_conf = vid_seg_confs > 0.1

        # If no segmentation gets found we just have to estimate it to being all 1
        is_all_zero_slice = (vid_seg_mean_above_conf.sum(dim=(-2, -1)) == 0)
        vid_seg_mean_above_conf[is_all_zero_slice] = 1

        # DEBUGS
        all_mask_preds_out_dir = vid_out_dir / "all_mask_preds"
        all_mask_preds_out_dir.mkdir(exist_ok=True)
        for i, f in enumerate(pred_eq_frames_torch):
            debug_vis = (f.cpu().numpy()).astype(np.uint8).copy()
            debug_vis = overlay_mask_over_image(debug_vis, vid_seg_mean_above_conf[0, i].cpu())
            Image.fromarray(debug_vis).save(all_mask_preds_out_dir / f"{i}.jpg")

        obj_cnt_persp_imgs, obj_cnt_rot_matrices = get_object_centred_persp_imgs(pred_eq_frames_torch,
                                                                                 vid_seg_mean_above_conf,
                                                                                 mapper, batchsize=1)
        # DEBUGS
        obj_centre_persp_imgs_out_dir = vid_out_dir / "obj_centre_persp_imgs"
        obj_centre_persp_imgs_out_dir.mkdir(exist_ok=True)
        for i, f in enumerate(obj_cnt_persp_imgs[0]):
            debug_vis = (f.cpu().numpy()).astype(np.uint8).copy()
            Image.fromarray(debug_vis).save(obj_centre_persp_imgs_out_dir / f"{i}.jpg")

        query_frame_points = mapper.point.vc.to_ij(trajectory)[0, 0]
        query_frame_points = query_frame_points.flip(-1)
        cotracker.model = cotracker.model.to(device)
        pred_tracks, pred_vis = run_through_cotracker(cotracker, 1, device,
                                                      obj_cnt_persp_imgs, query_frame_points)

        # DEBUGS
        obj_centre_persp_imgs_with_tracks_out_dir = vid_out_dir / "obj_centre_persp_imgs_with_tracks"
        obj_centre_persp_imgs_with_tracks_out_dir.mkdir(exist_ok=True)
        for i, f in enumerate(obj_cnt_persp_imgs[0]):
            debug_vis = (f.cpu().numpy()).astype(np.uint8).copy()
            for pnt in pred_tracks[0, i]:
                cv2.circle(debug_vis, (int(pnt[0]), int(pnt[1])), 3, (0, 0, 255), -1)
            Image.fromarray(debug_vis).save(obj_centre_persp_imgs_with_tracks_out_dir / f"{i}.jpg")

        eq_points_pred = mapper.point.ij.to_cr(pred_tracks, obj_cnt_rot_matrices)
        pred_unit_vectors = mapper.point.cr.to_vc(eq_points_pred[0].to(test["rots"].device), test["rots"])
        gt_unit_vectors = trajectory
        metrics = compute_metrics(pred_unit_vectors, gt_unit_vectors[0], visibility[0])
        results_dir = out_dir / "results"
        results_dir.mkdir(exist_ok=True)
        metrics_dir = out_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        # for b, seq_name in enumerate(data.seq_name):
        save_seq_name = data.seq_name.replace("/", "-")  # avoid creating subfolders
        torch.save(pred_unit_vectors.cpu(), results_dir / f"{save_seq_name}.pth")
        with open(metrics_dir / f"{save_seq_name}_metrics.json", 'w') as f:
            json.dump({m: metrics[m].mean().item() for m in metrics}, f, indent=4)
        for m in metrics:
            print(f"{m}: {metrics[m].mean().item()}")
