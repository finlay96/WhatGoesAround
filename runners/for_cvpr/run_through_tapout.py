import json
from tqdm import tqdm

from accelerate import Accelerator
import cv2
import numpy as np
from PIL import Image
import torch

from conversions.mapper import Mappers
from data.dataloader_lasot import LaSOTDataset
from metrics.tapvid360_metrics import compute_metrics
from runners.for_cvpr.settings import get_args, Settings, set_host_in_settings
from runners.model_utils import SAMRunner, get_models
from runners.run_argus_cotracker import _decode_pred_eq_frames, _resize_eq_frames, run_through_cotracker
from runners.utils import overlay_orig_persp_on_pred_eq, get_dataset, get_object_centred_persp_imgs
from runners.vis_utils import overlay_mask_over_image


def get_full_vid_eq_seg_masks(first_mask_eq, pred_eq_frames_torch, sam_runner):
    vid_seg_logits, vid_seg_confs = sam_runner.run_through_video(pred_eq_frames_torch,
                                                                 masks=first_mask_eq[None][None])
    vid_seg_mean_above_conf = vid_seg_confs > 0.1
    # If no segmentation gets found we just have to estimate it to being all 1
    is_all_zero_slice = (vid_seg_mean_above_conf.sum(dim=(-2, -1)) == 0)
    vid_seg_mean_above_conf[is_all_zero_slice] = 1
    return vid_seg_mean_above_conf


def get_inital_frame_mask(data, dataset, sam_runner, mapper, every_x_points=10, with_debug=False):
    if isinstance(dataset, LaSOTDataset):
        pos_points, neg_points = None, None
        bbox_xyxy = data.bboxes_xyxy[:, 0].cpu().tolist()
    else:
        bbox_xyxy, neg_points = None, None
        pos_points = mapper.point.vc.to_ij(data.trajectory)[0, 0].cpu().numpy().astype(int).tolist()
        pos_points = pos_points[::every_x_points]
        if with_debug:
            debug_vis = (data.video[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
            for pnt in pos_points:
                cv2.circle(debug_vis, (pnt[0], pnt[1]), 3, (0, 0, 255), -1)
    first_mask = sam_runner.run_through_image(
        (data.video[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
        bbox_xyxy=bbox_xyxy, positive_points=pos_points, negative_points=neg_points)
    return first_mask, pos_points


def main(args, settings):
    if args.split_file is not None:
        with open(args.split_file, "r") as f:
            settings.specific_video_names = json.load(f)
    dataset, dl = get_dataset(settings.paths.tapvid360_data_root, settings.ds_name, settings.specific_video_names)
    accelerator = Accelerator(mixed_precision='no')
    device = accelerator.device
    for data in tqdm(dl):
        assert len(data.seq_name) == 1, "Batch size greater than 1 not supported"
        latents_dir = settings.paths.out_root / "argus_feats" / settings.ds_name / data.seq_name[
            0] / f"gt_poses-{args.use_gt_rot}"
        assert latents_dir.exists(), f"Latents dir {latents_dir} does not exist"
        results_dir = settings.paths.out_root / "results" / f"gt_poses-{args.use_gt_rot}"
        metrics_dir = settings.paths.out_root / "metrics" / f"gt_poses-{args.use_gt_rot}"
        save_seq_name = data.seq_name[0].replace("/", "-")  # avoid creating subfolders
        metrics_file_name = metrics_dir / f"{save_seq_name}_metrics.json"
        if metrics_file_name.exists() and settings.skip_if_exists:
            print(f"Skipping {data.seq_name[0]} as metrics already exist")
            continue
        rotations_data = torch.load(latents_dir / "rotations.pth")
        fov_x = rotations_data["fov_x"]
        rots = rotations_data["rotations"].to(accelerator.device, non_blocking=True)
        data.video = data.video.to(accelerator.device, non_blocking=True)
        mapper = Mappers(data.video.shape[-1], data.video.shape[-2], data.equirectangular_height[0], fov_x=fov_x)
        vae, sam_pred_video, sam_pred_image, cotracker = get_models(settings.paths.sam_checkpoint, accelerator,
                                                                    accelerator.device, debug_skip_vae=False)
        vae = vae.eval()
        sam_runner = SAMRunner(sam_img_pred=sam_pred_image, sam_video_pred=sam_pred_video)

        latents_data = torch.load(latents_dir / "latents.pth")
        argus_latents = latents_data["latents"].to(accelerator.device)

        with torch.no_grad():
            orig_pred_eq_frames_torch = _decode_pred_eq_frames(argus_latents, settings, vae)
        orig_pred_eq_frames_torch = _resize_eq_frames(orig_pred_eq_frames_torch, new_shape=(1024, 2048))

        pred_eq_frames_torch = overlay_orig_persp_on_pred_eq(data, mapper, orig_pred_eq_frames_torch, rots)

        debug_vid_out_dir = settings.paths.out_root / "debugs"
        if args.debugs:
            debug_vid_out_dir.mkdir(exist_ok=True, parents=True)
            pred_eq_frames_out_dir = debug_vid_out_dir / "pred_eq_frames"
            pred_eq_frames_out_dir.mkdir(exist_ok=True)
            for i, pred_eq_frame_torch in enumerate(pred_eq_frames_torch):
                Image.fromarray(pred_eq_frame_torch.cpu().numpy().astype(np.uint8)).save(
                    pred_eq_frames_out_dir / f"{i}.jpg")

        first_mask, pos_points = get_inital_frame_mask(data, dataset, sam_runner, mapper, every_x_points=5)

        if args.debugs:
            first_mask_pred_out_dir = debug_vid_out_dir / "first_mask_pred"
            first_mask_pred_out_dir.mkdir(exist_ok=True)
            debug_vis = (data.video[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
            for pnt in pos_points:
                cv2.circle(debug_vis, (pnt[0], pnt[1]), 3, (0, 0, 255), -1)
            debug_vis = overlay_mask_over_image(debug_vis, first_mask)
            Image.fromarray(debug_vis).save(first_mask_pred_out_dir / "debug_vis.jpg")

        first_mask_eq_torch_, _ = mapper.image.perspective_image_to_equirectangular(
            torch.from_numpy(first_mask[None, ..., None]).to(rots.device), rots[0], interp_mode="nearest")
        first_mask_eq = first_mask_eq_torch_[0, ..., 0].to(bool)
        vid_seg_mean_above_conf = get_full_vid_eq_seg_masks(first_mask_eq, pred_eq_frames_torch, sam_runner)

        if args.debugs:
            all_mask_preds_out_dir = debug_vid_out_dir / "all_mask_preds"
            all_mask_preds_out_dir.mkdir(exist_ok=True)
            for i, f in enumerate(pred_eq_frames_torch):
                debug_vis = (f.cpu().numpy()).astype(np.uint8).copy()
                debug_vis = overlay_mask_over_image(debug_vis, vid_seg_mean_above_conf[0, i].cpu())
                Image.fromarray(debug_vis).save(all_mask_preds_out_dir / f"{i}.jpg")

        obj_cnt_persp_imgs, obj_cnt_rot_matrices = get_object_centred_persp_imgs(pred_eq_frames_torch,
                                                                                 vid_seg_mean_above_conf,
                                                                                 mapper, batchsize=1)
        if args.debugs:
            obj_centre_persp_imgs_out_dir = debug_vid_out_dir / "obj_centre_persp_imgs"
            obj_centre_persp_imgs_out_dir.mkdir(exist_ok=True)
            for i, f in enumerate(obj_cnt_persp_imgs[0]):
                debug_vis = (f.cpu().numpy()).astype(np.uint8).copy()
                Image.fromarray(debug_vis).save(obj_centre_persp_imgs_out_dir / f"{i}.jpg")

        query_frame_points = mapper.point.vc.to_ij(data.trajectory)[0, 0]
        query_frame_points = query_frame_points.flip(-1)
        cotracker.model = cotracker.model.to(device)
        pred_tracks, pred_vis = run_through_cotracker(cotracker, 1, device,
                                                      obj_cnt_persp_imgs, query_frame_points)

        if args.debugs:
            obj_centre_persp_imgs_with_tracks_out_dir = debug_vid_out_dir / "obj_centre_persp_imgs_with_tracks"
            obj_centre_persp_imgs_with_tracks_out_dir.mkdir(exist_ok=True)
            for i, f in enumerate(obj_cnt_persp_imgs[0]):
                debug_vis = (f.cpu().numpy()).astype(np.uint8).copy()
                for pnt in pred_tracks[0, i]:
                    cv2.circle(debug_vis, (int(pnt[0]), int(pnt[1])), 3, (0, 0, 255), -1)
                Image.fromarray(debug_vis).save(obj_centre_persp_imgs_with_tracks_out_dir / f"{i}.jpg")

        eq_points_pred = mapper.point.ij.to_cr(pred_tracks, obj_cnt_rot_matrices)
        pred_unit_vectors = mapper.point.cr.to_vc(eq_points_pred[0].to(rots.device), rots)
        metrics = compute_metrics(pred_unit_vectors, data.trajectory[0], data.visibility[0])
        results_dir.mkdir(exist_ok=True)
        metrics_dir.mkdir(exist_ok=True)
        torch.save(pred_unit_vectors.cpu(), results_dir / f"{save_seq_name}.pth")
        with open(metrics_file_name, 'w') as f:
            json.dump({m: metrics[m].mean().item() for m in metrics}, f, indent=4)
        for m in metrics:
            print(f"{m}: {metrics[m].mean().item()}")


if __name__ == "__main__":
    args = get_args()
    settings = set_host_in_settings(Settings())
    main(args, settings)
