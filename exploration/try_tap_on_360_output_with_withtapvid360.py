# TODO currently only 25 frames long which isnt long enough fot the dataset so take this into consideration

import gc
import inspect
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from accelerate import Accelerator
from accelerate.utils import is_compiled_module
from diffusers import AutoencoderKLTemporalDecoder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from argus_cotracker.components.pipelines.sam2.sam2_pipeline import get_sam2_predictor, run_through_img_sam, \
    run_through_sam_video
from conversions.mapper import Mappers
from data.data_utils import collate_fn
from data.dataloader import TAPVid360Dataset
from data.video_utils import decode_video_bytes_to_frames
from exploration.cotracker_utils import CoTracker
from metrics.metrics_utils import aggregate_metrics
from metrics.tapvid360_metrics import compute_metrics
from utils import DecordVideoReader, draw_point
from visualise import Visualiser, visualize_perspective_view


FORCE_RUN = False

def visualise_and_save_points(
        frames_to_visualise: torch.Tensor,
        points_to_visualise: torch.Tensor,
        rotations: torch.Tensor,
        mapper: Mappers,
        vis: Visualiser,
        output_dir: Path,
        filename_suffix: str
) -> None:
    """
    Generates a perspective view, visualises points on it, and saves the frames.

    Args:
        frames_to_visualise (torch.Tensor): The video frames to use as the background (GT or Pred).
        points_to_visualise (torch.Tensor): The equirectangular points to overlay (GT or Pred).
        rotations (torch.Tensor): The rotation matrices for generating the perspective view.
        mapper (Mappers): The mapper object for coordinate transformations.
        vis (Visualiser): The visualiser object.
        output_dir (Path): The root directory to save the output frames.
        filename_suffix (str): A descriptive suffix for the output sub-directory.
    """
    # 1. Generate the perspective view based on the input frames
    persp_view_vis = visualize_perspective_view(
        frames_to_visualise,
        rotations.to(frames_to_visualise.device),
        mapper,
        mode="both",
        ego_centric=True
    )

    # 2. Visualise the specified points on the generated perspective view
    # Ensure the visibility tensor matches the shape of the points tensor
    visibility = torch.ones_like(points_to_visualise.unsqueeze(0))[..., 0]

    visualisations = vis.visualise(
        persp_view_vis.unsqueeze(0) / 255.0,
        points_to_visualise.unsqueeze(0),
        vises=visibility,
        mapper=[mapper]
    )

    # 3. Create the specific output directory and save the frames
    final_output_dir = output_dir / f"final_{filename_suffix}"
    final_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving visualisation to: {final_output_dir}")
    for frame_idx, ego_eq_vis in enumerate(visualisations[0]):
        Image.fromarray(ego_eq_vis).save(final_output_dir / f"{frame_idx}.jpg")


def overlay_mask_over_image(image, mask, alpha=0.5, color=(0, 255, 0)):
    # Ensure mask and image have the same dimensions
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask dimensions do not match.")
    if not mask.any():
        return image
    if mask.max() > 1:
        mask = mask // 255
    overlay_mask = np.zeros_like(image)
    overlay_mask[mask == 1] = color

    anno_img = image.copy()
    anno_img[mask == 1] = cv2.addWeighted(image[mask == 1], 1 - alpha, overlay_mask[mask == 1], alpha, 0)

    return anno_img


def compute_center_of_mass(mask: torch.Tensor):
    """ Computes the center of mass for a batch of binary masks in a fully GPU-efficient manner. """
    assert mask.ndim == 3, "Mask must be a 3D tensor (B, H, W)"

    # Get dimensions
    B, h, w = mask.shape

    # Create coordinate grids for y and x (shape: (H, 1) for y, (1, W) for x)
    y_coords = torch.arange(h, device=mask.device, dtype=torch.float32).view(h, 1)  # (H, 1)
    x_coords = torch.arange(w, device=mask.device, dtype=torch.float32).view(1, w)  # (1, W)

    # Expand the grids to match the batch size (B, H, W)
    y_coords = y_coords.view(1, h, 1).expand(B, h, w)  # (B, H, W)
    x_coords = x_coords.view(1, 1, w).expand(B, h, w)  # (B, H, W)

    # Compute weighted sum of coordinates (batch-wise)
    total_mass = mask.sum(dim=(1, 2), keepdim=True)  # (B, 1, 1)

    # Avoid division by zero by setting empty masks' centers to (None) or (0,0)
    x_com = (mask * x_coords).sum(dim=(1, 2)) / total_mass.squeeze()  # (B,)
    y_com = (mask * y_coords).sum(dim=(1, 2)) / total_mass.squeeze()  # (B,)

    # Return as (B, 2) tensor with (x_com, y_com) for each batch
    return torch.stack((x_com, y_com), dim=1)  # (B, 2)


def get_object_centred_persp_imgs(eq_vid_frames, vid_seg_masks, mapper, batchsize=16):
    num_masks, num_frames = vid_seg_masks.shape[:2]

    rot_matrices = torch.empty(num_masks, num_frames, 3, 3, device=vid_seg_masks.device)
    for bs in range(0, len(vid_seg_masks), batchsize):
        batch_center_crs = compute_center_of_mass(vid_seg_masks[bs: bs + batchsize].flatten(0, 1))
        rot_matrices[bs: bs + batchsize] = mapper.compute_rotation_matrix_centred_at_point(
            batch_center_crs).view(-1, num_frames, 3, 3)

    centred_persp_imgs = []
    for i, rot_mat in enumerate(rot_matrices):
        centred_persp_imgs.append(mapper.image.equirectangular_image_to_perspective(eq_vid_frames, rot_mat))
        # debug_mosaic = create_mosaic(centred_persp_imgs[-1].cpu().numpy().copy())
        # Image.fromarray(debug_mosaic).save(f"debug_persp_mosaic-{i}.png")

    centred_persp_imgs = torch.stack(centred_persp_imgs, dim=0)

    return centred_persp_imgs, rot_matrices


def run_through_cotracker(co_tracker, batchsize, device, persp_imgs, persp_pos_points):
    # debug_vis_frame = persp_imgs[0].cpu().numpy().copy()
    # for pnt in starting_eq_coords:
    #     draw_point(debug_vis_frame, pnt, col=(0, 255, 0), radius=1)

    video_resize, _, query_points = co_tracker.preprocess(persp_imgs, query_points=persp_pos_points.flip(-1),
                                                          seg_mask=None, device=device)
    all_pred_tracks = []
    all_pred_vis = []
    for bs in range(0, len(video_resize), batchsize):
        # DEBUG
        # debug_vis = video_resize[0, 0].permute(1, 2, 0).cpu().int().numpy().copy()
        # for pnt in query_points[0][:, 1:]:
        #     draw_point(debug_vis, pnt, col=(0, 255, 0), radius=1)
        pred_tracks_batch, pred_vis_batch = co_tracker.run(video_resize[bs:bs + batchsize].contiguous(),
                                                           query_points=query_points[bs:bs + batchsize].contiguous())
        # frame_num = 5
        # debug_vis_output = video_resize[0, frame_num].permute(1, 2, 0).cpu().int().numpy().copy()
        # for pnt in pred_tracks_batch[0][frame_num]:
        #     draw_point(debug_vis_output, pnt, col=(255, 0, 0), radius=1)
        all_pred_tracks.append(pred_tracks_batch)
        all_pred_vis.append(pred_vis_batch)
    all_pred_tracks = torch.vstack(all_pred_tracks)
    all_pred_vis = torch.vstack(all_pred_vis)
    # Rescale the cotracker points back to expected image size
    # all_pred_tracks = co_tracker.rescale_points(all_pred_tracks.flip(-1), co_tracker.interp_shape,
    #                                             persp_imgs.shape[-3:-1])
    all_pred_tracks = co_tracker.rescale_points(all_pred_tracks, [co_tracker.interp_shape[-1],
                                                                  co_tracker.interp_shape[-2]],
                                                (persp_imgs.shape[-2], persp_imgs.shape[-3]))

    return all_pred_tracks, all_pred_vis


def decode_latents(vae, latents: torch.Tensor, num_frames: int, decode_chunk_size: int = 25,
                   extended_decoding: bool = False,
                   blend_decoding_ratio: int = 4) -> torch.Tensor:
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


def run_visualisation(ct_points, device, eq_points_pred, gt_eq_coords, gt_eq_frames, mapper, obj_cnt_persp_imgs,
                      out_root, persp_images_to_use, pred_eq_frames_torch, sample, vid_seg_mean_above_conf, vis):
    data_out_dir = out_root / "gt_points_gt_eq"
    data_out_dir.mkdir(exist_ok=True)
    for i, (frame, frame_pnts) in enumerate(zip(gt_eq_frames[:len(obj_cnt_persp_imgs[0])], gt_eq_coords)):
        vis_img = frame.copy()
        for pnt in frame_pnts:
            draw_point(vis_img, pnt, col=(0, 0, 255), radius=3)
        Image.fromarray(vis_img).save(str(data_out_dir / f"{i}.jpg"))
    data_out_dir = out_root / "gt_points_pred_masks"
    data_out_dir.mkdir(exist_ok=True)
    for i, (frame, frame_pnts) in enumerate(zip(gt_eq_frames[:len(obj_cnt_persp_imgs[0])], gt_eq_coords)):
        vis_img = frame.copy()
        mask_over_img = overlay_mask_over_image(vis_img, vid_seg_mean_above_conf[0, i].cpu().numpy())
        Image.fromarray(mask_over_img).save(str(data_out_dir / f"{i}.jpg"))
    obj_cnt_persp_dir = out_root / "obj_centred_persp_imgs"
    obj_cnt_persp_dir.mkdir(exist_ok=True)
    for i, obj_persp in enumerate(persp_images_to_use[0]):
        vis_img = obj_persp.cpu().numpy().copy()
        for pnt in ct_points[i]:
            draw_point(vis_img, pnt, col=(0, 0, 255), radius=3)
        Image.fromarray(vis_img).save(str(obj_cnt_persp_dir / f"{i}.jpg"))
    gt_eq_frames_torch = torch.from_numpy(gt_eq_frames)[:len(pred_eq_frames_torch)].to(device)
    eq_pred_persp_dir = out_root / "eq_preds_vs_gt"
    eq_pred_persp_dir.mkdir(exist_ok=True)
    # for i, eq_frame in enumerate(pred_eq_frames_torch):
    for i, eq_frame in enumerate(gt_eq_frames_torch):
        vis_img = eq_frame.cpu().numpy().copy()
        for pnt in eq_points_pred[i]:
            draw_point(vis_img, pnt, col=(0, 0, 255), radius=3)
        for pnt in gt_eq_coords[i]:
            draw_point(vis_img, pnt, col=(255, 0, 0), radius=3)
        Image.fromarray(vis_img).save(str(eq_pred_persp_dir / f"{i}.jpg"))
    visualise_and_save_points(gt_eq_frames_torch, gt_eq_coords, sample.rotations[0], mapper, vis, out_root,
                              "gt_points_gt_eq")
    visualise_and_save_points(pred_eq_frames_torch, gt_eq_coords, sample.rotations[0], mapper, vis, out_root,
                              "gt_points_pred_eq")
    visualise_and_save_points(gt_eq_frames_torch, eq_points_pred, sample.rotations[0], mapper, vis, out_root,
                              "pred_points_gt_eq")
    visualise_and_save_points(pred_eq_frames_torch, eq_points_pred, sample.rotations[0], mapper, vis, out_root,
                              "pred_points_pred_eq")


@torch.no_grad()
def main(device, ds_root, video_names):
    dataset = TAPVid360Dataset(ds_root / f"TAPVid360-10k", transforms.Compose([transforms.ToTensor()]),
                               num_queries=256, num_frames=25, specific_videos_list=video_names)
    dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True)

    accelerator = Accelerator(mixed_precision='no')
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        'stabilityai/stable-video-diffusion-img2vid', subfolder="vae", revision=None)
    vae = accelerator.unwrap_model(vae)
    vae = vae.to(device).eval()

    sam_checkpoint = Path("/home/finlay/Shared/pretrained_models/segmentation") / "sam2" / "sam2_hiera_large.pt"
    sam_pred_video = get_sam2_predictor(single_image=False, checkpoint=sam_checkpoint, device=device)
    sam_pred_image = get_sam2_predictor(single_image=True, checkpoint=sam_checkpoint, device=device)
    cotracker = CoTracker()

    # eq_search_dirs = [ds_root / "vd360_32_gen_outputs=12000-13000", ds_root / "vd360_32_gen_outputs=13000-14000",
    #                   ds_root / "vd360_32_gen_outputs=14000-15000", ds_root / "vd360_32_gen_outputs=15000-16000",
    #                   ds_root / "vd360_32_gen_outputs=16000-17000", ds_root / "vd360_32_gen_outputs=17000-18000"]
    eq_search_dirs = [ds_root / "vd360_32_gen_outputs_FROM_VIKING", ds_root / "vd360_32_gen_outputs=1000-2000",
                      ds_root / "vd360_32_gen_outputs=2000-3000", ds_root / "vd360_32_gen_outputs=3000-4000",
                      ds_root / "vd360_32_gen_outputs=4000-5000", ds_root / "vd360_32_gen_outputs=5000-6000",
                      ds_root / "vd360_32_gen_outputs=6000-7000", ds_root / "vd360_32_gen_outputs=7000-8000",
                      ds_root / "vd360_32_gen_outputs=8000-9000", ds_root / "vd360_32_gen_outputs=9000-10000",
                      ds_root / "vd360_32_gen_outputs=10000-11000"]

    argus_feats_dir = Path("/media/finlay/BigDaddyDrive/Outputs/tracker/whatGoesAround/argus_feats")
    mapper = Mappers(320, 240, 1024, fov_x=75.18)
    vis = Visualiser(point_radius=3)
    out_root = Path(
        "/media/finlay/BigDaddyDrive/Outputs/tracker/whatGoesAround/exploration/outputs/try_on_360_output_withtapvid360")
    out_root.mkdir(parents=True, exist_ok=True)
    out_root_metrics = out_root / "metrics"
    out_root_metrics.mkdir(exist_ok=True)
    out_root_vis = out_root / "vis"
    out_root_vis.mkdir(exist_ok=True)
    for i, sample in enumerate(tqdm(dl)):
        # # TODO DEBUG
        # if i < 15:
        #     continue

        metrics_path = out_root_metrics / f"{sample.seq_name[0].replace('/', '-')}_metrics.json"
        if metrics_path.exists() and not FORCE_RUN:
            print(f"Metrics already exist for {sample.seq_name[0]}, skipping...")
            continue

        print(f"Processing video: {sample.seq_name[0]} at idx: {i}")

        persp_frames = (sample.video[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        gt_uv_coords = sample.trajectory[0]
        gt_rot_mats = sample.rotations[0]

        # gt_eq_coords = mapper.point.vc.to_cr(gt_uv_coords, gt_rot_mats)
        # TODO SOMETHING IS VERY WRONG WITH THESE EQ COORDS
        gt_eq_coords = mapper.point.vc.to_cr(gt_uv_coords, gt_rot_mats)
        gt_persp_coords = mapper.point.vc.to_ij(gt_uv_coords, gt_rot_mats)
        # TODO need gt eq frames - THIS NEEDS DOING NEXT
        # TODO for this I would need to search in which vd360_32_gen_outputs=... file the video is and then get the eq frames from there
        gt_eq_frames = None
        for eq_search_dir in eq_search_dirs:
            if (eq_search_dir / sample.seq_name[0].split("/")[0]).exists():
                eq_frames_path = eq_search_dir / sample.seq_name[0].split("/")[0] / "frames_32.pth"
                assert eq_frames_path.exists()
                gt_eq_frames = decode_video_bytes_to_frames(torch.load(eq_frames_path, map_location='cpu'))
                break
        assert gt_eq_frames is not None

        starting_eq_coords = gt_eq_coords[0]
        argus_feats_path = argus_feats_dir / (sample.seq_name[0].replace("/", "-") + ".pth")
        assert argus_feats_path.exists()
        argus_feats = torch.load(argus_feats_path)
        argus_latents = argus_feats["latents"].to(device)
        with torch.no_grad():
            pred_eq_frames = decode_latents(vae.eval(), argus_latents, argus_latents.shape[1], 5, False, 4)
        pred_eq_frames_np = ((pred_eq_frames[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).cpu().to(
            torch.float32).numpy().astype(np.uint8)

        pred_eq_frames_torch = torch.from_numpy(pred_eq_frames_np).to(device)
        pred_eq_frames_torch = torch.nn.functional.interpolate(pred_eq_frames_torch.permute(0, 3, 1, 2).float(),
                                                               gt_eq_frames.shape[1:3], mode='bilinear',
                                                               align_corners=False).permute(0, 2, 3, 1).clamp(0,
                                                                                                              255).byte()

        # First get sam image results for first frame
        first_mask = run_through_img_sam(sam_pred_image, starting_eq_coords[::20], [],
                                         pred_eq_frames_torch.cpu().numpy()[0])
        vid_seg_logits, vid_seg_confs = run_through_sam_video(sam_pred_video, pred_eq_frames_torch,
                                                              masks=torch.from_numpy(first_mask)[None][None])
        vid_seg_mean_above_conf = vid_seg_confs > 0.1

        # If no segmentation gets found we just have to estimate it to being all 1
        is_all_zero_slice = (vid_seg_mean_above_conf.sum(dim=(-2, -1)) == 0)
        vid_seg_mean_above_conf[is_all_zero_slice] = 1

        obj_cnt_persp_imgs, obj_cnt_rot_matrices = get_object_centred_persp_imgs(pred_eq_frames_torch,
                                                                                 vid_seg_mean_above_conf,
                                                                                 mapper, batchsize=1)

        persp_images_to_use = obj_cnt_persp_imgs.clone()
        # Put through original perspective image for first frame
        persp_images_to_use[0, 0] = torch.from_numpy(persp_frames[0])

        # TODO now need to get the original points in the right place in the object-centred-perspective images (use the first image from the dataset)
        ct_points, ct_vis = run_through_cotracker(cotracker, 1, torch.device("cuda:0"),
                                                  persp_images_to_use, gt_persp_coords[0])
        eq_points_pred = mapper.point.ij.to_cr(ct_points, obj_cnt_rot_matrices)

        pred_uv_coords = mapper.point.cr.to_vc(eq_points_pred[0].to(gt_rot_mats.device), gt_rot_mats)

        out_root_vis_vid = out_root_vis / sample.seq_name[0].replace("/", "-")
        out_root_vis_vid.mkdir(exist_ok=True)

        metrics = compute_metrics(pred_uv_coords, gt_uv_coords, sample.visibility[0])
        with open(metrics_path, 'w') as f:
            json.dump({m: metrics[m].mean().item() for m in metrics}, f, indent=4)

        run_visualisation(ct_points[0], device, eq_points_pred[0], gt_eq_coords, gt_eq_frames, mapper,
                          obj_cnt_persp_imgs, out_root_vis, persp_images_to_use, pred_eq_frames_torch, sample,
                          vid_seg_mean_above_conf, vis)

        del pred_eq_frames, pred_eq_frames_torch, vid_seg_logits, vid_seg_confs
        del vid_seg_mean_above_conf, obj_cnt_persp_imgs, obj_cnt_rot_matrices
        del persp_images_to_use, ct_points, ct_vis

        gc.collect()
        torch.cuda.empty_cache()

    aggregate_metrics(out_root_metrics)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds_root = Path("/media/finlay/BigDaddyDrive/Outputs/tracker/tapvid360")
    assert ds_root.exists()

    with open(Path(__file__).parent.parent / "data/mini_dataset_names_100_items.txt", "r") as f:
        video_names = [line.strip() for line in f.readlines()]

    main(device, ds_root, video_names)
