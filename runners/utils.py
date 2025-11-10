from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from conversions.rotations import rot
from data.data_utils import collate_fn, ResizeTensor
from data.dataloader import TAPVid360Dataset
from data.dataloader_lasot import collate_fn_laSOT, LaSOTDataset


def get_dataset(ds_root, ds_name, specific_video_names=None):
    assert ds_root.exists()
    if ds_name == "tapvid360-10k":
        if specific_video_names is None:
            with open(Path(__file__).parent.parent / "data/tapvid10k_wga_dataset.txt", "r") as f:
                video_names = [line.strip() for line in f.readlines()]
        else:
            video_names = specific_video_names
        dataset = TAPVid360Dataset(ds_root / f"TAPVid360-10k", transforms.Compose([transforms.ToTensor()]),
                                   num_queries=256, num_frames=32, specific_videos_list=video_names)
        dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True)
    elif ds_name == "lasot_oov":
        dataset = LaSOTDataset(ds_root,
                               transforms.Compose([transforms.ToTensor(), ResizeTensor(calibration_img_size=512)]),
                               num_frames=-1, specific_videos_list=specific_video_names)
        dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn_laSOT,
                        pin_memory=True)
    else:
        raise NotImplementedError(f"Dataset {ds_name} not implemented")

    return dataset, dl


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


def get_object_centred_persp_imgs_with_interpolation(eq_vid_frames, vid_seg_masks, mapper, batchsize=16):
    num_masks, num_frames = vid_seg_masks.shape[:2]
    device = vid_seg_masks.device

    # Placeholder for all centers: (num_masks, num_frames, 2)
    # Assuming compute_center_of_mass returns 2D coordinates (e.g., lat/lon or xy)
    all_centers = torch.zeros(num_masks, num_frames, 2, device=device)

    # Track which frames actually had a mask
    valid_frames_mask = torch.zeros(num_masks, num_frames, dtype=torch.bool, device=device)

    # --- Pass 1: Compute centers only for existing masks ---
    for bs in range(0, num_masks, batchsize):
        # Flatten batch frames: (batch_objects * num_frames, H, W)
        batch_masks_flat = vid_seg_masks[bs: bs + batchsize].flatten(0, 1)

        # Identify non-empty masks in this batch
        # (Assuming masks are binary or soft 0-1; sum > epsilon means it exists)
        mask_exists = batch_masks_flat.flatten(1).sum(-1) > 1e-4

        if mask_exists.any():
            # Only compute CoM for valid masks to avoid NaNs/errors
            valid_centers = compute_center_of_mass(batch_masks_flat[mask_exists])

            # Place results into the main storage
            # We use a temporary flat view to index easily with the flat 'mask_exists'
            batch_centers_flat = all_centers[bs: bs + batchsize].view(-1, 2)
            batch_centers_flat[mask_exists] = valid_centers.to(batch_centers_flat.dtype)

            batch_valid_mask_flat = valid_frames_mask[bs: bs + batchsize].view(-1)
            batch_valid_mask_flat[mask_exists] = True

    # --- Pass 2: Interpolate missing centers ---
    # Using CPU numpy for flexible 1D interpolation (linear middle, static ends)
    all_centers_np = all_centers.cpu().numpy()
    valid_mask_np = valid_frames_mask.cpu().numpy()
    all_frames_idx = np.arange(num_frames)

    for i in range(num_masks):
        mask_i = valid_mask_np[i]
        # If completely empty, skip (stays 0.0 or handle as needed)
        if not mask_i.any():
            continue
        # If fully dense, skip interpolation
        if mask_i.all():
            continue

        valid_idx = all_frames_idx[mask_i]
        valid_vals = all_centers_np[i][mask_i]

        # np.interp defaults: linear in middle, repeats LEFT value at start, repeats RIGHT value at end.
        # This satisfies "If it is never seen again keep static" (repeats rightmost value).
        interp_x = np.interp(all_frames_idx, valid_idx, valid_vals[:, 0])
        interp_y = np.interp(all_frames_idx, valid_idx, valid_vals[:, 1])

        all_centers_np[i] = np.stack([interp_x, interp_y], axis=1)

    # Move interpolated results back to GPU
    all_centers = torch.from_numpy(all_centers_np).to(device=device, dtype=all_centers.dtype)

    # --- Pass 3: Compute Rotation Matrices ---
    rot_matrices = torch.empty(num_masks, num_frames, 3, 3, device=device)
    for bs in range(0, num_masks, batchsize):
        batch_center_crs = all_centers[bs: bs + batchsize].flatten(0, 1)
        rot_matrices[bs: bs + batchsize] = mapper.compute_rotation_matrix_centred_at_point(
            batch_center_crs).view(-1, num_frames, 3, 3)

    # --- Generate Images ---
    centred_persp_imgs = []
    for i, rot_mat in enumerate(rot_matrices):
        centred_persp_imgs.append(mapper.image.equirectangular_image_to_perspective(eq_vid_frames, rot_mat))

    centred_persp_imgs = torch.stack(centred_persp_imgs, dim=0)

    return centred_persp_imgs, rot_matrices


def overlay_orig_persp_on_pred_eq(data, mapper, pred_eq_frames_torch, R_w2c, with_erode=True):
    warped_gt_persp_batch, valid_mask_batch = mapper.image.perspective_image_to_equirectangular(
        data.video[0].permute(0, 2, 3, 1),
        R_w2c.to(data.video.device),
        to_uint=False
    )
    warped_gt_persp_batch = warped_gt_persp_batch.clip(0, 1)
    warped_gt_persp_batch *= 255

    if with_erode:
        mask_float_with_channel = valid_mask_batch.float().unsqueeze(1)
        kernel_size = 3
        eroded_mask_float = 1.0 - torch.nn.functional.max_pool2d(
            1.0 - mask_float_with_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2
        ).squeeze(1)
    else:
        eroded_mask_float = valid_mask_batch
    valid_mask_broadcastable = eroded_mask_float.unsqueeze(-1).bool()

    return torch.where(valid_mask_broadcastable, warped_gt_persp_batch, pred_eq_frames_torch.float())
