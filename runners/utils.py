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
            with open(Path(__file__).parent.parent / "data/mini_dataset_names_100_items.txt", "r") as f:
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


def overlay_orig_persp_on_pred_eq(data, mapper, pred_eq_frames_torch, R_w2c):
    warped_gt_persp_batch, valid_mask_batch = mapper.image.perspective_image_to_equirectangular(
        data.video[0].permute(0, 2, 3, 1) * 255,
        R_w2c.to(data.video.device),
        to_uint=False
    )
    valid_mask_broadcastable = valid_mask_batch.unsqueeze(-1).bool()

    return torch.where(valid_mask_broadcastable, warped_gt_persp_batch, pred_eq_frames_torch)
