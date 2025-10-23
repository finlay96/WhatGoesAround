# TODO maybe this should be in the main repo

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch

from conversions.mapper import Mappers


# from src.tapvid360.common.conversions.mapper import Mappers


def add_border_to_vis(vis: np.ndarray, border_size=(3, 3, 3, 3), border_col_val=127):
    """
    Adds a constant-colored border around the height and width dimensions
    of an image or video (supports shape (H, W[, C]), (T, H, W[, C]), (B, T, H, W[, C])).

    Parameters:
        vis: np.ndarray
        border_size: tuple(int, int, int, int) of form (pad_top, pad_bottom, pad_left, pad_right)
        border_col_val: int or float

    Returns:
        np.ndarray: padded array with border
    """
    pad_top, pad_bottom, pad_left, pad_right = border_size
    pad = [(0, 0)] * vis.ndim
    # Assume H and W are always at -3 and -2 positions
    pad[-3] = (pad_top, pad_bottom)  # H
    pad[-2] = (pad_left, pad_right)  # W

    return np.pad(vis, pad, mode='constant', constant_values=border_col_val)


def combine_vises(vises: list[np.ndarray] | np.ndarray, axis=-1):
    return np.concatenate(vises, axis=axis)


def add_text(frame: np.ndarray, txt: str, pos: tuple = (0, 0), font: int = cv2.FONT_HERSHEY_PLAIN, scale: float = 1.0,
             thickness: int = 1, txt_col: tuple = (0, 0, 0), bg_col: Optional[tuple] = None) -> np.ndarray:
    was_float = False
    if frame.dtype.kind == 'f':
        was_float = True
        frame = (frame * 255).astype(np.uint8)
    x, y = pos
    text_size, _ = cv2.getTextSize(txt, font, scale, thickness)
    text_w, text_h = text_size
    if bg_col is not None:
        cv2.rectangle(frame, (pos[0] - 5, pos[1] - 5), (x + text_w + 5, y + text_h + 5), bg_col, -1)
    cv2.putText(frame, txt, (x, int(y + text_h + 1)), font, scale, txt_col, thickness)

    if was_float:
        return frame / 255.0

    return frame


def draw_line(rgb, coord_y, coord_x, color, linewidth):
    draw = ImageDraw.Draw(rgb)
    draw.line(
        (coord_y[0], coord_y[1], coord_x[0], coord_x[1]),
        fill=tuple(color),
        width=linewidth,
    )
    return rgb


def pad_to_max_dimensions(*images):
    """Pads all images to the maximum height and width among them."""
    # Get max height and width
    heights = [img.shape[-3] for img in images]
    widths = [img.shape[-2] for img in images]
    target_h = max(heights)
    target_w = max(widths)

    def pad_image(img, target_h, target_w):
        h = img.shape[-3]
        w = img.shape[-2]
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left

        return add_border_to_vis(img, (pad_top, pad_bottom, pad_left, pad_right))

    return [pad_image(img, target_h, target_w) for img in images]


def add_weighted(rgb, alpha, original, beta, gamma):
    return (rgb * alpha + original * beta + gamma).astype("uint8")


class Cols:
    def __init__(self):
        cmap = plt.cm.get_cmap("tab20")
        self.colours = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in (cmap(i) for i in range(cmap.N))]

    def get_col(self, idx: int):
        return self.colours[idx % len(self.colours)]


def convert_to_numpy_img(img: torch.Tensor, scale_by: int = 255) -> np.ndarray:
    return (img * scale_by).cpu().numpy().astype(np.uint8)


class Visualiser:
    def __init__(self, tracks_leave_trace_len: int = 0, point_radius: int = 9):
        self.cols = Cols()
        self.tracks_leave_trace_len = tracks_leave_trace_len  # -1 is infinite
        self.linewidth: int = 2
        self.point_radius = point_radius

    def _draw_pred_tracks(self, rgb: np.ndarray, tracks: torch.Tensor, mask: torch.Tensor, alpha: float = 0.5):
        T, N, _ = tracks.shape
        rgb = Image.fromarray(np.uint8(rgb))
        for t in range(T - 1):
            original = rgb.copy()
            alpha = (t / T) ** 2
            for i in range(N):
                if not mask[i]:
                    continue
                coord_y = (int(tracks[t, i, 0]), int(tracks[t, i, 1]))
                coord_x = (int(tracks[t + 1, i, 0]), int(tracks[t + 1, i, 1]))
                if coord_y[0] != 0 and coord_y[1] != 0:
                    rgb = draw_line(rgb, coord_y, coord_x, self.cols.get_col(i), self.linewidth)
            if self.tracks_leave_trace_len > 0:
                rgb = Image.fromarray(np.uint8(add_weighted(np.array(rgb), alpha, np.array(original), 1 - alpha, 0)))
        rgb = np.array(rgb)
        return rgb

    def visualise(self, imgs, coords, mask: Optional[torch.Tensor] = None, vises: Optional[torch.Tensor] = None,
                  mapper=None, pad_value=300, query_frame=0):
        """_summary_

        Args:
            imgs (torch.Tensor): _description_
            coords (_type_): _description_
            mask (torch.Tensor): Boolean mask of shape (B, T, N)
            vises (torch.Tensor):Boolean mask of shape (B, T, N)
            mapper (_type_, optional): _description_. Defaults to None.
            pad_value (int, optional): _description_. Defaults to 300.

        Returns:
            _type_: _description_
        """
        if mask is None:
            mask = torch.ones_like(coords, dtype=torch.bool)[..., 0:1]
        if vises is None:
            vises = torch.ones_like(coords, dtype=torch.bool)[..., 0:1]
        if coords.shape[1] < imgs.shape[1]:
            rep_num = int(torch.ceil(torch.Tensor([imgs.shape[1] / coords.shape[1]])))
            mask = mask.repeat(1, rep_num, 1, 1)[:, :imgs.shape[1]]
            vises = vises.repeat(1, rep_num, 1, 1)[:, :imgs.shape[1]]
            # We will just repeat the points, but also set the mask to off so that they are not visualised.
            # In instances where you may only have the first frame gt
            mask[:, coords.shape[1]:] = 0
            vises[:, coords.shape[1]:] = 0
            coords = coords.repeat(1, rep_num, 1, 1)[:, :imgs.shape[1]]
        annos_per_batch = []
        for b in range(imgs.shape[0]):
            annos = []
            for img_idx, img in enumerate(imgs[b]):
                vis_img = self._run_vis_on_single_image(img, coords[b], mask[b], vises[b], img_idx, mapper[b],
                                                        pad_value, query_frame)

                annos.append(vis_img)
            annos_per_batch.append(annos)

        return np.stack(annos_per_batch)

    def _run_vis_on_single_image(self, img, vis_coords, mask, vises, img_idx, mapper, pad_value, query_frame=0):
        if img.shape[-1] != 3:
            img = img.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
        if vis_coords.shape[-1] > 2:
            img, pad_value, vis_coords = self.setup_image_for_uv(img, mapper, vis_coords)
        img = self._convert_to_numpy(img)
        vis_img = img.copy()
        vis_coords, vis_img = self.pad_image(pad_value, vis_coords, vis_img)
        vis_img = self._vis_track_traces(img_idx, mask, query_frame, vis_coords, vis_img)
        self.draw_points(img_idx, mask, vis_coords, vis_img, vises)
        return vis_img

    def draw_points(self, img_idx, mask, vis_coords, vis_img, vises):
        for i, (pnt, m) in enumerate(zip(vis_coords[img_idx], mask[img_idx])):
            if not m:
                continue
            cv2.circle(vis_img, (int(pnt[0]), int(pnt[1])), self.point_radius, self.cols.get_col(i),
                       -1 if vises[img_idx, i] else 2)

    def _vis_track_traces(self, img_idx, mask, query_frame, vis_coords, vis_img):
        if self.tracks_leave_trace_len != 0:
            if query_frame > img_idx:
                raise ValueError("Query frame must be less than the image index")
            vis_img = self._draw_pred_tracks(vis_img, vis_coords[query_frame:img_idx], mask[img_idx])
        return vis_img

    def pad_image(self, pad_value, vis_coords, vis_img):
        if pad_value > 0:
            vis_coords = vis_coords + pad_value
            vis_img = np.pad(vis_img, ((pad_value, pad_value), (pad_value, pad_value), (0, 0)),
                             mode='constant', constant_values=255)
        return vis_coords, vis_img

    def _convert_to_numpy(self, img):
        if isinstance(img, torch.Tensor):
            img = convert_to_numpy_img(img)
        return img

    def setup_image_for_uv(self, img, mapper, vis_coords):
        if img.ndim == 3:
            img = img.unsqueeze(0)
        img, _ = mapper.image.perspective_image_to_equirectangular_ego(img)
        img = img[0]

        vis_coords = mapper.point.vc.to_cr(vis_coords, torch.eye(3).to(vis_coords.device))
        pad_value = 0
        return img, pad_value, vis_coords

    def save_visualisations(self, vises: list[np.ndarray] | np.ndarray, vid_names: list, out_dir: Path):
        # Each instance of vises to be of shape (B, T, H, W, C) nd vid_names to be of len (B)
        out_dir.mkdir(exist_ok=True)
        if isinstance(vises, np.ndarray):
            vises = [vises]
        assert all(v.shape == vises[0].shape for v in vises), "All visualisations must have the same shape"
        assert len(vid_names) == vises[0].shape[0], "Number of video names must match the batch size"


def visualize_perspective_view(
        equirectangular_image: torch.Tensor,
        rotation_matrix: torch.Tensor,
        mapper: 'Mappers',
        mode: str = 'border',
        border_color: tuple = (255.0, 0.0, 0.0),
        border_thickness: int = 3,  # Thickness works by dilating the edge
        dim_factor: float = 0.5,  # Controls brightness of the masked area
        ego_centric: bool = False  # The new flag to control the view
) -> torch.Tensor:
    """
    Visualizes a perspective crop on an equirectangular image.

    Can operate in three modes:
    1. 'border': Draws a coloured border representing the perspective view.
    2. 'mask': Dims and greys out the area outside the perspective view.
    3. 'both': Applies the mask and then draws the border on top.

    Args:
        equirectangular_image (torch.Tensor): Shape (B, H_eq, W_eq, C).
        rotation_matrix (torch.Tensor): The R_w2c rotation matrix. Shape (B, 3, 3).
        mapper (Mappers): Mapper object with conversion function and config.
        mode (str): 'border', 'mask', or 'both'.
        border_color (tuple): RGB tuple for the border color.
        border_thickness (int): The thickness of the border in pixels.
        dim_factor (float): Brightness multiplier for the masked-out area (0.0 to 1.0).
        ego_centric (bool): If True, rotates the image to center the perspective view.

    Returns:
        torch.Tensor: The modified equirectangular image.
    """
    B, H_eq, W_eq, C = equirectangular_image.shape
    device = equirectangular_image.device

    if mode not in ('border', 'mask', 'both'):
        raise ValueError(f"Unknown mode '{mode}'. Choose 'border', 'mask', or 'both'.")

    # --- Ego-centric View Handling ---
    if ego_centric:
        output_image = mapper.image.equirectangular_image_to_equirectangular_ego(equirectangular_image, rotation_matrix)
        # # 1. Determine the camera's pointing direction (yaw, pitch) from the rotation matrix.
        # # The forward vector of the camera in world coordinates is the first column of R_w2c.T
        # forward_vec = torch.transpose(rotation_matrix, 1, 2)[:, :, 0]
        #
        # yaw = torch.atan2(forward_vec[:, 1], forward_vec[:, 0])  # Azimuth
        # pitch = torch.asin(torch.clamp(forward_vec[:, 2], -1.0, 1.0))  # Elevation, clamped for stability
        #
        # # 2. Convert angles to pixel shifts for rolling the image.
        # horizontal_shift = yaw / (2 * torch.pi) * W_eq
        # vertical_shift = pitch / torch.pi * H_eq
        #
        # # 3. Roll the base image to place the camera's view in the center.
        # # We loop here because torch.roll doesn't support per-item shifts in a batch.
        # rolled_images = []
        # for i in range(B):
        #     # The shifts are negative to bring the target direction TO the center of the image.
        #     shifts_for_roll = (-int(vertical_shift[i].round()), -int(horizontal_shift[i].round()))
        #     # Dims (0, 1) correspond to (H, W) for a single image in the batch.
        #     rolled_img = torch.roll(equirectangular_image[i], shifts=shifts_for_roll, dims=(0, 1))
        #     rolled_images.append(rolled_img)
        # output_image = torch.stack(rolled_images, dim=0)

        # 4. To draw the outline in the center, we use an identity matrix for the visualization.
        viz_rotation_matrix = torch.eye(
            3, device=device, dtype=rotation_matrix.dtype
        ).unsqueeze(0).expand(B, -1, -1)
    else:
        # Standard behaviour: use original image and rotation.
        output_image = equirectangular_image.clone()
        viz_rotation_matrix = rotation_matrix

    # --- Step 1: Calculate the valid area mask (SHARED LOGIC for all modes) ---
    c_grid = torch.linspace(0, W_eq - 1, W_eq, device=device)
    r_grid = torch.linspace(0, H_eq - 1, H_eq, device=device)
    mg_c, mg_r = torch.meshgrid(c_grid, r_grid, indexing='xy')
    cr_grid = torch.stack([mg_c, mg_r], dim=-1)

    cr_grid_batched = cr_grid.unsqueeze(0).expand(B, -1, -1, -1)
    # Use the appropriate rotation matrix for visualization
    vc_grid = mapper.point.cr.to_vc(cr_grid_batched, viz_rotation_matrix)
    ij_grid = mapper.point.vc.to_ij(vc_grid)

    valid_i = (ij_grid[..., 0] >= 0) & (ij_grid[..., 0] < mapper.image.cfg.crop_width)
    valid_j = (ij_grid[..., 1] >= 0) & (ij_grid[..., 1] < mapper.image.cfg.crop_height)
    front_facing_mask = vc_grid[..., 0] > 1e-6

    valid_mask = (front_facing_mask & valid_i & valid_j)  # (B, H_eq, W_eq)

    # --- Step 2: Apply effects based on the selected mode ---

    # Apply the dimming/greyscale mask if mode is 'mask' or 'both'
    if mode in ('mask', 'both'):
        # Use the (potentially rolled) output_image as the base for greyscaling
        greyscale_values = torch.mean(
            output_image.float(), dim=-1, keepdim=True
        )
        dimmed_greyscale = (greyscale_values * dim_factor).to(output_image.dtype)
        output_image = torch.where(valid_mask.unsqueeze(-1), output_image, dimmed_greyscale)

    # Draw the border if mode is 'border' or 'both'
    if mode in ('border', 'both'):
        mask_float = valid_mask.float().unsqueeze(1)
        # Dilate and erode to find the edge
        dilated_mask = torch.nn.functional.max_pool2d(mask_float, kernel_size=3, stride=1, padding=1)
        eroded_mask = -torch.nn.functional.max_pool2d(-mask_float, kernel_size=3, stride=1, padding=1)
        border_mask = (dilated_mask - eroded_mask).bool()

        # Apply thickness by dilating the border mask itself
        if border_thickness > 1:
            kernel_size = border_thickness if border_thickness % 2 != 0 else border_thickness + 1
            padding = kernel_size // 2
            border_mask = torch.nn.functional.max_pool2d(border_mask.float(), kernel_size, stride=1,
                                                         padding=padding).bool()

        color_tensor = torch.tensor(border_color, device=device, dtype=output_image.dtype).view(1, 1, 1, C)
        output_image = torch.where(border_mask.permute(0, 2, 3, 1), color_tensor, output_image)

    return output_image
