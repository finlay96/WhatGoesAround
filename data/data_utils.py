from dataclasses import dataclass, fields
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision import transforms


def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])


@dataclass(eq=False)
class DataSample:
    """
    Dataclass for storing video tracks data.
    """

    video: torch.Tensor  # B, S, C, H, W
    trajectory: torch.Tensor  # B, S, N, 2/3
    seq_name: list[str] | str = None
    # optional data
    visibility: Optional[torch.Tensor] = None  # B, S, N
    valid: Optional[torch.Tensor] = None  # B, S, N
    segmentation: Optional[torch.Tensor] = None  # B, S, 1, H, W
    query_points: Optional[torch.Tensor] = None  # TapVID evaluation format
    equirectangular_height: Optional[int] = None
    fov_x: Optional[float] = None
    # mapper: Optional[Mappers | list[Mappers]] = None,
    rotations: Optional[torch.Tensor] = None,  # B, S, 3, 3
    orig_image_shapes: Optional[list] = None

    def to_device(self, device: torch.device):
        """
        Moves all torch.Tensor attributes to the specified device.
        """
        # Iterate over all fields of the dataclass
        for field in fields(self):  # fields(self) gives access to the dataclass fields
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):  # If the field is a tensor
                setattr(self, field.name, value.to(device))  # Move tensor to device


def collate_fn(data: list[DataSample]):
    return DataSample(
        video=torch.stack([d.video for d in data]),
        trajectory=torch.stack([d.trajectory for d in data]),
        rotations=torch.stack([d.rotations for d in data]) if data[0].rotations is not None else None,
        seq_name=[d.seq_name for d in data],
        equirectangular_height=[d.equirectangular_height for d in data
                                ] if data[0].equirectangular_height is not None else None,
        fov_x=[d.fov_x for d in data] if data[0].fov_x is not None else None,
        segmentation=torch.stack([d.segmentation for d in data]) if data[0].segmentation is not None else None,
        visibility=torch.stack([d.visibility for d in data]) if data[0].visibility is not None else None,
    )


class ResizeTensor:
    """
    A transform class to resize a tensor, mirroring the logic
    from the user's cv2.resize function.

    Args:
        width (int, optional): Target width.
        height (int, optional): Target height.
        calibration_img_size (int, optional): The target size for the
                                              longest side.
    """

    def __init__(self, width=None, height=None, calibration_img_size=None):
        self.width = width
        self.height = height
        self.calibration_img_size = calibration_img_size

    def __call__(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Tensor to be resized,
                                   expected shape (C, H, W).

        Returns:
            torch.Tensor: Resized tensor.
        """

        # --- Logic 1: Resize to specific width and height ---
        if self.width is not None and self.height is not None:
            # Note: F.interpolate size is (height, width)
            size = (self.height, self.width)

            # F.interpolate needs a batch dimension (B, C, H, W)
            # We add one with unsqueeze(0) and remove it with squeeze(0)
            return F.interpolate(tensor.unsqueeze(0),
                                 size=size,
                                 mode='bilinear',
                                 align_corners=False).squeeze(0)

        # --- Logic 2: Resize longest side to calibration_img_size ---
        if self.calibration_img_size is not None:
            # Get current (H, W) from the tensor shape (C, H, W)
            h, w = tensor.shape[-2:]

            if h > w:
                new_h = self.calibration_img_size
                new_w = int(w * self.calibration_img_size / h)
            else:
                new_w = self.calibration_img_size
                new_h = int(h * self.calibration_img_size / w)

            size = (new_h, new_w)
            return F.interpolate(tensor.unsqueeze(0),
                                 size=size,
                                 mode='bilinear',
                                 align_corners=False).squeeze(0)

        # --- Logic 3: No resize specified ---
        else:
            return tensor


def scale_bboxes(persp_img_shapes, bboxes_xywh, new_h, new_w):
    scaled_bboxes_xywh = []
    # Ensure we only process as many frames as we have images AND bboxes
    num_frames = min(len(persp_img_shapes), len(bboxes_xywh))

    for i in range(num_frames):
        # Get original dimensions (W, H)
        orig_w, orig_h = persp_img_shapes[i]

        # Get original bbox [x, y, w, h]
        bbox = bboxes_xywh[i]

        # Calculate scaling factors
        # Note: Ensure floating point division
        w_scale = new_w / float(orig_w)
        h_scale = new_h / float(orig_h)

        # Apply scaling
        new_x = bbox[0] * w_scale
        new_y = bbox[1] * h_scale
        new_w_bbox = bbox[2] * w_scale
        new_h_bbox = bbox[3] * h_scale

        scaled_bboxes_xywh.append([new_x, new_y, new_w_bbox, new_h_bbox])

    return torch.tensor(scaled_bboxes_xywh, dtype=torch.float32)
