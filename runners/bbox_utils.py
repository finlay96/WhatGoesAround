import numpy as np
import torch


def xywh_to_xyxy(bboxes_xywh: np.ndarray | torch.Tensor):
    """
    Converts an array of bounding boxes from [x, y, w, h] format
    to [x1, y1, x2, y2] format.

    Args:
      bboxes_xywh: A NumPy | Torch array of shape (N, 4) where N is the
                     number of boxes.

    Returns:
      A NumPy | Torch array of shape (N, 4) in [x1, y1, x2, y2] format.
    """
    if isinstance(bboxes_xywh, torch.Tensor):
        bboxes_xyxy = bboxes_xywh.clone()
    else:
        bboxes_xyxy = bboxes_xywh.copy()
    # x2 = x + w
    bboxes_xyxy[:, 2] = bboxes_xywh[:, 0] + bboxes_xywh[:, 2]
    # y2 = y + h
    bboxes_xyxy[:, 3] = bboxes_xywh[:, 1] + bboxes_xywh[:, 3]
    return bboxes_xyxy


def get_bbox_xyxy_from_points(points: torch.Tensor):
    xmin_ymin = torch.min(points, dim=0).values
    xmax_ymax = torch.max(points, dim=0).values

    return torch.cat([xmin_ymin, xmax_ymax])


def get_bbox_xyxy_from_mask(masks: torch.Tensor):
    """
    Calculates bounding boxes from a batch of binary masks.

    Args:
        masks (torch.Tensor): A boolean or 0-1 tensor of shape [N, H, W].

    Returns:
        torch.Tensor: A tensor of shape [N, 4] with boxes in [xmin, ymin, xmax, ymax] format.
    """
    if masks.ndim == 2:
        # Add a batch dimension if a single mask is provided
        masks = masks.unsqueeze(0)

    N, H, W = masks.shape
    device = masks.device

    # Find rows and columns that contain any non-zero value
    rows_with_object = torch.any(masks, dim=2)  # Shape [N, H]
    cols_with_object = torch.any(masks, dim=1)  # Shape [N, W]

    # Find the min/max indices.
    # argmax() returns the *first* True index.
    # For ymax/xmax, we flip the tensor and find the first True index from the end.

    # Initialise boxes with zeros (for empty masks)
    boxes = torch.zeros((N, 4), dtype=torch.int64, device=device)

    # Find indices for non-empty masks
    # torch.any(tensor, dim=(1,2)) checks if *any* pixel is True in each H,W plane
    non_empty_indices = torch.any(masks, dim=(1, 2))

    if non_empty_indices.any():
        # Only compute for masks that are not empty
        rows_nz = rows_with_object[non_empty_indices]
        cols_nz = cols_with_object[non_empty_indices]

        rows_nz_int = rows_nz.to(torch.uint8)
        cols_nz_int = cols_nz.to(torch.uint8)

        # Find ymin and xmin
        ymin = torch.argmax(rows_nz_int, dim=1)
        xmin = torch.argmax(cols_nz_int, dim=1)

        # Find ymax and xmax
        ymax = (H - 1) - torch.argmax(torch.flip(rows_nz_int, dims=[1]), dim=1)
        xmax = (W - 1) - torch.argmax(torch.flip(cols_nz_int, dims=[1]), dim=1)

        # Store results
        boxes[non_empty_indices] = torch.stack([xmin, ymin, xmax, ymax], dim=1)

    return boxes


def bbox_iou(boxes1, boxes2):
    """
    Calculates the Intersection over Union (IoU) between two sets of bounding boxes.

    Supports any number of leading batch dimensions.
    Assumes boxes are in [xmin, ymin, xmax, ymax] format.

    Args:
        boxes1 (torch.Tensor): A tensor of shape (..., N, 4).
        boxes2 (torch.Tensor): A tensor of shape (..., M, 4).

    Returns:
        torch.Tensor: A tensor of shape (..., N, M) where element [..., i, j]
                      is the IoU of boxes1[..., i, :] and boxes2[..., j, :].
    """

    # 1. Get the areas of each box
    # Slicing from the end works for any ndim
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # (..., N)
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # (..., M)

    # 2. Find the coordinates of the intersection boxes
    # We need to broadcast (..., N, 4) and (..., M, 4) to (..., N, M, 4)

    # b1 becomes (..., N, 1, 4)
    b1 = boxes1.unsqueeze(-2)
    # b2 becomes (..., 1, M, 4)
    b2 = boxes2.unsqueeze(-3)

    # Broadcasting (..., N, 1, 4) and (..., 1, M, 4) gives (..., N, M, 4)
    # We then slice the last dim. The result for each is (..., N, M)
    inter_xmin = torch.max(b1[..., 0], b2[..., 0])
    inter_ymin = torch.max(b1[..., 1], b2[..., 1])
    inter_xmax = torch.min(b1[..., 2], b2[..., 2])
    inter_ymax = torch.min(b1[..., 3], b2[..., 3])

    # 3. Calculate intersection area
    inter_width = (inter_xmax - inter_xmin).clamp(min=0)  # (..., N, M)
    inter_height = (inter_ymax - inter_ymin).clamp(min=0)  # (..., N, M)
    intersection = inter_width * inter_height  # (..., N, M)

    # 4. Calculate union area
    # area1 (..., N) -> unsqueeze to (..., N, 1)
    # area2 (..., M) -> unsqueeze to (..., 1, M)
    # Broadcasting gives (..., N, M)
    union = area1.unsqueeze(-1) + area2.unsqueeze(-2) - intersection

    # 5. Calculate IoU
    iou = intersection / (union + 1e-6)

    return iou
