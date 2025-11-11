import cv2
import numpy as np
import torch

from runners.bbox_utils import get_bbox_xyxy_from_points, bbox_iou


def basic_tracking(seg_masks, mapper, R_w2c):
    # This list stores the final, (potentially interpolated) bounding boxes
    pred_bboxes = []
    # This tracks the last valid bounding box found
    last_good_bbox = None
    # This tracks the index (frame number) where the last valid bbox was found
    last_good_idx = -1
    # This stores the index (frame number) of the *start* of the current gap
    gap_start_idx = -1
    for i, pred_mask in enumerate(seg_masks[0]):
        # --- 1. Check if the mask is empty ---
        is_empty = pred_mask.sum() == 0

        if is_empty:
            # 1a. Start tracking the gap if we just entered it
            if gap_start_idx == -1:
                gap_start_idx = i

            # 1b. Placeholder: Append None for now. We will fill this in later.
            pred_bboxes.append(None)

        else:
            # --- 2. Process a valid mask ---

            # 2a. Calculate the new bounding box (as per original logic)
            pred_eq_pnts = torch.vstack(torch.where(pred_mask)).permute(1, 0)
            rot_mat = R_w2c[i]
            pred_persp_points = mapper.point.cr.to_ij(pred_eq_pnts.flip(-1), rot_mat.to(pred_eq_pnts.device))

            try:
                pred_bbox = get_bbox_xyxy_from_points(pred_persp_points)
            except Exception:
                # If get_bbox_xyxy_from_points fails even with non-empty points, treat it as empty.
                print(f"Warning: BBox extraction failed at index {i}. Treating as empty mask.")
                pred_bboxes.append(None)

                if gap_start_idx == -1:
                    gap_start_idx = i

                # Skip the rest of the current iteration
                continue

            # 2b. A valid bbox was found: Handle interpolation for any preceding gap
            if gap_start_idx != -1:
                current_bbox = pred_bbox

                # Check if there was a previous good bbox to interpolate from
                if last_good_bbox is not None:

                    # Perform linear interpolation over the gap
                    gap_length = i - last_good_idx

                    # Interpolate from last_good_bbox to current_bbox
                    for j in range(gap_start_idx, i):
                        alpha = (j - last_good_idx) / gap_length

                        # Simple linear interpolation: BBOX_j = (1 - alpha) * BBOX_start + alpha * BBOX_end
                        interpolated_bbox = (1 - alpha) * last_good_bbox + alpha * current_bbox

                        # Fill the placeholder in pred_bboxes
                        pred_bboxes[j] = interpolated_bbox

                # Reset gap tracking
                gap_start_idx = -1

            # 2c. Update state and append the current good bbox
            pred_bboxes.append(pred_bbox)
            last_good_bbox = pred_bbox
            last_good_idx = i

    # --- 3. Handle trailing gap (if the video ends with empty masks) ---
    if gap_start_idx != -1 and last_good_bbox is not None:
        # If the gap is at the end of the sequence, just use the last good bbox (hold)
        for j in range(gap_start_idx, len(pred_bboxes)):
            pred_bboxes[j] = last_good_bbox

        # NOTE: If last_good_bbox is None, the whole sequence was empty, and all entries are None.

    return pred_bboxes


def get_lasot_bboxes(data, vid_seg_mean_above_conf, mapper, R_w2c):
    pred_bboxes = basic_tracking(vid_seg_mean_above_conf, mapper, R_w2c)
    pred_bboxes = torch.stack(pred_bboxes).cpu()

    pred_bboxes_clipped = pred_bboxes.clone()
    pred_bboxes_clipped[:, 0::2] = pred_bboxes_clipped[:, 0::2].clamp(0, data.video.shape[-1] - 1)
    pred_bboxes_clipped[:, 1::2] = pred_bboxes_clipped[:, 1::2].clamp(0, data.video.shape[-2] - 1)

    return pred_bboxes_clipped


def get_lasot_scores(data, pred_bboxes):
    gt_bboxes = data.bboxes_xyxy[0]

    iou_scores = bbox_iou(gt_bboxes, pred_bboxes).diag()
    vis_iou_scores = iou_scores[data.visibility[0].to(iou_scores.device).bool()]

    return iou_scores, vis_iou_scores
