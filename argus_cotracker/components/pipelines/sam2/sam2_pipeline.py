"""
Altered to allow for frames to just be inputted and not needing a directory
"""

from collections import OrderedDict
import types
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

# from src.tapvid360.utils.general_utils import suppress_output


def run_through_sam_video(sam_pred_video, frames: torch.Tensor, masks: torch.Tensor):
    vid_sam_frames, pp_sam_masks = sam_preprocessing(frames, masks,
                                                     resize_to=(sam_pred_video.image_size, sam_pred_video.image_size))
    assert vid_sam_frames is not None, "Need to pass in all the frames for video sam"
    inference_state = sam_pred_video.init_state(vid_sam_frames, frames[0].shape[0],
                                                frames[0].shape[1], sam_pred_video.device)
    for i, mask in enumerate(pp_sam_masks):
        sam_pred_video.add_new_mask(inference_state, 0, i + 1, mask)
    # with suppress_output():
    out_logits = [out_mask_logits for _, _, out_mask_logits in sam_pred_video.propagate_in_video(inference_state)]
    confidence_scores = [torch.sigmoid(logits) for logits in out_logits]
    assert len(out_logits), "No masks generated"

    return torch.stack(out_logits)[:, :, 0].transpose(1, 0), torch.stack(confidence_scores)[:, :, 0].transpose(1, 0)

    # with suppress_output():
    #     vid_seg_masks = torch.stack(
    #         [out_mask_logits[:, 0] > 0 for _, _, out_mask_logits in sam_pred_video.propagate_in_video(inference_state)])
    # assert len(vid_seg_masks), "No masks generated"
    #
    # return vid_seg_masks


@torch.inference_mode()
def init_state(
        self,
        images,
        video_height,
        video_width,
        compute_device,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
):
    """Initialize an inference state."""
    inference_state = {}
    inference_state["images"] = images
    inference_state["num_frames"] = len(images)
    # whether to offload the video frames to CPU memory
    # turning on this option saves the GPU memory with only a very small overhead
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    # whether to offload the inference state to CPU memory
    # turning on this option saves the GPU memory at the cost of a lower tracking fps
    # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
    # and from 24 to 21 when tracking two objects)
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    # the original video height and width, used for resizing final output scores
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = compute_device
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = compute_device
    # inputs on each frame
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    # visual features on a small number of recently visited frames for quick interactions
    inference_state["cached_features"] = {}
    # values that don't change across frames (so we only need to hold one copy of them)
    inference_state["constants"] = {}
    # mapping between client-side object id and model-side object index
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    # A storage to hold the model's tracking results and states on each frame
    inference_state["output_dict"] = {
        "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
    }
    # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
    inference_state["output_dict_per_obj"] = {}
    # A temporary storage to hold new outputs when user interact with a frame
    # to add clicks or mask (it's merged into "output_dict" before propagation starts)
    inference_state["temp_output_dict_per_obj"] = {}
    # Frames that already holds consolidated outputs from click or mask inputs
    # (we directly use their consolidated outputs during tracking)
    inference_state["consolidated_frame_inds"] = {
        "cond_frame_outputs": set(),  # set containing frame indices
        "non_cond_frame_outputs": set(),  # set containing frame indices
    }
    # metadata for each tracking frame (e.g. which direction it's tracked)
    inference_state["tracking_has_started"] = False
    inference_state["frames_already_tracked"] = {}
    inference_state["frames_tracked_per_obj"] = {}

    # Warm up the visual backbone and cache the image feature on frame 0
    self._get_image_feature(inference_state, frame_idx=0, batch_size=1)

    return inference_state


def sample_points(mask_to_use: np.ndarray, max_sample_points: int):
    pos_points = np.column_stack(np.where(mask_to_use)[::-1])
    if (max_sample_points > 0) and (len(pos_points) > max_sample_points):
        sampled_indices = sorted(np.random.choice(pos_points.shape[0], size=max_sample_points, replace=False))
        pos_points = pos_points[sampled_indices]
    pos_labels = np.ones(len(pos_points))

    return pos_points, pos_labels


def get_sam2_predictor(single_image: bool = False, model_cfg: str = "sam2_hiera_l.yaml",
                       checkpoint: str | Path = "sam2_hiera_large.pt", device: torch.device = torch.device("cuda")):
    if single_image:
        return SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))
    pred = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    # Monkey patch the new init state method to allow for frames to just be fed to the method
    pred.init_state = types.MethodType(init_state, pred)

    return pred


def predict_masks_sam(sam_pred, query_mask, running_frames, img_shape):
    inference_state = sam_pred.init_state(running_frames, img_shape[0], img_shape[1], sam_pred.device)
    _, _, out_mask_logits = sam_pred.add_new_mask(inference_state, 0, 1, query_mask)
    video_segments = [(out_mask_logits[0, 0] > 0.0).cpu().numpy() for _, _, out_mask_logits in
                      sam_pred.propagate_in_video(inference_state)]
    sam_pred.reset_state(inference_state)

    return video_segments


def sam_preprocessing(frames: torch.Tensor, mask: torch.Tensor, resize_to: tuple) -> Tuple[
    torch.Tensor, torch.Tensor]:
    if not len(frames):
        raise ValueError("No frames given")
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)

    frames_tens = frames.permute(0, 3, 1, 2)

    frames_tens_scaled = frames_tens / 255.0
    frames_tens_scaled = torch.nn.functional.interpolate(frames_tens_scaled, size=resize_to, mode='bilinear',
                                                         align_corners=False)

    # normalize by mean
    frames_tens_scaled -= torch.tensor(img_mean, dtype=torch.float32)[:, None, None].to(frames_tens_scaled.device)

    # normalize by std
    frames_tens_scaled /= torch.tensor(img_std, dtype=torch.float32)[:, None, None].to(frames_tens_scaled.device)

    return frames_tens_scaled, torch.nn.functional.interpolate(mask.float(), size=resize_to, mode='bilinear',
                                                               align_corners=False)[:, 0]


def run_through_img_sam(sam_pred, positive_points: list, negative_points: list, img: np.ndarray):
    # Convert points to NumPy array
    positive_points_np = np.array(positive_points)
    negative_points_np = np.array(negative_points)

    # Prepare image for SAM model
    sam_pred.set_image(img)

    # Create input labels for SAM: 1 for positive, 0 for negative
    input_labels = np.array([1] * len(positive_points_np) + [0] * len(negative_points_np))

    # Concatenate positive and negative points
    if len(negative_points_np):
        input_points = np.concatenate([positive_points_np, negative_points_np], axis=0)
    else:
        input_points = positive_points_np

    # Perform prediction
    masks, scores, logits = sam_pred.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )
    mask = masks[np.argmax(scores)]

    return mask.astype(np.uint8) * 255
