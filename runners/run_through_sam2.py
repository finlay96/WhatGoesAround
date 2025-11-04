from PIL import Image
from accelerate import Accelerator
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from runners.bbox_utils import get_bbox_xyxy_from_points, get_bbox_xyxy_from_mask, bbox_iou
from runners.model_utils import SAMRunner, get_models
from runners.run_argus_cotracker import Settings
from runners.utils import get_dataset
from runners.vis_utils import overlay_mask_over_image


@torch.no_grad()
def main(device, settings):
    # dataset, dl = get_dataset(settings.ds_root, settings.ds_name, settings.specific_video_names)
    dataset, dl = get_dataset(settings.ds_root, settings.ds_name, None)

    accelerator = Accelerator(mixed_precision='no')
    _, sam_pred_video, sam_pred_image, _ = get_models(settings.sam_checkpoint, accelerator, accelerator.device,
                                                      debug_skip_vae=True)
    sam_runner = SAMRunner(sam_img_pred=sam_pred_image, sam_video_pred=sam_pred_video)
    m_vis_ious = []
    for data in tqdm(dl):
        bbox_xyxy = data.bboxes_xyxy[:, 0].cpu().tolist()
        first_mask = sam_runner.run_through_image(
            (data.video[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
            bbox_xyxy=bbox_xyxy, positive_points=None, negative_points=None)
        frames = (data.video[0, :].permute(0, 2, 3, 1) * 255).to(torch.uint8)
        _, confs = sam_runner.run_through_video(frames, masks=torch.from_numpy(first_mask)[None][None])
        vid_seg_mean_above_conf = confs > 0.1
        pred_bboxes_xyxy = get_bbox_xyxy_from_mask(vid_seg_mean_above_conf[0])
        gt_bboxes = data.bboxes_xyxy[0].to(device)
        iou_scores = bbox_iou(gt_bboxes, pred_bboxes_xyxy).diag()
        vis_iou_scores = iou_scores[data.visibility[0].to(device).bool()]
        lost_item = False
        if (vis_iou_scores < 0.05).any():
            lost_item = True

        vid_out_dir = settings.out_root / "sam2_track_outputs" / settings.ds_name / f"lost_item-{lost_item}" / data.seq_name[0]
        vid_out_dir.mkdir(exist_ok=True, parents=True)
        for fidx in range(frames.shape[0]):
            frame_img = frames[fidx].cpu().numpy().copy()
            frame_img = overlay_mask_over_image(frame_img, vid_seg_mean_above_conf[0, fidx].cpu().numpy())
            Image.fromarray(frame_img).save(vid_out_dir / f"{fidx}.png")

        m_vis_ious.append(vis_iou_scores.mean())

    print("Overall mean IOU on visible boxes:", torch.stack(m_vis_ious).mean().item())


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    settings = Settings()
    main(device, settings)
