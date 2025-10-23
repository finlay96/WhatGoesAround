from pathlib import Path
from typing import Optional

import numpy as np
import torch

# from cotracker.utils.visualizer import Visualizer


def scale_points(points, old_size, new_size):
    """
    Scale a list of (y, x) points from old_size to new_size.

    :param points: Tensor of (y, x) coordinates
    :param old_size: Tuple (old_width, old_height)
    :param new_size: Tuple (new_width, new_height)
    :return: Scaled points
    """
    old_width, old_height = old_size
    new_width, new_height = new_size

    scale_x = new_width / old_width
    scale_y = new_height / old_height

    points = points.float()

    points[..., 0] *= scale_x
    points[..., 1] *= scale_y

    return points.to(int)


class CoTracker:
    def __init__(self, device=torch.device("cuda:0"), grid_size=10, interp_shape=(384, 512)):
        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device).eval()
        self.grid_size = grid_size
        self.interp_shape = interp_shape

    def preprocess(self, video: torch.Tensor, seg_mask: Optional[np.ndarray] = None,
                   query_points: Optional[torch.Tensor] = None, device=torch.device("cpu")):
        # video of shape # B T C H W
        # segm mask of shape # H W
        # query of shape B, N, 3
        if video.ndim == 4:
            video = video.unsqueeze(0)
        videos = video.permute(0, 1, 4, 2, 3).float().to(device)
        if self.interp_shape is not None:
            videos_resize = []
            for v in videos:
                videos_resize.append(torch.nn.functional.interpolate(v, tuple(self.interp_shape), mode="bilinear"))
            videos_resize = torch.stack(videos_resize)
            segm_mask_resize = None
            if seg_mask is not None:
                segm_mask_resize = torch.nn.functional.interpolate(torch.from_numpy(seg_mask).float()[None][None],
                                                                   tuple(self.interp_shape), mode="bilinear")
            if query_points is not None:
                query_points = self.rescale_points(query_points.flip(-1), [videos.shape[-1], videos.shape[-2]],
                                                   (self.interp_shape[1], self.interp_shape[0]))
        else:
            videos_resize = videos
            segm_mask_resize = torch.from_numpy(seg_mask).float()[None][None]

        if query_points is not None:
            if query_points.ndim == 2:
                query_points = query_points.unsqueeze(0)
            query_points = torch.cat([torch.zeros_like(query_points[..., 0:1]), query_points], dim=-1).float().to(
                videos_resize.device)

        return videos_resize, segm_mask_resize, query_points

    @torch.no_grad()
    def run(self, video: torch.Tensor, seg_mask: Optional[torch.Tensor] = None,
            query_points: Optional[torch.Tensor] = None):

        pred_tracks, pred_visibility = self.model(video, segm_mask=seg_mask, queries=query_points,
                                                  grid_size=self.grid_size)  # B T N 2,  B T N 1

        return pred_tracks, pred_visibility

    @staticmethod
    def rescale_points(points, old_size, new_size):
        return scale_points(points, old_size, new_size)

    @staticmethod
    def visualize(video: torch.Tensor, pred_tracks: torch.Tensor, pred_visibility: torch.Tensor,
                  save_dir: Path | str = "./saved_videos", pad_value=120):
        vis = Visualizer(save_dir=save_dir, pad_value=pad_value, linewidth=3)
        vis.visualize(video, pred_tracks, pred_visibility, opacity=0.7)
