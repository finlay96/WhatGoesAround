from pathlib import Path

import cv2
from decord import VideoReader
from decord import cpu, gpu  # Cannot get gpu installed properly
import numpy as np


class DecordVideoReader:
    def __init__(self, video_path: Path | str, use_gpu: bool = False):
        self.vr = VideoReader(str(video_path), ctx=gpu(0) if use_gpu else cpu(0))  # can set to cpu or gpu .. ctx=gpu(0)

    def count_frames(self) -> int:
        return len(self.vr)

    def load_frames(self, frame_indices: list[int] | None = None) -> np.ndarray:
        if frame_indices is None:
            frame_indices = list(range(self.count_frames()))
        return self.vr.get_batch(frame_indices).asnumpy()


def draw_point(img, pnt, col=(0, 255, 0), is_vis=True, radius=9):
    cv2.circle(img, (int(pnt[0]), int(pnt[1])), radius, col, -1 if is_vis else 2)
