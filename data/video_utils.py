import os
import tempfile
from pathlib import Path
from typing import Optional, List

from decord import VideoReader
from decord import cpu, gpu  # Cannot get gpu installed properly
import cv2
import numpy as np


class DecordVideoReader:
    def __init__(self, video_path: Path | str, use_gpu: bool = False):
        self.vr = VideoReader(str(video_path), ctx=gpu(0) if use_gpu else cpu(0))  # can set to cpu or gpu .. ctx=gpu(0)

    def count_frames(self) -> int:
        return len(self.vr)

    def load_frames(self, frame_indices: list[int]) -> np.ndarray:
        return self.vr.get_batch(frame_indices).asnumpy()


def encode_frames_to_video_bytes(frame_list, fps=30):
    """
    Encodes a list of NumPy frames into a video byte stream in memory.

    Args:
        frame_list (list): A list of frames, where each frame is a NumPy array
                           of shape (H, W, 3) in RGB order and dtype uint8.
        fps (int): Frames per second for the output video.

    Returns:
        bytes: A byte string containing the compressed video data.
    """
    if not len(frame_list):
        return b''

    height, width, _ = frame_list[0].shape

    # We need to write to a temporary file on disk because OpenCV's VideoWriter
    # requires a file path. We then read the bytes from this file.
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        temp_filename = tmp.name

    # Define the codec (H.264 is a great choice) and create VideoWriter object
    # 'avc1' or 'h264' are common FourCCs for H.264. 'mp4v' is also very compatible.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_filename, fourcc, fps, (width, height))

    for frame in frame_list:
        # OpenCV expects BGR format, so we convert from RGB
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()

    # Read the compressed video data from the temporary file
    with open(temp_filename, 'rb') as f:
        video_bytes = f.read()

    # Clean up the temporary file
    os.remove(temp_filename)

    return video_bytes


# --- DECODING FUNCTION ---
def decode_video_bytes_to_frames(video_bytes: bytes, frame_numbers: Optional[List[int]] = None) -> List[np.ndarray]:
    """
    Decodes specific frames from a video byte stream by frame number.

    This updated function uses direct frame seeking, which is much more
    efficient for accessing non-sequential frames than iterating from the
    beginning of the video.

    Args:
        video_bytes (bytes): A byte string containing compressed video data.
        frame_numbers (Optional[List[int]]): An optional list of integer frame numbers to decode.

    Returns:
        List[np.ndarray]: List of frames
    """
    if not video_bytes:
        return []

    # Write the bytes to a temporary file to be read
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        temp_filename = tmp.name
        tmp.write(video_bytes)

    vid_reader = DecordVideoReader(temp_filename)
    if not frame_numbers:
        frame_numbers = list(range(vid_reader.count_frames()))
    try:
        decoded_frames = vid_reader.load_frames(frame_numbers)
    finally:
        os.remove(temp_filename)

    return decoded_frames
