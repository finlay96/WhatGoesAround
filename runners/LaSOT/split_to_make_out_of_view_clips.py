from natsort import natsorted
from pathlib import Path
import shutil

import numpy as np


def merge_close_clips(clips, max_gap=20):
    """
    Merges clips that are close to each other.

    Args:
        clips (list): A list of (start, end) tuples.
        max_gap (int): The maximum number of frames *between*
                       two clips to be considered "close".
    """
    if not clips:
        return []

    # Sort clips by start time, just in case
    clips.sort(key=lambda x: x[0])

    merged_clips = []
    # Start with the first clip
    current_start, current_end = clips[0]

    for next_start, next_end in clips[1:]:
        # Calculate the size of the gap between clips
        # (e.g., clip 1 ends at 20, clip 2 starts at 25, gap is 4 frames: 21,22,23,24)
        gap_size = next_start - current_end - 1

        if gap_size <= max_gap:
            # They are close enough, so merge them
            # We just extend the end of the current clip
            current_end = max(current_end, next_end)
        else:
            # They are too far apart. The current merged clip is finished.
            merged_clips.append((current_start, current_end))
            # The next clip becomes the new "current" clip
            current_start, current_end = next_start, next_end

    # After the loop, add the last merged clip
    merged_clips.append((current_start, current_end))

    return merged_clips


def pad_clips(clips, data, pad_size=3):
    padded_clips = []
    max_frame_idx = len(data) - 1
    for start, end in clips:
        padded_start = start - pad_size
        padded_end = end + pad_size
        if padded_start < 0 or padded_end > max_frame_idx:
            continue
        is_valid = True

        # Check prior padding (must be 0s)
        for i in range(padded_start, start):
            if data[i] == 1:  # Fail if it's a 1
                is_valid = False
                break

        if not is_valid:
            continue

        # Check end padding (must be 0s)
        for i in range(end + 1, padded_end + 1):
            if data[i] == 1:  # Fail if it's a 1
                is_valid = False
                break

        if is_valid:
            padded_clips.append((padded_start, padded_end))

    return padded_clips


def find_absence_clips(data, min_duration=1, max_duration=None, min_absent_per_clip=5):
    """
    Finds continuous periods of *absence* (all 1s).
    """
    if max_duration is None:
        max_duration = float('inf')

    clips = []
    in_absence_clip = False
    clip_start = None

    for i, status in enumerate(data):
        if status == 1 and not in_absence_clip:  # CHANGED: Look for 1 to start
            # This is the start of a new absence period
            in_absence_clip = True
            clip_start = i
        elif status == 0 and in_absence_clip:  # CHANGED: Look for 0 to end
            # This is the end of the current absence period
            in_absence_clip = False
            clip_end = i - 1  # The last '1' was at the previous frame
            duration = (clip_end - clip_start) + 1

            if min_duration <= duration <= max_duration:
                clips.append((clip_start, clip_end))

    # Important: Check if the file ended while still in an absence period
    if in_absence_clip:
        clip_end = len(data) - 1
        duration = (clip_end - clip_start) + 1
        if min_duration <= duration <= max_duration:
            clips.append((clip_start, clip_end))

    merged_clips = merge_close_clips(clips, max_gap=10)

    # Now pad the clips so we ensure we have some context
    padded_clips = pad_clips(merged_clips, data, pad_size=5)

    final_clips = []
    for start, end in padded_clips:
        if data[start: end].sum() < min_absent_per_clip:
            continue
        final_clips.append((start, end))

    return final_clips


def convert_indices_to_original(clips, factor, max_frame_idx):
    """
    Converts clip indices from the downsampled space back to the
    original frame space.

    Args:
        clips (list): List of (start, end) tuples from downsampled data.
        factor (int): The downsampling factor used (e.g., 5).
        max_frame_idx (int): The last valid index of the *original* data.

    Returns:
        list: List of (start, end) tuples in original frame indices.
    """
    original_clips = []
    for start, end in clips:
        # Convert start index
        original_start = start * factor

        # Convert end index
        # (end + 1) * factor - 1 gives the last frame in the last block
        original_end = (end + 1) * factor - 1

        # Clamp the end to the actual max frame index
        original_end = min(original_end, max_frame_idx)

        original_clips.append((original_start, original_end))

    return original_clips


def downsample_data(data, factor=5):
    """
    Downsamples the data by aggregating in blocks.

    If any value in a block of 'factor' frames is 1,
    the entire downsampled block becomes 1.

    Args:
        data (np.array): The original data (1s and 0s).
        factor (int): The downsampling factor (e.g., 5).

    Returns:
        np.array: The new, smaller downsampled array.
    """
    # Calculate the number of blocks, padding if necessary
    num_blocks = int(np.ceil(len(data) / factor))

    # Create the new downsampled array
    downsampled = np.zeros(num_blocks, dtype=int)

    for i in range(num_blocks):
        start = i * factor
        end = start + factor
        # Get the chunk from the original data
        chunk = data[start:end]

        # If max is 1 (meaning any(chunk == 1)), set to 1
        downsampled[i] = np.max(chunk)

    return downsampled


if __name__ == '__main__':
    ds_root = Path("/media/finlay/BigDaddyDrive/Datasets/tracking/object-tracking/LaSOT/data")
    output_dir = Path("/media/finlay/BigDaddyDrive/Datasets/tracking/object-tracking/LaSOT/custom_out_of_frame_clips")
    output_dir.mkdir(exist_ok=True)
    downsample_factor = 5
    for data_type in ds_root.glob("*"):
        for example in data_type.glob("*"):
            out_of_view_file = example / "out_of_view.txt"
            if not out_of_view_file.exists():
                print(f"Missing out_of_view.txt for {example}")
                continue
            with open(out_of_view_file, 'r') as f:
                out_of_view_lines = f.readlines()
                out_of_view_vals = np.array([int(val) for val in out_of_view_lines[0].split(",")])

            gt_data_file = example / "groundtruth.txt"
            with open(gt_data_file, 'r') as f:
                gt_bbox_lines = f.readlines()
                gt_bboxes_xywh = np.array([list(map(int, line.strip().split(','))) for line in gt_bbox_lines])

            downsampled_of_view_vals = downsample_data(out_of_view_vals, downsample_factor)
            downsampled_absence_clips = find_absence_clips(downsampled_of_view_vals)
            if not len(downsampled_absence_clips):
                continue

            absence_clip_idxs = convert_indices_to_original(
                downsampled_absence_clips,
                downsample_factor,
                len(out_of_view_vals) - 1
            )

            img_dir = example / "img"
            if not img_dir.exists():
                print(f"Missing img folder for {example.name}")
                continue

            # Get a sorted list of all frame paths. This is CRITICAL.
            # img_files[0] will correspond to out_of_view_vals[0]
            img_files = natsorted(list(img_dir.glob("*.jpg")))

            assert len(img_files) == len(out_of_view_vals)

            # Use enumerate to give each clip a unique ID (0, 1, 2...)
            for clip_index, (start_frame, end_frame) in enumerate(absence_clip_idxs):

                # Create a specific output folder for this clip
                clip_output_folder = output_dir / example.name / f"clip_{clip_index:03d}"
                clip_output_folder.mkdir(parents=True, exist_ok=True)

                img_out = clip_output_folder / "img"
                img_out.mkdir(exist_ok=True)

                bboxes = []
                # Loop through the frame indices of this single clip
                for frame_index in range(start_frame, end_frame + 1, downsample_factor):
                    # Get the source path of the frame
                    source_frame_path = img_files[frame_index]

                    # Define the destination path
                    dest_frame_path = img_out / source_frame_path.name

                    # Copy the file
                    shutil.copy(str(source_frame_path), str(dest_frame_path))
                    bboxes.append(gt_bboxes_xywh[frame_index])

                out_of_view_output_vals = out_of_view_vals[start_frame: end_frame][::downsample_factor]
                assert len(bboxes) == len(out_of_view_output_vals)

                np.save(clip_output_folder / "bboxes_xywh", np.array(bboxes))
                np.save(clip_output_folder / "out_of_view", out_of_view_output_vals)

            print(f"Finished copying clips for {example.name}")
