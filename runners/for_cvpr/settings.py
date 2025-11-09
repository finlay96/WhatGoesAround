import argparse
import dataclasses
from pathlib import Path
import socket

import torch


@dataclasses.dataclass
class VikingPaths:
    unet_path = "/mnt/scratch/projects/cs-dclabs-2019/WhatGoesAround/pretrained_models/argus/checkpoints"
    out_root = Path("/mnt/scratch/projects/cs-dclabs-2019/WhatGoesAround-FOR-CVPR")
    sam_checkpoint = Path(
        "/home/userfs/f/fgch500/storage/pretrained_models/segmentation") / "sam2" / "sam2_hiera_large.pt"
    tapvid360_data_root = Path("/mnt/scratch/projects/cs-dclabs-2019/tapvid360/outputs")


@dataclasses.dataclass
class CSGPUPaths:
    unet_path = "/home/userfs/f/fgch500/storage/pretrained_models/video_generation/argus"
    out_root = Path("/home/userfs/f/fgch500/storage/shared/WhatGoesAround-FOR-CVPR")
    sam_checkpoint = Path(
        "/shared/storage/cs/staffstore/fgch500/pretrained_models/segmentation") / "sam2" / "sam2_hiera_large.pt"
    tapvid360_data_root = Path("/home/userfs/f/fgch500/storage/shared/TAPVid360/data")


@dataclasses.dataclass
class LocalPaths:
    out_root = Path("/media/finlay/BigDaddyDrive/Outputs/tracker/WhatGoesAround-FOR-CVPR")
    sam_checkpoint = Path("/home/finlay/Shared/pretrained_models/segmentation") / "sam2" / "sam2_hiera_large.pt"
    tapvid360_data_root = Path("/media/finlay/BigDaddyDrive/Outputs/tracker/tapvid360")


@dataclasses.dataclass
class Settings:
    ds_name = "tapvid360-10k"
    specific_video_names = None #["AfXjCJcexSI_clip_0010/0"]  # "nqR-umO4bvY_clip_0006/0"

    # ds_name = "lasot_oov"
    # # specific_video_names = "mouse-15/clip_000"
    # # specific_video_names = "kite-5/clip_000"
    # specific_video_names = None
    # ds_root = Path("/home/userfs/f/fgch500/storage/datasets/tracking/object_tracking/LaSOT/custom_out_of_frame_clips")
    # out_root = Path("/mnt/scratch/projects/cs-dclabs-2019/WhatGoesAround")

    csgpu_paths = CSGPUPaths()
    viking_paths = VikingPaths()
    local_paths = LocalPaths()
    paths = None # use set_host_in_settings to set

    # ds_root = Path("/media/finlay/BigDaddyDrive/Datasets/tracking/object-tracking/LaSOT/custom_out_of_frame_clips")

    force_get_latents = True
    eq_height = 512  # TODO watchout this is currently hardcoded
    weight_dtype = torch.float32
    decode_chunk_size = 10  # Should be able to be bigger for larger gpu's
    offload_to_cpu = True


def get_args():
    parser = argparse.ArgumentParser(description="Runners for argus cotracker")
    parser.add_argument("-gt", "--use_gt_rot", action='store_true', help="Use ground truth rotation")
    parser.add_argument("-d", "--debugs", action='store_true', help="Output debugs")
    parser.add_argument(
        '--split_file',  # The name of the flag
        type=str,  # The type of value to expect
        default=None,  # The default value if the flag is not provided
        help="Optional: Path to the input file."  # Help message
    )

    return parser.parse_args()


def set_host_in_settings(settings):
    hostname = socket.gethostname()
    if "viking" in hostname:
        settings.paths = settings.viking_paths
    elif "csgpu" in hostname:
        settings.paths = settings.csgpu_paths
    else:
        settings.paths = settings.local_paths

    return settings