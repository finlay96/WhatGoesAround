from pathlib import Path
import random

from exploration.find_matches_between_argus_train_andtapvid360 import extract_video_data

if __name__ == "__main__":
    random.seed(42)
    # --- How to use the function ---
    # file_path = '/home/finlay/part_time_PHD_stuff/tracking/WhatGoesAround/data/argus_orig_training_names/clips_filtered.txt'
    # hq_file_path = '/home/finlay/part_time_PHD_stuff/tracking/WhatGoesAround/data/argus_orig_training_names/clips_filtered_high_quality.txt'
    # tapvid_ds_root = Path("/media/finlay/BigDaddyDrive/Outputs/tracker/tapvid360/TAPVid360-10k/dataset")
    file_path = '/users/fgch500/scratch/tracking/WhatGoesAround/data/argus_orig_training_names/clips_filtered.txt'
    hq_file_path = '/users/fgch500/scratch/tracking/WhatGoesAround/data/argus_orig_training_names/clips_filtered_high_quality.txt'
    tapvid_ds_root = Path("/mnt/scratch/projects/cs-dclabs-2019/tapvid360/outputs/TAPVid360-10k/dataset")

    video_info = extract_video_data(file_path)
    hq_video_info = extract_video_data(hq_file_path, skip_lines=1)

    all_vid_ids = {entry['video_id'] for entry in video_info} | {entry['video_id'] for entry in hq_video_info}

    tapvid360_img_names = [f"{item.parent.name}/{item.name}" for sublist in
                           [list(vid_name.glob("*")) for vid_name in tapvid_ds_root.glob("*")] for item in sublist]
    tapvid360_vid_ids = {vid.split("_clip")[0] for vid in tapvid360_img_names}

    common_vid_ids = tapvid360_vid_ids & all_vid_ids

    print("There are ", len(common_vid_ids), "common video IDs between Argus training and TAPVid360.")

    tapvid_names_not_in_argus_train_set = tapvid360_vid_ids - all_vid_ids

    remaining_im_names = [img_name for img_name in tapvid360_img_names if
                          img_name.split("/")[0].split("_clip")[0] in tapvid_names_not_in_argus_train_set]
    print("Which remaining images:", len(remaining_im_names))

    with open(Path(__file__).parent / "tapvid10k_wga_dataset.txt", "w") as f:
        for vid_name in remaining_im_names:
            f.write(f"{vid_name}\n")
