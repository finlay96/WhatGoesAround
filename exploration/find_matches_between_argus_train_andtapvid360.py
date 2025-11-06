import pprint  # Used for printing the result neatly
from pathlib import Path


def extract_video_data(filename, skip_lines=0):
    """
    Reads a tab-separated file and extracts the data into a list of dictionaries.

    Args:
        filename (str): The path to the text file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a line.
              Returns an empty list if the file is not found or is empty.
    """
    extracted_data = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i < skip_lines:
                    continue
                # 1. Remove leading/trailing whitespace (like the newline character)
                clean_line = line.strip()

                # 2. Skip any empty lines
                if not clean_line:
                    continue

                # 3. Split the line into three parts based on whitespace (tab or spaces)
                parts = clean_line.split()

                # 4. Check if the line has the expected number of parts (3)
                if len(parts) == 3:
                    # 5. Create a dictionary and convert the number to an integer
                    record = {
                        'category': parts[0],
                        'video_id': parts[1],
                        'views_count': int(parts[2])
                    }
                    extracted_data.append(record)
                else:
                    print(f"Skipping malformed line: {line.strip()}")

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except ValueError as e:
        print(f"Error converting number in line: {line.strip()} - {e}")

    return extracted_data


if __name__ == "__main__":
    # --- How to use the function ---
    file_path = '/home/finlay/part_time_PHD_stuff/tracking/WhatGoesAround/data/argus_orig_training_names/clips_filtered.txt'
    video_info = extract_video_data(file_path)

    hq_file_path = '/home/finlay/part_time_PHD_stuff/tracking/WhatGoesAround/data/argus_orig_training_names/clips_filtered_high_quality.txt'
    hq_video_info = extract_video_data(hq_file_path, skip_lines=1)

    all_vid_ids = {entry['video_id'] for entry in video_info} | {entry['video_id'] for entry in hq_video_info}

    tapvid_ds_root = Path("/media/finlay/BigDaddyDrive/Outputs/tracker/tapvid360/TAPVid360-10k/dataset")
    tapvid360_img_names = [f"{item.parent.name}/{item.name}" for sublist in
                           [list(vid_name.glob("*")) for vid_name in tapvid_ds_root.glob("*")] for item in sublist]
    tapvid360_vid_ids = {vid.split("_clip")[0] for vid in tapvid360_img_names}

    common_vid_ids = tapvid360_vid_ids & all_vid_ids

    print("There are ", len(common_vid_ids), "common video IDs between Argus training and TAPVid360.")

    with open(Path(__file__).parent.parent / "data/mini_dataset_names.txt", "r") as f:
        video_names = [line.strip() for line in f.readlines()]
        mini_dataset_vid_idxs = {vid.split("_clip")[0] for vid in video_names}

    common_mini_dataset_vid_ids = tapvid360_vid_ids & mini_dataset_vid_idxs
    print("There are ", len(common_mini_dataset_vid_ids), "common video IDs between the TAPVid360 and the mini dataset used for my Argus testing")

    print("")

    # Print the extracted data to see the result
    # if video_info:
    #     print("Successfully extracted the following data:")
    #     pprint.pprint(video_info)
