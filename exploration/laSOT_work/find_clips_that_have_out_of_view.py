import numpy as np

from pathlib import Path

if __name__ == '__main__':
    ds_root = Path("/media/finlay/BigDaddyDrive/Datasets/tracking/object-tracking/LaSOT/data")
    for data_type in ds_root.glob("*"):
        for example in data_type.glob("*"):
            out_of_view_file = example / "out_of_view.txt"
            if not out_of_view_file.exists():
                print(f"Missing out_of_view.txt for {example}")
                continue
            with open(out_of_view_file, 'r') as f:
                out_of_view_lines = f.readlines()
                out_of_view_vals = np.array([int(val) for val in out_of_view_lines[0].split(",")])
            if out_of_view_vals.any():
                print(f"Clip with out-of-view frames: {example.relative_to(ds_root)} with num out of frame being {out_of_view_vals.sum()}")
