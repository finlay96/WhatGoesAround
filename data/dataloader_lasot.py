from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from natsort import natsorted
from torch.utils.data import Dataset
from torchvision import transforms

from data.data_utils import ResizeTensor, scale_bboxes
from runners.bbox_utils import xywh_to_xyxy


@dataclass(eq=False)
class DataSampleLaSOT:
    """
    Dataclass for storing video tracks data.
    """

    video: torch.Tensor  # B, S, C, H, W
    bboxes_xywh: torch.Tensor  # B, S, 4
    bboxes_xyxy: torch.Tensor  # B, S, 4
    seq_name: list[str] | str
    visibility: torch.Tensor  # B, S, N

    def to_device(self, device: torch.device):
        """
        Moves all torch.Tensor attributes to the specified device.
        """
        # Iterate over all fields of the dataclass
        for field in fields(self):  # fields(self) gives access to the dataclass fields
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):  # If the field is a tensor
                setattr(self, field.name, value.to(device))  # Move tensor to device


def collate_fn_laSOT(data: list[DataSampleLaSOT]):
    return DataSampleLaSOT(
        video=torch.stack([d.video for d in data]),
        bboxes_xywh=torch.stack([d.bboxes_xywh for d in data]),
        bboxes_xyxy=torch.stack([d.bboxes_xyxy for d in data]),
        seq_name=[d.seq_name for d in data],
        visibility=torch.stack([d.visibility for d in data])
    )


class LaSOTDataset(Dataset):
    def __init__(self, ds_root, transform=None, debug_max_items=-1, num_frames=-1, specific_videos_list=None):
        super().__init__()
        self.ds_root = Path(ds_root)
        self.transform = transform
        self.vid_names = [f"{item.parent.name}/{item.name}" for sublist in
                          [list(vid_name.glob("*")) for vid_name in self.ds_root.glob("*")] for item in sublist]
        if specific_videos_list is not None:
            self.vid_names = [vid for vid in self.vid_names if vid in specific_videos_list]
        if debug_max_items > 0:
            if debug_max_items == 0:
                raise ValueError("debug_max_items must be greater than 1")
            self.vid_names = self.vid_names[:debug_max_items]
        self.max_num_frames = num_frames

    def _get_the_data(self, spherical_img_name):
        t_imgs = []
        persp_img_shapes = []
        img_names = natsorted(list((self.ds_root / spherical_img_name / "img").glob("*")))
        for i, img_name in enumerate(img_names):
            persp_img = Image.open(img_name)
            persp_img_shapes.append(persp_img.size)
            t_imgs.append(self.transform(persp_img))
            if self.max_num_frames > 0 and i + 1 >= self.max_num_frames:
                break
        t_imgs = torch.stack(t_imgs)

        bboxes_xywh = np.load(self.ds_root / spherical_img_name / "bboxes_xywh.npy")
        _, _, h, w = t_imgs.shape

        scaled_bboxes_xywh = scale_bboxes(persp_img_shapes, bboxes_xywh, h, w)

        visibility = torch.from_numpy(~((np.load(self.ds_root / spherical_img_name / "out_of_view.npy")).astype(bool)))

        return DataSampleLaSOT(video=t_imgs, bboxes_xywh=scaled_bboxes_xywh,
                               bboxes_xyxy=xywh_to_xyxy(scaled_bboxes_xywh), seq_name=spherical_img_name,
                               visibility=visibility)

    def __len__(self):
        return len(self.vid_names)

    def __getitem__(self, idx):
        return self._get_the_data(self.vid_names[idx])


if __name__ == "__main__":
    dataset = LaSOTDataset(
        "/media/finlay/BigDaddyDrive/Datasets/tracking/object-tracking/LaSOT/custom_out_of_frame_clips",
        transform=transforms.Compose([transforms.ToTensor(), ResizeTensor(calibration_img_size=512)]),
        debug_max_items=2)
    for i in range(len(dataset)):
        sample = dataset[i]
        print(sample)
