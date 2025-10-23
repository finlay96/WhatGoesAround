from pathlib import Path
from tqdm import tqdm

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from argus_cotracker.components.argus_cotracker_approach import ArgusCoTracker
from argus_cotracker.components.pipelines.argus.utils import get_models_custom
from conversions.mapper import Mappers
from data.data_utils import get_transform, collate_fn
from data.dataloader import TAPVid360Dataset
from exploration.argus_src.inference import SVDArguments

NUM_FRAMES = 32 #25
if __name__ == "__main__":
    device = torch.device("cuda:0")
    unet_path = "/home/userfs/f/fgch500/storage/pretrained_models/video_generation/argus"
    ds_root = Path("/home/userfs/f/fgch500/storage/shared/track-everywhere/tapvid360")
    assert ds_root.exists()
    out_root = Path(f"/home/userfs/f/fgch500/storage/outputs/tracking/whatGoesAround/argus_feats_NUM_FRAMES-{NUM_FRAMES}")
    out_root.mkdir(parents=True, exist_ok=True)
    args = SVDArguments(unet_path=unet_path, num_inference_steps=50)
    # TODO TRYING TO HANDLE MORE THAN 25 frames
    args.blend_frames = 5
    args.num_frames_batch = 25
    args.num_frames = NUM_FRAMES
    weight_dtype = torch.float32
    accelerator = Accelerator(mixed_precision='no')

    argus_pipeline = get_models_custom(args, accelerator, weight_dtype)
    model = ArgusCoTracker(accelerator, weight_dtype, args, argus_pipeline, None, device=device)

    # TODO get dataloader
    with open(Path(__file__).parent.parent.parent / "data/mini_dataset_names_100_items.txt", "r") as f:
        video_names = [line.strip() for line in f.readlines()]

    dataset = TAPVid360Dataset(ds_root / f"TAPVid360-10k", transforms.Compose([transforms.ToTensor()]),
                               num_queries=256, num_frames=NUM_FRAMES, specific_videos_list=video_names)
    dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True)

    for i, sample in enumerate(tqdm(dl)):
        sample.to_device(device)
        mapper = Mappers(320, 240, 512, fov_x=75.18)  # TODO should use actual data for this
        # TODO currently not batchable
        eq_frames, eq_mask = mapper.image.perspective_image_to_equirectangular(
            sample.video[0].permute(0, 2, 3, 1) * 255,
            sample.rotations[0], to_uint=False)
        video = (sample.video[0] * 255) / 127.5 - 1
        video = video.to(device)

        eq_frames = (eq_frames.permute(0, 3, 1, 2).float()) / 127.5 - 1
        eq_frames = eq_frames.to(device)
        eq_mask = eq_mask.to(device).unsqueeze(1)

        with torch.no_grad():
            latents, last_layer_unet_outs = model.forward_argus(sample.video[0], eq_frames, eq_mask)

        latents= latents.cpu()
        last_layer_unet_outs = {k1: {k2: v2.to('cpu') for k2, v2 in v1.items()}
                                     for k1, v1 in last_layer_unet_outs.items()}
        # out_data = {
        #     "latents": latents,
        #     "last_layer_unet_outs": last_layer_unet_outs,
        # }
        out_data = {
            "latents": latents,
            # "last_layer_unet_outs": last_layer_unet_outs,
        }

        torch.save(out_data, out_root / f"{sample.seq_name[0].replace('/', '-')}.pth")
