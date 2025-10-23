from pathlib import Path
from tqdm import tqdm

from accelerate import Accelerator
from diffusers import AutoencoderKLTemporalDecoder
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from data.data_utils import get_transform, collate_fn
from data.dataloader import TAPVid360Dataset
from exploration.try_tap_on_360_output_with_withtapvid360 import decode_latents

NUM_FRAMES = 32  # 25
if __name__ == "__main__":
    device = torch.device("cuda:0")
    ds_root = Path("/home/userfs/f/fgch500/storage/shared/track-everywhere/tapvid360")
    assert ds_root.exists()
    argus_feats_dir = Path(
        f"/home/userfs/f/fgch500/storage/outputs/tracking/whatGoesAround/argus_feats_NUM_FRAMES-{NUM_FRAMES}")
    out_dir = Path(
        f"/home/userfs/f/fgch500/storage/outputs/tracking/whatGoesAround/argus_vises_NUM_FRAMES-{NUM_FRAMES}")
    out_dir.mkdir(exist_ok=True)

    accelerator = Accelerator(mixed_precision='no')
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        'stabilityai/stable-video-diffusion-img2vid', subfolder="vae", revision=None)
    vae = accelerator.unwrap_model(vae)
    vae = vae.to(device).eval()

    with open(Path(__file__).parent.parent.parent / "data/mini_dataset_names_100_items.txt", "r") as f:
        video_names = [line.strip() for line in f.readlines()]

    dataset = TAPVid360Dataset(ds_root / f"TAPVid360-10k", transforms.Compose([transforms.ToTensor()]),
                               num_queries=256, num_frames=NUM_FRAMES, specific_videos_list=video_names)
    dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True)

    for i, sample in enumerate(tqdm(dl)):
        argus_feats_path = argus_feats_dir / (sample.seq_name[0].replace("/", "-") + ".pth")
        if not argus_feats_path.exists():
            print(f"Missing argus feats for {sample.seq_name[0]}")
            continue
        argus_feats = torch.load(argus_feats_path)
        argus_latents = argus_feats["latents"].to(device)
        with torch.no_grad():
            pred_eq_frames = decode_latents(vae.eval(), argus_latents, argus_latents.shape[1], 5, False, 4)

        pred_eq_frames_np = ((pred_eq_frames[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).cpu().to(
            torch.float32).numpy().astype(np.uint8)

        # TODO should try and make a vis where we put the original perspective frame ontop of the pred equirectangular image

        vid_out_dir = out_dir / sample.seq_name[0].replace("/", "-")
        vid_out_dir.mkdir(exist_ok=True)
        for f in range(pred_eq_frames_np.shape[0]):
            Image.fromarray(pred_eq_frames_np[f]).save(vid_out_dir / f"{f}.jpg")

        print("")
