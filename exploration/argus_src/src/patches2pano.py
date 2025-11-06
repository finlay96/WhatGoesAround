import re
import sys
# sys.path.append('./')
from exploration.argus_src.src import pers2equi_batch
import cv2
import numpy as np
import torch
import math
import os
import argparse

def patch_to_equi(video_root_path, output_path, batch_size=25, height=1280, width=2560, device='cuda'):

    VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']

    video_paths = [os.path.join(video_root_path, video_name) for video_name in os.listdir(video_root_path) if any(video_name.endswith(ext) for ext in VIDEO_EXTENSIONS)]

    canvas, mask_canvas = None, None
    fps = None

    for video_path in video_paths:
        fov_x, roll, pitch, yaw = video_path.split('/')[-1][:-4].split('_')[:4]
        fov_x, roll, pitch, yaw = float(fov_x.split('fov')[-1]), float(roll.split('roll')[-1]), float(pitch.split('pitch')[-1]), float(yaw.split('yaw')[-1])
        frames = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1
        frames = frames.to(device)

        rolls, pitches, yaws = [roll] * len(frames), [pitch] * len(frames), [yaw] * len(frames)

        pers_frames_stack, mask_stack = [], []
        for i in range(0, len(frames), batch_size):
            end = min(i+batch_size, len(frames))
            pers_frames, mask = pers2equi_batch(frames[i:end], fov_x=fov_x, 
                                                roll=rolls[i:end], 
                                                pitch=pitches[i:end],
                                                yaw=yaws[i:end],
                                                height=height, width=width,
                                                device=device, return_mask=True,
                                                shrink_mask_pixels=2)
            pers_frames_stack.append(pers_frames)
            mask_stack.append(mask)
        pers_frames = torch.cat(pers_frames_stack, dim=0)
        mask = torch.cat(mask_stack, dim=0)
        
        if canvas is None:
            canvas = (pers_frames * 0.5 + 0.5).cpu() # (T, 3, H, W)
            mask_canvas = mask.cpu() # (T, 1, H, W)
        else:
            canvas += (pers_frames * 0.5 + 0.5).cpu()
            mask_canvas += mask.cpu()

        print(canvas.shape, mask_canvas.shape)
    
    # canvas / mask_canvas, first set 0 entry to 1 to avoid division by zero
    mask_canvas = torch.clamp(mask_canvas, min=1)
    canvas = torch.divide(canvas, mask_canvas).clip(0, 1)
    canvas = (canvas * 255).byte().cpu().numpy().transpose(0, 2, 3, 1)

    print('Saving video to', output_path)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for i, frame in enumerate(canvas):
        writer.write(frame)
    writer.release()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_root_path', '-v', type=str, required=True)
    parser.add_argument('--output_path', '-o', type=str, required=True)
    parser.add_argument('--batch_size', '-b', type=int, default=30)
    parser.add_argument('--height', type=int, default=1280)
    parser.add_argument('--width', type=int, default=2560)
    args = parser.parse_args()

    patch_to_equi(args.video_root_path, args.output_path, height=args.height, width=args.width)

