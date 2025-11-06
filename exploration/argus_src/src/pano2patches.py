from equilib import equi2pers
import cv2
import numpy as np
import torch
import math
import argparse
import os

def seperate_into_patches(video_path, output_path, patch_size):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    frames = np.stack(frames)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    frames = frames.cuda()

    height, width = patch_size, patch_size
    fov_x = 100
    yaws = torch.Tensor([0, 0, 0, math.pi / 2, math.pi, 3 * math.pi / 2])
    pitches = torch.Tensor([0, math.pi / 2, -math.pi / 2, 0, 0, 0])
    rolls = torch.Tensor([0, 0, 0, 0, 0, 0])

    for roll, pitch, yaw in zip(rolls, pitches, yaws):
        # convert to perspective images
        rots = [{'roll': roll, 'pitch': pitch, 'yaw': yaw}] * len(frames)
        perspective_frames = equi2pers(frames, rots=rots, fov_x=fov_x, height=height, width=width, z_down=True)
        perspective_frames = perspective_frames.permute(0, 2, 3, 1).cpu().numpy() * 255
        perspective_frames = perspective_frames.astype(np.uint8)

        # save into video
        os.makedirs(output_path, exist_ok=True)
        writer = cv2.VideoWriter(os.path.join(output_path, f'fov{fov_x}_roll{roll:.3f}_pitch{pitch:.3f}_yaw{yaw:.3f}.mp4'),
                                 cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for frame in perspective_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', '-v',
                        type=str, default='./cache/360video.mp4')
    parser.add_argument('--output_path', '-o', type=str, default='./cache')
    parser.add_argument('--patch_size', '-p', type=int, default=256)

    args = parser.parse_args()

    seperate_into_patches(args.video_path, args.output_path, args.patch_size)

        