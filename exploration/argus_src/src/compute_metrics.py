import cv2
import numpy as np
import os
import math
from piq import ssim, psnr
import torch
from equilib import equi2pers
import argparse
from PIL import Image
import sys
sys.path.append('.')
from src import LPIPS, video_psnr, calculate_fvd, AverageMeter
from tqdm import tqdm
from torch.utils.data import DataLoader
import random

def eval_videos_fixed_view(video1: torch.Tensor, video2: torch.Tensor,
                            roll=0, pitch=0, yaw=0, fov=90): 
    '''
    two videos are in equirectangular format, T*C*H*W
    '''
    assert video1.shape == video2.shape

    rots = [{'roll': roll, 'pitch': pitch, 'yaw': yaw}] * video1.shape[0]
    video1_pers = equi2pers(video1, fov_x=fov, rots=rots)
    video2_pers = equi2pers(video2, fov_x=fov, rots=rots)

    ssim_val = ssim(video1_pers, video2_pers, data_range=1.0, reduction='mean')
    psnr_val = psnr(video1_pers, video2_pers, data_range=1.0, reduction='mean')

    return ssim_val, psnr_val

def eval_videos(video1: torch.Tensor, video2: torch.Tensor, fov=90):
    '''
    two videos are in equirectangular format, T*C*H*W
    yaw in {0, 90, 180, 270}, pitch in {-60, 0, 60}, roll in {0}
    '''

    yaws = torch.Tensor([0, 0, 0, math.pi / 2, math.pi, math.pi, math.pi, 3 * math.pi / 2])
    pitches = torch.Tensor([-math.pi / 3, 0, math.pi / 3, 0, -math.pi / 3, 0, math.pi / 3, 0])
    rolls = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0])

    ssim_vals = []
    psnr_vals = []

    for yaw, pitch, roll in zip(yaws, pitches, rolls):
        ssim_val, psnr_val = eval_videos_fixed_view(video1, video2, roll, pitch, yaw, fov)
        ssim_vals.append(ssim_val)
        psnr_vals.append(psnr_val)

    return ssim_vals, psnr_vals

def read_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    video = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video.append(frame)
    video = np.stack(video, axis=0)
    video = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0
    return video

class PairedVideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_gen, root_gt, root_mask=None, height=None, width=None):
        self.root_gen = root_gen
        self.root_gt = root_gt
        self.root_mask = root_mask
        self.height = height
        self.width = width

        self.gen_video_paths, self.gt_video_paths, self.mask_paths = [], [], []

        gt_video_names = [x for x in os.listdir(self.root_gt) if x.endswith('.mp4')]

        for video_name in gt_video_names:
            gt_video_path = os.path.join(self.root_gt, video_name)
            self.gt_video_paths.append(gt_video_path)

            gen_video_path = os.path.join(self.root_gen, video_name)
            assert os.path.exists(gen_video_path), "gen_video_path does not exist: {}".format(gen_video_path)
            self.gen_video_paths.append(gen_video_path)

            if self.root_mask is not None:
                mask_path = os.path.join(self.root_mask, video_name[:-4], 'mask_stack.png')
                assert os.path.exists(mask_path), "mask_path does not exist: {}".format(mask_path)
                self.mask_paths.append(mask_path)

        print('Number of videos', len(self.gen_video_paths))

    def __len__(self):
        return len(self.gen_video_paths)

    def __getitem__(self, idx):

        gen_video_path = self.gen_video_paths[idx]
        gt_video_path = self.gt_video_paths[idx]

        gen_video = read_video(gen_video_path) # (T, C, H, W)
        gt_video = read_video(gt_video_path) # (T, C, H, W)

        # if gen_video.shape[2] / gen_video.shape[3] != gt_video.shape[2] / gt_video.shape[3]:
        #     print('Aspect ratio mismatch, video_path: {}, \
        #           gen_video.shape: {}, gt_video.shape: {}'.format(gen_video_path, gen_video.shape, gt_video.shape))

        if self.root_mask is not None:
            mask = Image.open(self.mask_paths[idx]).convert('L')
            mask = torch.tensor(np.array(mask)).unsqueeze(0).unsqueeze(0).float() / 255.0 # (1, H, W)

        if self.height is not None and self.width is not None:
            gen_video = torch.nn.functional.interpolate(gen_video, (self.height, self.width), mode='bilinear')
            gt_video = torch.nn.functional.interpolate(gt_video, (self.height, self.width), mode='bilinear')
            if self.root_mask is not None:
                mask = torch.nn.functional.interpolate(mask, (self.height, self.width), mode='nearest')
        elif self.height is not None: # resize all videos' height to the same value, width is resized accordingly
            new_width = int(self.height / gt_video.shape[2] * gt_video.shape[3])
            gen_video = torch.nn.functional.interpolate(gen_video, (self.height, new_width), mode='bilinear')
            gt_video = torch.nn.functional.interpolate(gt_video, (self.height, new_width), mode='bilinear')
            if self.root_mask is not None:
                mask = torch.nn.functional.interpolate(mask, (self.height, new_width), mode='nearest')
        elif self.width is not None: # resize all videos' width to the same value, height is resized accordingly
            new_height = int(self.width / gt_video.shape[3] * gt_video.shape[2])
            gen_video = torch.nn.functional.interpolate(gen_video, (new_height, self.width), mode='bilinear')
            gt_video = torch.nn.functional.interpolate(gt_video, (new_height, self.width), mode='bilinear')
        else:
            height, width = gt_video.shape[2], gt_video.shape[3]
            gen_video = torch.nn.functional.interpolate(gen_video, (height, width), mode='bilinear')
            if self.root_mask is not None:
                mask = torch.nn.functional.interpolate(mask, (height, width), mode='nearest')

        return (gen_video, gt_video, mask) if self.root_mask is not None else (gen_video, gt_video)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_gen', type=str, required=True)
    parser.add_argument('--root_gt', type=str, default='/home/rl897/datasets/test-split-101-ICCV/clips')
    parser.add_argument('--root_mask', type=str, default=None)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--rotate', action='store_true')

    args = parser.parse_args()
    device = torch.device('cuda')
    video_dataset = PairedVideoDataset(args.root_gen, args.root_gt, args.root_mask,
                                        height=args.height, width=args.width)

    '''calculate PSNR and LPIPS'''

    if args.root_mask is not None:
        lpips = LPIPS(net='vgg').to(device)

        psnr_meter, lpips_meter = AverageMeter(), AverageMeter()

        for (gen_video, gt_video, mask) in tqdm(video_dataset):
            gen_video, gt_video, mask = gen_video.to(device), gt_video.to(device), mask.to(device)

            psnr_vals = video_psnr(gen_video, gt_video, mask).mean().item()
            lpips_val = lpips(gen_video, gt_video, mask).mean().item()

            psnr_meter.update(psnr_vals)
            lpips_meter.update(lpips_val)

        print('PSNR:', psnr_meter.avg)
        print('LPIPS:', lpips_meter.avg)

    '''calculate FVD'''
    # video_loader = DataLoader(video_dataset, batch_size=len(video_dataset), shuffle=False)
    # batch = next(iter(video_loader))
    # gen_videos, gt_videos = batch[0], batch[1]
    # if args.rotate:
    #     gen_videos = torch.roll(gen_videos, gen_videos.shape[-1] // 2, dims=-1)
    #     gt_videos = torch.roll(gt_videos, gt_videos.shape[-1] // 2, dims=-1)

    # fvd = calculate_fvd(gen_videos, gt_videos, device, nearest_k=5)

    # print(fvd)
    


