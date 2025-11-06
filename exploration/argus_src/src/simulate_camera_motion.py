import numpy as np
import random
from equilib import equi2pers
import cv2
from regex import B
import torch
import sys

# sys.path.append('.')
from exploration.argus_src.src import generate_mask_batch, pers2equi_batch
import os
from tqdm import tqdm
import math

def rotation_matrix_to_euler(R, z_down=True):
    """
    Convert a rotation matrix to Euler angles (roll, pitch, yaw).
    
    Parameters:
    R : ndarray
        A 3x3 rotation matrix.
        
    Returns:
    roll, pitch, yaw : tuple of float
        The Euler angles in radians.
    """
    assert R.shape == (3, 3), "Input rotation matrix must be 3x3"

    # Check for gimbal lock
    if np.isclose(R[2, 0], -1.0):
        pitch = np.pi / 2
        yaw = 0
        roll = np.arctan2(R[0, 1], R[0, 2])
    elif np.isclose(R[2, 0], 1.0):
        pitch = -np.pi / 2
        yaw = 0
        roll = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    if z_down:
        yaw = -yaw
        pitch = -pitch

    return roll, pitch, yaw

def get_rpy(fps=3, timesteps=100, 
            sigma_yaw=0.15, sigma_pitch=0.15, sigma_roll=0.1,
            A_yaw=1, A_pitch=0.5, A_roll=0.2,
            drift_yaw=15, drift_pitch=3, drift_roll=1): 
    """
    Generate random pitch, yaw, roll angles for camera motion simulation.
    sigma_yaw, sigma_pitch, sigma_roll: standard deviation of noise
    A_pitch, A_yaw: amplitude of pitch and yaw oscillations
    drift_yaw: drift rate for yaw
    """

    # Parameters
    t = np.linspace(0, timesteps / fps, timesteps)
    f = 1.5  # Walking frequency in Hz

    # Oscillatory component
    pitch = A_pitch * np.sin(2 * np.pi * (f * t + np.random.rand())) * np.random.rand()
    yaw = A_yaw * np.sin(2 * np.pi * (f * t + np.random.rand())) * np.random.rand()
    roll = A_roll * np.sin(2 * np.pi * (f * t + np.random.rand())) * np.random.rand()

    # Add drift
    rnd_pitch, rnd_yaw, rnd_roll = np.random.rand(), np.random.rand(), np.random.rand()
    drift_yaw = random.uniform(-drift_yaw, drift_yaw) if rnd_yaw > 1/3 else 0
    drift_pitch = random.uniform(-drift_pitch, drift_pitch) if rnd_pitch > 1/3 else 0
    drift_roll = random.uniform(-drift_roll, drift_roll) if rnd_roll > 1/3 else 0
    yaw += drift_yaw * t
    pitch += drift_pitch * t
    roll += drift_roll * t

    # Add noise
    pitch += np.random.normal(0, sigma_pitch, size=t.shape)
    yaw += np.random.normal(0, sigma_yaw, size=t.shape)
    roll += np.random.normal(0, sigma_roll, size=t.shape)

    # convert into radians
    pitch, yaw, roll = np.radians(pitch), np.radians(yaw), np.radians(roll)

    return pitch, yaw, roll

def simulate_camera_motion(frames, pitch, yaw, roll, save_path, fps=5): 

    mask = generate_mask_batch(fov_x=90,
                            roll=roll, pitch=pitch, yaw=yaw, 
                            height=512, width=1024, device=frames.device,
                            hw_ratio = 2 / 3,) # (T, 1, H, W)
    rots = [{'pitch': p, 'yaw': y, 'roll': r} for p, y, r in zip(pitch, yaw, roll)]
    pers_frames = equi2pers(frames, rots=rots, height=480, width=720, fov_x=90, z_down=True)
    
    frames = frames.to('cuda')
    frames = torch.where(mask == 1, frames, 0.75 * torch.ones_like(frames, device=frames.device))

    # rots = [{'pitch': p, 'yaw': y, 'roll': r} for p, y, r in zip(pitch, yaw, roll)]
    # frames = equi2pers(frames.to('cuda'), rots, height=512, width=1024, fov_x=90, z_down=True)
    frames = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    # save frames
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1024, 512))
    for frame in frames:
        writer.write(frame)

    writer.release()

    save_frames_pers_path = save_path.replace('.mp4', '_pers.mp4')
    pers_frames = (pers_frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    writer = cv2.VideoWriter(save_frames_pers_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (pers_frames.shape[2], pers_frames.shape[1]))
    for frame in pers_frames:
        writer.write(frame)
    writer.release()

if __name__ == '__main__':

    fov_x_min, fov_x_max = 40, 120
    fov_y_min, fov_y_max = 40, 90

    save_motion_root = '/home/rl897/datasets/test-split-101-ICCV/camera_trajectories/real_v2'
    os.makedirs(os.path.join(save_motion_root, 'motion'), exist_ok=True)
    os.makedirs(os.path.join(save_motion_root, 'fov'), exist_ok=True)

    for i in range(101):

        # pitch, yaw, roll = get_rpy(fps=8, timesteps=37)

        # np.savetxt(f'{save_motion_root}/motion/{i:03d}.txt', np.array([pitch, yaw, roll]))

        fov_x = random.uniform(fov_x_min, fov_x_max)
        # make sure hw_ratio is between 1/2 to 2
        fov_y_min_this = 2 * math.degrees(math.atan(math.tan(np.radians(fov_x / 2)) / 2))
        # fov_y_max_this = 2 * math.degrees(math.atan(math.tan(np.radians(fov_x / 2)) * 2))
        fov_y_max_this = fov_x
        # fov_y = random.uniform(max(fov_y_min, fov_y_min_this), min(fov_y_max, fov_y_max_this))

        # set height/width to be either 9/16 or 3/4
        hw_ratio = random.choice([9/16, 3/4])
        height = round(fov_x * hw_ratio)
        width = round(fov_x)
        fov_y = 2 * math.degrees(math.atan(math.tan(np.radians(fov_x / 2)) / hw_ratio))
        
        np.savetxt(f'{save_motion_root}/fov/{i:04d}.txt', np.array([fov_x, fov_y]))

    # save_motion_root = '/home/rl897/datasets/camera_trajectories/real'
    # os.makedirs(os.path.join(save_motion_root, 'motion'), exist_ok=True)
    # os.makedirs(os.path.join(save_motion_root, 'fov'), exist_ok=True)
    # calibration_file_real_world = '/home/rl897/datasets/real-world-tests/calibration_files_25'
    # calibration_files = [os.path.join(calibration_file_real_world, f) for f in os.listdir(calibration_file_real_world)]
    # random.shuffle(calibration_files)

    # split_path = '/home/rl897/data_filtering/split_files_high_quality/test_chosen.txt'

    # for idx, calibration_file_path in enumerate(calibration_files):
    #     poses, intrinsics = torch.load(calibration_file_path)

    #     convention_rotation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    #     convention_inverse = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    #     rolls, pitches, yaws = np.zeros(len(poses)), np.zeros(len(poses)), np.zeros(len(poses))
    #     R1 = poses[0, :3, :3].cpu().numpy()

    #     for i in range(1, len(poses)):
    #         R2 = poses[i, :3, :3].cpu().numpy()
    #         roll, pitch, yaw = rotation_matrix_to_euler(convention_inverse @ R2.T @ R1 @ convention_rotation, z_down=True) # rotation matrix are camera-to-world, cam1 --> cam2 is R2.T @ R1
    #         rolls[i] = -roll
    #         pitches[i] = pitch
    #         yaws[i] = yaw

    #     np.savetxt(f'{save_motion_root}/motion/{idx}.txt', np.array([pitches, yaws, rolls]))

    #     # sample fov_x from 45 to 105
    #     fov_x = random.uniform(45, 105)
    #     # hw_ratio should be 9/16 or 16/9
    #     while True:
    #         hw_ratio = random.choice([9/16, 16/9])
    #         fov_y = 2 * math.degrees(math.atan(math.tan(np.radians(fov_x / 2)) * hw_ratio))
    #         if fov_y >= 45 and fov_y <= 105:
    #             break
    #     np.savetxt(f'{save_motion_root}/fov/{idx}.txt', np.array([fov_x, fov_y]))
    

    # video_info = []

    # with open(split_path, 'r') as f:
    #     for line in f:
    #         video_info.append(line.strip().split('\t'))

    # for (category, video_id, clip_id) in tqdm(video_info):
        
    #     pitch, yaw, roll = get_rpy(fps=fps, timesteps=25)
    #     np.savetxt(f'{save_motion_root}/motion/{category}*{video_id}*{clip_id}.txt', np.array([pitch, yaw, roll]))

    #     fov_x = random.uniform(fov_x_min, fov_x_max)
    #     # make sure hw_ratio is between 9/16 to 16/9
    #     fov_y_min_this = 2 * math.degrees(math.atan(math.tan(np.radians(fov_x / 2)) / 16 * 9))
    #     fov_y_max_this = 2 * math.degrees(math.atan(math.tan(np.radians(fov_x / 2)) / 9 * 16))
    #     fov_y = random.uniform(max(fov_y_min, fov_y_min_this), min(fov_y_max, fov_y_max_this))
        
    #     np.savetxt(f'{save_motion_root}/fov/{category}*{video_id}*{clip_id}.txt', np.array([fov_x, fov_y]))

    # video_path = '/home/rl897/360VideoGeneration/5.mp4'
    # frames = []


    # cap = cv2.VideoCapture(video_path)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # num_frames = 120

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frames.append(frame)
    #     if len(frames) >= num_frames:
    #         break
    
    # cap.release()

    # frames = (torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float() / 255).to('cuda')

    # save_dir = '/home/rl897/360VideoGeneration/cache/simulate_trajectory'
    # os.makedirs(save_dir, exist_ok=True)

    # simulate camera motion
    # 1. sigma (noise only)
    # pitch, yaw, roll = get_rpy(fps=fps, timesteps=num_frames,
    #                            sigma_yaw=0.15, sigma_pitch=0.15, sigma_roll=0.1,
    #                            A_pitch=0, A_yaw=0, drift_yaw=0, drift_pitch=0)

    # simulate_camera_motion(frames, pitch, yaw, roll, os.path.join(save_dir, 'noise.mp4'), fps=fps)
    # print('noise done')

    # # 2. oscillation
    # pitch, yaw, roll = get_rpy(fps=fps, timesteps=num_frames,
    #                            sigma_yaw=0, sigma_pitch=0, sigma_roll=0,
    #                            A_pitch=0.5, A_yaw=1, drift_yaw=0, drift_pitch=0)
    # simulate_camera_motion(frames, pitch, yaw, roll, os.path.join(save_dir, 'oscillation.mp4'), fps=fps)

    # # 3. drift
    # pitch, yaw, roll = get_rpy(fps=fps, timesteps=num_frames,
    #                            sigma_yaw=0, sigma_pitch=0, sigma_roll=0,
    #                            A_pitch=0, A_yaw=0, drift_yaw=12.5, drift_pitch=3)
    # simulate_camera_motion(frames, pitch, yaw, roll, os.path.join(save_dir, 'drift.mp4'), fps=fps)

    # # 4. all, generate 10 videos
    # for i in range(10):
    #     pitch, yaw, roll = get_rpy(fps=fps, timesteps=num_frames,
    #                                sigma_yaw=0.15, sigma_pitch=0.15, sigma_roll=0.1,
    #                                A_pitch=0.5, A_yaw=1, drift_yaw=12.5, drift_pitch=3)
    #     simulate_camera_motion(frames, pitch, yaw, roll, os.path.join(save_dir, f'all_{i}.mp4'), fps=fps)
    

