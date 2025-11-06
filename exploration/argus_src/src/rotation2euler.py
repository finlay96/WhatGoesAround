import numpy as np
import json
from equilib import equi2pers

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

def relative_euler_angles(R1, R2, z_down=False):
    # Compute relative rotation matrix
    R_rel = np.dot(R2, R1.T)
    
    # Convert relative rotation matrix to yaw, pitch, roll (ZYX convention)
    return rotation_matrix_to_euler(R_rel, z_down)

if __name__ == "__main__":

    z_down = True

    # read from file
    camera_param_file = '/home/rl897/mast3r/mast3r_output/cameras.json'
    with open(camera_param_file, 'r') as f:
        camera_params = json.load(f)['poses']
    camera_params = np.array(camera_params)
    # print(camera_params.shape)

    R1, R2 = camera_params[0][:3, :3], camera_params[1][:3, :3]

    roll, pitch, yaw = relative_euler_angles(R1, R2, z_down=z_down)