import torch


def rot_x(alpha: torch.Tensor, degrees=False) -> torch.Tensor:
    """Computes batched 3x3 rotation matrices around the X-axis. For our purpose this is a roll rotation"""
    if degrees:
        alpha = torch.deg2rad(alpha)  # Convert to radians if needed

    cos_a, sin_a = torch.cos(alpha), torch.sin(alpha)

    R_x = torch.stack([
        torch.ones_like(alpha), torch.zeros_like(alpha), torch.zeros_like(alpha),
        torch.zeros_like(alpha), cos_a, -sin_a,
        torch.zeros_like(alpha), sin_a, cos_a
    ], dim=-1).reshape(-1, 3, 3)  # Shape: (B, 3, 3)

    return R_x


def rot_y(beta: torch.Tensor, degrees=False) -> torch.Tensor:
    """Computes batched 3x3 rotation matrices around the Y-axis. For our purpose this is a up/down rotation"""
    if degrees:
        beta = torch.deg2rad(beta)

    cos_b, sin_b = torch.cos(beta), torch.sin(beta)

    R_y = torch.stack([
        cos_b, torch.zeros_like(beta), sin_b,
        torch.zeros_like(beta), torch.ones_like(beta), torch.zeros_like(beta),
        -sin_b, torch.zeros_like(beta), cos_b
    ], dim=-1).reshape(-1, 3, 3)  # Shape: (B, 3, 3)

    return R_y


def rot_z(gamma: torch.Tensor, degrees=False) -> torch.Tensor:
    """Computes batched 3x3 rotation matrices around the Z-axis. For our purpose this is a left/right rotation"""
    if degrees:
        gamma = torch.deg2rad(gamma)

    cos_g, sin_g = torch.cos(gamma), torch.sin(gamma)

    R_z = torch.stack([
        cos_g, -sin_g, torch.zeros_like(gamma),
        sin_g, cos_g, torch.zeros_like(gamma),
        torch.zeros_like(gamma), torch.zeros_like(gamma), torch.ones_like(gamma)
    ], dim=-1).reshape(-1, 3, 3)  # Shape: (B, 3, 3)

    return R_z


def rot(alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, degrees=False) -> torch.Tensor:
    """Computes batched 3x3 rotation matrices for given roll (α), pitch (β), and yaw (γ)."""
    assert alpha.shape == beta.shape == gamma.shape, "Input angles must have the same shape"

    R_x = rot_x(alpha, degrees)  # (B, 3, 3)
    R_y = rot_y(beta, degrees)  # (B, 3, 3)
    R_z = rot_z(gamma, degrees)  # (B, 3, 3)

    # Batched matrix multiplication -> (B, 3, 3), defaults to bfloat16 but we want float32 later on
    return (R_x @ R_y @ R_z).to(torch.float32)


def update_rotation_matrix(R: torch.Tensor, d_alpha: torch.Tensor, d_beta: torch.Tensor, d_gamma: torch.Tensor,
                           degrees=False) -> torch.Tensor:
    """
    Updates a batch of existing rotation matrices by applying small changes in roll, pitch, and yaw.

    Args:
        R (torch.Tensor): Shape (B, 3, 3), existing rotation matrices.
        d_alpha (torch.Tensor): Shape (B,), incremental pitch (Y-axis rotation).
        d_beta (torch.Tensor): Shape (B,), incremental roll (X-axis rotation).
        d_gamma (torch.Tensor): Shape (B,), incremental yaw (Z-axis rotation).
        degrees (bool): If True, inputs are in degrees.

    Returns:
        torch.Tensor: Updated rotation matrices of shape (B, 3, 3).
    """
    if not isinstance(d_alpha, torch.Tensor):
        d_alpha = torch.tensor(d_alpha)
    if not isinstance(d_beta, torch.Tensor):
        d_beta = torch.tensor(d_beta)
    if not isinstance(d_gamma, torch.Tensor):
        d_gamma = torch.tensor(d_gamma)

    d_alpha = d_alpha.to(R.device)
    d_beta = d_beta.to(R.device)
    d_gamma = d_gamma.to(R.device)

    if degrees:
        d_alpha = d_alpha * torch.pi / 180
        d_beta = d_beta * torch.pi / 180
        d_gamma = d_gamma * torch.pi / 180

    # Compute batched small incremental rotation matrices
    R_x = rot_x(d_alpha)  # Shape (B, 3, 3)
    R_y = rot_y(d_beta)  # Shape (B, 3, 3)
    R_z = rot_z(d_gamma)  # Shape (B, 3, 3)

    # Compute batched delta rotation matrices: R_delta = R_x @ R_y @ R_z
    R_delta = R_x @ R_y @ R_z  # Shape (B, 3, 3)

    # Apply incremental rotation
    R_new = R @ R_delta.to(R.device)  # Shape (B, 3, 3)

    return R_new


# TODO NEEDS TESTING
def rot_to_euler_xyz(R: torch.Tensor, degrees: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts batched 3x3 rotation matrices (X-Y-Z order) to Euler angles (roll, pitch, yaw).

    Args:
        R: Rotation matrices of shape (..., 3, 3).
        degrees: Whether to return angles in degrees.

    Returns:
        A tuple of (alpha, beta, gamma) angles in radians (or degrees)
        with the same batch shape as R (...).
    """
    # R[..., 0, 2] = sin(beta)
    # We must clamp the input to asin to avoid NaNs from floating point inaccuracies
    sin_beta = torch.clamp(R[..., 0, 2], -1.0, 1.0)
    beta = torch.asin(sin_beta)

    # Check for gimbal lock
    # This occurs when beta is +/- 90 degrees (cos(beta) = 0)
    epsilon = 1e-6  # A small tolerance for floating point comparison
    is_gimbal_lock = sin_beta.abs() > (1.0 - epsilon)

    # --- Normal Case (no gimbal lock) ---
    # alpha = atan2(-R[..., 1, 2], R[..., 2, 2])
    # gamma = atan2(-R[..., 0, 1], R[..., 0, 0])
    alpha_normal = torch.atan2(-R[..., 1, 2], R[..., 2, 2])
    gamma_normal = torch.atan2(-R[..., 0, 1], R[..., 0, 0])

    # --- Gimbal Lock Case ---
    # We set alpha (roll) to 0 by convention
    alpha_gimbal = torch.zeros_like(beta)

    # We can solve for gamma using other elements
    # R[..., 1, 0] = sin(alpha)*sin(beta)*cos(gamma) + cos(alpha)*sin(gamma)
    # R[..., 1, 1] = -sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma)
    # If alpha = 0:
    # R[..., 1, 0] = sin(gamma)
    # R[..., 1, 1] = cos(gamma)
    # So, gamma = atan2(R[..., 1, 0], R[..., 1, 1])
    # This formula holds for both +90 and -90 deg gimbal lock.
    gamma_gimbal = torch.atan2(R[..., 1, 0], R[..., 1, 1])

    # Combine using torch.where
    alpha = torch.where(is_gimbal_lock, alpha_gimbal, alpha_normal)
    gamma = torch.where(is_gimbal_lock, gamma_gimbal, gamma_normal)

    if degrees:
        alpha = torch.rad2deg(alpha)
        beta = torch.rad2deg(beta)
        gamma = torch.rad2deg(gamma)

    return alpha, beta, gamma