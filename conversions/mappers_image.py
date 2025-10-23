import torch

from conversions.convert import Conversions
from conversions.mapper_config import MapperConfig


class MappersImage:
    def __init__(self, config: MapperConfig):
        self.cfg = config
        self.conversions = Conversions(config)
        # 1. Create spherical angle grid (theta, phi) in new ego frame
        u = torch.linspace(0, self.cfg.equirectangular_width - 1, self.cfg.equirectangular_width)
        v = torch.linspace(0, self.cfg.equirectangular_height - 1, self.cfg.equirectangular_height)
        grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')
        self.cr_grid = torch.stack([grid_u, grid_v], dim=-1)  # (H_eq, W_eq, 2)

        i_coords = torch.arange(0, self.cfg.crop_width)
        j_coords = torch.arange(0, self.cfg.crop_height)
        ii, jj = torch.meshgrid(i_coords, j_coords, indexing='xy')
        self.ij_grid = torch.stack([ii, jj], dim=-1)

    def equirectangular_image_to_perspective(
            self,
            equirectangular_images: torch.Tensor,  # (B, H, W, 3)
            R_w2c: torch.Tensor,  # (B, 3, 3)
            to_uint: bool = True
    ) -> torch.Tensor:
        """
        Converts a batch of equirectangular images to perspective images using batched rotation matrices.

        Args:
            equirectangular_images (torch.Tensor): (B, H, W, 3)
            R_w2c (torch.Tensor): (B, 3, 3)
            to_uint (bool): Convert output images to uint8. Default: True

        Returns:
            torch.Tensor: Perspective images (B, crop_H, crop_W, 3)
        """
        #  Create meshgrid of pixel coordinates (i, j) in perspective image space
        cr_grid = self.conversions.ij.to_cr(
            self.ij_grid.expand(*equirectangular_images.shape[:-3], -1, -1, -1).to(equirectangular_images.device),
            R_w2c)
        c = cr_grid[..., 0]
        r = cr_grid[..., 1]

        # Normalize to [-1, 1] for grid_sample
        c_normalized = (c / (self.cfg.equirectangular_width - 1)) * 2 - 1
        r_normalized = (r / (self.cfg.equirectangular_height - 1)) * 2 - 1
        cr_normalized_grid = torch.stack([c_normalized, r_normalized], dim=-1)

        perspective_images = torch.nn.functional.grid_sample(
            equirectangular_images.flatten(0, -4).permute(0, 3, 1, 2).float(),
            cr_normalized_grid.flatten(0, -4),
            mode='bilinear', padding_mode='border',
            align_corners=None).permute(0, 2, 3, 1)
        perspective_images = perspective_images.view(*equirectangular_images.shape[:-3], self.cfg.crop_height,
                                                     self.cfg.crop_width, -1)

        # Convert to uint8 if requested
        if to_uint:
            perspective_images = (perspective_images.clamp(0, 255)).to(torch.uint8)

        return perspective_images

    def perspective_image_to_equirectangular(self, perspective_images, R_w2c, to_uint=True):
        """
        Projects perspective images back into the equirectangular image space using inverse rotation.

        Args:
            perspective_images (torch.Tensor): (B, H_p, W_p, 3)
            R_w2c (torch.Tensor): (B, 3, 3)
            to_uint (bool): Whether to return result as uint8

        Returns:
            torch.Tensor: Equirectangular image (B, H_eq, W_eq, 3) with perspective regions filled.
        """
        B, H_p, W_p, C = perspective_images.shape

        cr_grid = self.cr_grid.unsqueeze(0).expand(B, -1, -1, -1).to(perspective_images.device)

        vc = self.conversions.cr.to_vc(cr_grid,
                                       R_w2c)  # Convert to perspective pixel coordinates
        ij = self.conversions.vc.to_ij(vc)  # (B, H_eq, W_eq, 2)

        # Normalize coordinates for grid_sample in [-1, 1]
        i_norm = (ij[..., 0] / (W_p - 1)) * 2 - 1
        j_norm = (ij[..., 1] / (H_p - 1)) * 2 - 1
        grid = torch.stack([i_norm, j_norm], dim=-1)  # (B, H_eq, W_eq, 2)

        # Step 7: Sample from perspective image
        perspective_images = perspective_images.permute(0, 3, 1, 2)  # (B, 3, H_p, W_p)
        eq_recon = torch.nn.functional.grid_sample(
            perspective_images.float(),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )  # (B, 3, H_eq, W_eq)

        eq_recon = eq_recon.permute(0, 2, 3, 1)  # (B, H_eq, W_eq, 3)

        # Remove the "double projection" artifact in spherical mappings
        # Determine which ij coords fall inside the valid perspective image
        valid_i = (ij[..., 0] >= 0) & (ij[..., 0] < W_p)
        valid_j = (ij[..., 1] >= 0) & (ij[..., 1] < H_p)
        valid_mask = (vc[..., 0] > 0) & valid_i & valid_j

        if to_uint:
            eq_recon = torch.clamp(eq_recon, 0, 255).to(torch.uint8)

        return eq_recon * valid_mask[..., None], valid_mask

    def equirectangular_image_to_equirectangular_ego(
            self,
            input_equirectangular_img: torch.Tensor,
            rotation_matrices: torch.Tensor,
            apply_fov_mask: bool = False,
            to_uint: bool = True
    ):
        """
        Re-orients a batch of equirectangular images based on rotation_matrices.

        Args:
            input_equirectangular_img (torch.Tensor): (B, H_eq, W_eq, C)
            rotation_matrices (torch.Tensor): (B, 3, 3) - transforms from new_ego_frame to original_input_frame
            apply_fov_mask (bool): Whether to apply FOV mask
            to_uint (bool): Whether to return result as uint8

        Returns:
            torch.Tensor: (B, H_eq, W_eq, C) - reoriented equirectangular image
            torch.Tensor (optional): (B, H_eq, W_eq, 1) - FOV mask
        """
        B, H_eq, W_eq, C = input_equirectangular_img.shape
        device = input_equirectangular_img.device

        # rescaled = False
        # if input_equirectangular_img.max() > 1:
        #     rescaled = True
        #     input_equirectangular_img = input_equirectangular_img / 255.

        cr_grid = self.cr_grid.unsqueeze(0).expand(B, -1, -1, -1).to(device)
        vw = self.conversions.cr.to_vw(cr_grid)  # (H_eq, W_eq, 2) normalized

        # 2. Apply inverse rotation to get directions in input frame
        vw_rotated = self.conversions.vw.to_vc(vw, torch.inverse(rotation_matrices))  # (B, H_eq, W_eq, 3)

        # 3. Convert to (lon, lat) in radians
        x, y, z = vw_rotated[..., 0], vw_rotated[..., 1], vw_rotated[..., 2]
        lon = torch.atan2(y, x)
        lat = torch.atan2(z, torch.sqrt(x ** 2 + y ** 2 + 1e-9))

        # 4. Normalize to [-1, 1] for grid_sample
        u_norm = lon / torch.pi
        v_norm = -lat / (torch.pi / 2.0)
        grid = torch.stack([u_norm, v_norm], dim=-1)  # (B, H_eq, W_eq, 2)

        # 5. Sample image
        input_img = input_equirectangular_img.permute(0, 3, 1, 2).float()
        output_img = torch.nn.functional.grid_sample(
            input_img, grid, mode='bilinear', padding_mode='border', align_corners=False
        )
        output_img = output_img.permute(0, 2, 3, 1)

        # if rescaled:
        #     output_img = output_img * 255

        if to_uint:
            output_img = torch.clamp(output_img, 0, 255).to(torch.uint8)

        if not apply_fov_mask:
            return output_img

        # 6. Generate FOV mask from original spherical directions (vw)
        tan_half_fov_x = torch.tan(torch.deg2rad(torch.tensor(self.cfg.fov_x, device=device) / 2.0))
        tan_half_fov_y = torch.tan(torch.deg2rad(torch.tensor(self.cfg.fov_y, device=device) / 2.0))

        x_ego = vw[..., 0]  # forward
        y_ego = vw[..., 1]  # right
        z_ego = vw[..., 2]  # up

        front = x_ego > 1e-6
        u_proj = torch.full_like(x_ego, 2.0, device=device)
        v_proj = torch.full_like(x_ego, 2.0, device=device)

        u_proj[front] = (y_ego[front] / x_ego[front]) / tan_half_fov_x
        v_proj[front] = -(z_ego[front] / x_ego[front]) / tan_half_fov_y

        in_fov = (u_proj.abs() <= 1.0) & (v_proj.abs() <= 1.0)
        mask = (front & in_fov).float().unsqueeze(-1).expand(B, -1, -1, 1)

        masked_img = output_img * mask
        if to_uint:
            return torch.clamp(masked_img, 0, 255).to(torch.uint8)

        return masked_img
