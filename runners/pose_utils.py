import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def focal2fov(focal_length, width):
    """
    Convert focal length to field of view in degrees.
    """
    return 2 * np.arctan(width / (2 * focal_length)) * 180 / np.pi


class VGGTPoseRunner:
    def __init__(self, device):
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.device = device
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device).eval()

    def preprocess(self, images, mode="crop"):
        # Images: A 4D tensor with shape (F, H, W, 3).
        return preprocess_image_batch(images, mode)

    @staticmethod
    def convert_to_tapvid360_format(c2w_vggt):
        """
        Applies a coordinate space transformation in PyTorch.

        Args:
            c2w_vggt (torch.Tensor): Camera-to-world matrices, shape (3, 3) or (N, 3, 3).

        Returns:
            torch.Tensor: The transformed matrices.
        """
        # --- Define constant matrices outside the function for efficiency ---
        # (Note: This matrix is its own inverse)
        INV_R_LIB = torch.tensor(
            [[0., -1., 0.],
             [-1., 0., 0.],
             [0., 0., -1.]]
        )

        # (Note: The inverse is the transpose)
        INV_R_ALIGN = torch.tensor(
            [[0., -1., 0.],
             [0., 0., 1.],
             [-1., 0., 0.]]
        )
        # Move constant tensors to the same device and dtype as the input
        inv_R_lib_d = INV_R_LIB.to(device=c2w_vggt.device, dtype=c2w_vggt.dtype)
        inv_R_align_d = INV_R_ALIGN.to(device=c2w_vggt.device, dtype=c2w_vggt.dtype)

        # Apply the transformation using torch matrix multiplication
        # Broadcasting handles (N, 3, 3) or (3, 3) inputs
        c2w_tapvid360 = inv_R_lib_d @ c2w_vggt @ inv_R_align_d

        return c2w_tapvid360

    @staticmethod
    def align_to_ground_truth_start_rotations(pred_rots, initial_gt_rot):
        """
        Calculates corrected prediction rotations in PyTorch.

        Args:
            pred_rots (torch.Tensor): Predicted rotation tensor (e.g., shape [N, 3, 3]).
                                      Must be on the same device or gt_rots will be moved.
            gt_rots (torch.Tensor): Ground truth rotation tensor (e.g., shape [3, 3]).

        Returns:
            torch.Tensor: The aligned predicted rotations.
        """
        # Ensure gt_rots[0] is on the same device as pred_rots
        gt_rots_0 = initial_gt_rot.to(device=pred_rots.device, dtype=pred_rots.dtype)

        # Calculate the inverse of the first predicted rotation
        inv_pred_rots_0 = torch.linalg.inv(pred_rots[0])

        # Calculate the correction matrix
        R_correction = gt_rots_0 @ inv_pred_rots_0

        # Apply the correction to all predicted rotations
        # This uses broadcasting: (3, 3) @ (N, 3, 3) -> (N, 3, 3)
        pred_rots_aligned = R_correction @ pred_rots

        return pred_rots_aligned

    @torch.no_grad()
    def run(self, images, mode="crop"):
        images = self.preprocess(images, mode).to(self.device)
        with torch.cuda.amp.autocast(dtype=self.dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = self.model.aggregator(images)
        pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        focal_length = intrinsic[0, 0, 0, 0].cpu().item()  # focal length in pixels
        fov_x = torch.tensor(focal2fov(focal_length, images.shape[-1]), dtype=torch.float32).item()

        return extrinsic, fov_x


def preprocess_image_batch(image_batch_tensor, mode="crop"):
    """
    Preprocesses a 4D batch of images.

    This function has been modified to exactly match the logic of the
    'load_and_preprocess_images' (file-path) function. It does this by
    converting tensors back to PIL Images, resizing with PIL, and then
    converting back to tensors.

    Args:
        image_batch_tensor (torch.Tensor): A 4D tensor with shape (B, H, W, 3).
                                           Values are assumed to be [0, 255] uint8 or [0, 1] float.
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and centre crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (B, 3, H_out, W_out).
                      Output will be identical to 'load_and_preprocess_images'.
    """

    # --- Input Validation ---
    if image_batch_tensor.dim() != 4:
        raise ValueError(f"Input tensor must be 4D (B, H, W, 3), but got {image_batch_tensor.dim()}D")
    if image_batch_tensor.shape[-1] != 3:
        raise ValueError(f"Last dimension must be 3 (channels), but got {image_batch_tensor.shape[-1]}")
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    target_size = 518

    # --- 1. Convert Tensors to PIL Images ---
    # This is the core change to match the original function's logic.

    # If input is [0, 1] float, convert to [0, 255] uint8
    if image_batch_tensor.dtype != torch.uint8:
        # Use .round() to mimic 'load_and_preprocess_images' behavior
        images_uint8 = (image_batch_tensor * 255.0).round().to(torch.uint8)
    else:
        images_uint8 = image_batch_tensor

    # Move to CPU for .numpy() and PIL conversion
    images_uint8_cpu = images_uint8.cpu().numpy()

    # Loop through batch (THIS IS THE SLOW PART)
    images_pil_list = []
    for i in range(images_uint8_cpu.shape[0]):
        # Get one (H, W, 3) image
        img_np = images_uint8_cpu[i]
        # Convert to PIL Image
        img_pil = Image.fromarray(img_np, 'RGB')
        images_pil_list.append(img_pil)

    # --- 2. Re-implement the original PIL-based processing ---
    processed_tensors = []

    for img in images_pil_list:
        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Set width to 518px, calculate height
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height) using PIL
        # This is the same as the original function
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

        # Convert to tensor (0, 1) AFTER resizing
        # We use the functional 'to_tensor' which is what TF.ToTensor() calls
        img = TF.to_tensor(img)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y: start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        processed_tensors.append(img)

    # --- 3. Stack final tensors ---
    # Since the input was a uniform batch, all outputs will have the
    # same shape, so we can just stack them.

    images = torch.stack(processed_tensors)

    # Move back to the original device
    return images.to(device=image_batch_tensor.device)
