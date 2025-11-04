import cv2
import numpy as np


def overlay_mask_over_image(image, mask, alpha=0.5, color=(0, 255, 0)):
    # Ensure mask and image have the same dimensions
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask dimensions do not match.")
    if not mask.any():
        return image
    if mask.max() > 1:
        mask = mask // 255
    overlay_mask = np.zeros_like(image)
    overlay_mask[mask == 1] = color

    anno_img = image.copy()
    anno_img[mask == 1] = cv2.addWeighted(image[mask == 1], 1 - alpha, overlay_mask[mask == 1], alpha, 0)

    return anno_img
