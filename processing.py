"""
Aqua Vision - Image Pre-Processing Engine
Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to underwater imagery.
"""

from typing import Union
import cv2
import numpy as np
from PIL import Image


def apply_clahe(
    image: Union[Image.Image, np.ndarray],
    clip_limit: float = 3.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Applies CLAHE enhancement to an underwater image in LAB color space.

    Steps:
      1. Ensure image is in RGB format.
      2. Convert RGB image to LAB color space using OpenCV.
      3. Split channels into L, A, and B.
      4. Apply CLAHE with specified clipLimit and tileGridSize to the L (lightness) channel.
      5. Merge the enhanced L channel with original A and B channels.
      6. Convert the merged LAB image back to RGB.

    Args:
        image: PIL Image or numpy array in RGB format.
        clip_limit: Threshold for contrast limiting (default 3.0).
        tile_grid_size: Size of grid for histogram equalization (default (8, 8)).

    Returns:
        np.ndarray: Enhanced RGB image as a numpy array uint8.
    """
    if isinstance(image, Image.Image):
        # Convert PIL Image to RGB numpy array
        rgb_img = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            # Grayscale to RGB
            rgb_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA to RGB
            rgb_img = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            rgb_img = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # 1. Convert RGB to LAB color space
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)

    # 2. Split LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 3. Apply CLAHE to L (Lightness) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    # 4. Merge enhanced L channel back with A and B channels
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))

    # 5. Convert back to RGB
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    return enhanced_rgb
