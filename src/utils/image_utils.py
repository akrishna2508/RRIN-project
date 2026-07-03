"""
src/utils/image_utils.py
========================
Pure utility functions for image I/O and tensor preparation.
These are called by the Dataset classes and the inference pipeline.

BEGINNER NOTES:
  - "float array" means pixel values are 0.0–1.0 (not 0–255).
  - "tensor" is the format PyTorch uses for computation (like a matrix).
  - "normalise to [-1, 1]" shifts pixel range so the network trains better.
"""

import cv2
import numpy as np
import torch
from PIL import Image


# ---- Image loading -----------------------------------------

def load_image_as_float_array(file_path: str) -> np.ndarray:
    """
    Read an image from disk and return it as a float32 array
    with pixel values in the range [0.0, 1.0].

    Handles JPEG, PNG, TIFF, and BMP formats automatically.
    Always returns an RGB array of shape (H, W, 3).
    """
    img = Image.open(file_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def save_float_array_as_image(array: np.ndarray, file_path: str) -> None:
    """
    Save a float32 array (values in [0, 1]) as a PNG image.
    """
    uint8_arr = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(uint8_arr).save(file_path)


# ---- Spatial operations ------------------------------------

def center_crop_single(image: np.ndarray, crop_size: int) -> np.ndarray:
    """
    Crop the centre (crop_size × crop_size) patch from a single image.
    If the image is smaller than crop_size, it is padded with zeros first.
    """
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        h, w = image.shape[:2]
    top  = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return image[top: top + crop_size, left: left + crop_size]


def center_crop_pair(
    image_a: np.ndarray,
    image_b: np.ndarray,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop the same centre patch from both images in a (input, target) pair.
    This guarantees the two images stay spatially aligned.
    """
    # Both images must be the same size; resize B to A's size if needed.
    h_a, w_a = image_a.shape[:2]
    h_b, w_b = image_b.shape[:2]
    if (h_a, w_a) != (h_b, w_b):
        image_b = cv2.resize(image_b, (w_a, h_a), interpolation=cv2.INTER_LINEAR)

    top  = max(0, (h_a - crop_size) // 2)
    left = max(0, (w_a - crop_size) // 2)
    return (
        image_a[top: top + crop_size, left: left + crop_size],
        image_b[top: top + crop_size, left: left + crop_size],
    )


def resize_image(image: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize image so its shortest side equals target_size, then centre-crop.
    Keeps aspect ratio intact during resizing.
    """
    h, w = image.shape[:2]
    scale = target_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return center_crop_single(resized, target_size)


# ---- Tensor conversion -------------------------------------

def normalize_and_to_tensor(image: np.ndarray) -> torch.FloatTensor:
    """
    Convert an (H, W, 3) float32 numpy array in [0, 1]
    to a (3, H, W) float32 PyTorch tensor in [-1, 1].

    The network expects values in [-1, 1] because:
      - tanh output activations produce [-1, 1]
      - this symmetric range trains more stably than [0, 1]
    """
    arr = (image * 2.0) - 1.0          # shift [0,1] → [-1,1]
    arr = np.transpose(arr, (2, 0, 1)) # HWC → CHW
    return torch.from_numpy(arr.copy()).float()


def build_four_channel_input_tensor(degraded_image: np.ndarray) -> torch.FloatTensor:
    """
    Build the 4-channel input tensor the generator expects:
      Channels 0-2 = normalised RGB image  (range [-1, 1])
      Channel   3  = green channel copy    (range [-1, 1])

    The extra green channel is included because retinal vessel contrast
    is highest in the green channel — giving it its own dedicated input
    channel helps the network preserve fine vascular structure.
    """
    rgb_tensor   = normalize_and_to_tensor(degraded_image)       # (3, H, W)
    green_chan   = rgb_tensor[1:2, :, :]                          # (1, H, W), already in [-1,1]
    return torch.cat([rgb_tensor, green_chan], dim=0)              # (4, H, W)


def tensor_to_float_array(tensor: torch.FloatTensor) -> np.ndarray:
    """
    Convert a (C, H, W) tensor in [-1, 1] back to an (H, W, C) float32
    numpy array in [0, 1]. Used for saving results and computing metrics.
    """
    arr = tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))          # CHW → HWC
    arr = (arr + 1.0) / 2.0                     # [-1,1] → [0,1]
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


# ---- VGG normalisation -------------------------------------

_VGG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_VGG_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def rescale_to_imagenet_normalization(batch: torch.FloatTensor) -> torch.FloatTensor:
    """
    Convert a batch of images from the network's [-1, 1] training range
    to the ImageNet-normalised range that VGG16 expects.

    VGG was trained with per-channel mean subtraction (mean=[0.485,0.456,0.406]
    and std=[0.229,0.224,0.225] on [0,1]-scaled pixels), so we must apply
    the same transform before passing images through the frozen VGG.
    """
    device = batch.device
    mean = _VGG_MEAN.to(device)
    std  = _VGG_STD.to(device)
    pixels_0_1 = (batch + 1.0) / 2.0            # [-1,1] → [0,1]
    return (pixels_0_1 - mean) / std


# ---- FOV mask from tensor ----------------------------------

def derive_fov_mask_from_input_tensor(
    input_tensor_batch: torch.FloatTensor,
) -> torch.FloatTensor:
    """
    Derive a rough field-of-view (FOV) mask from a batch of input tensors.

    The FOV mask marks which pixels are inside the circular fundus disc.
    Pixels outside the disc are pure black in all fundus photos; we identify
    them by finding pixels where the average channel value is very low.

    Returns a (B, 1, H, W) float tensor of 0.0/1.0 values
    (1 = inside disc, 0 = outside/black border).
    """
    # Convert from [-1, 1] to [0, 1]
    imgs_01 = (input_tensor_batch[:, :3, :, :] + 1.0) / 2.0
    mean_channel = imgs_01.mean(dim=1, keepdim=True)            # (B, 1, H, W)
    mask = (mean_channel > 0.05).float()                        # threshold at near-black
    return mask


# ---- Misc array helpers ------------------------------------

def rescale_array_to_range(
    arr: np.ndarray,
    out_min: float,
    out_max: float,
) -> np.ndarray:
    """Linearly rescale array values from their current range to [out_min, out_max]."""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-9:
        return np.full_like(arr, (out_min + out_max) / 2.0)
    normalised = (arr - arr_min) / (arr_max - arr_min)
    return normalised * (out_max - out_min) + out_min
