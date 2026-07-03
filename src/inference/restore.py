"""
src/inference/restore.py
========================
Inference pipeline: load the trained model and restore retina images.

HOW TO USE (for beginners):
  1. Training must be complete and a "best.pt" checkpoint must exist.
  2. Call restore_single_image("path/to/degraded.jpg", "path/to/output.png")
  3. The AI produces a restored version at the output path.

UNCERTAINTY MAPS:
  We also offer Monte Carlo Dropout uncertainty estimation.
  During inference, we run the model N times with dropout still ON
  (instead of off, which is normal during inference). The variance across
  runs tells us "which pixels is the AI most uncertain about?"
  These uncertain regions should be flagged for clinical review.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from src.config import DEVICE, CROP_SIZE, INPUT_CHANNELS, OUTPUT_CHANNELS
from src.models.generator import UNetGenerator
from src.training.checkpoints import load_checkpoint
from src.utils.image_utils import (
    build_four_channel_input_tensor,
    load_image_as_float_array,
    save_float_array_as_image,
    tensor_to_float_array,
    resize_image,
)


# ---- Model loading ----------------------------------------

def load_generator_for_inference(
    checkpoint_path: str = "checkpoints/best.pt",
) -> UNetGenerator:
    """
    Load the trained generator from a checkpoint for inference-only use.

    Optimizer states are not loaded (not needed for inference).
    Model is put into eval() mode (disables dropout/batch-norm randomness).

    params: checkpoint_path — path to the .pt checkpoint file
    returns: UNetGenerator ready for inference
    side effects: loads weights into GPU/CPU memory
    """
    generator = UNetGenerator().to(DEVICE)

    # We need a dummy discriminator to satisfy load_checkpoint's signature,
    # but we immediately discard it after loading
    from src.models.discriminator import PatchGANDiscriminator
    dummy_disc = PatchGANDiscriminator().to(DEVICE)

    load_checkpoint(checkpoint_path, generator, dummy_disc)
    del dummy_disc

    generator.eval()
    print(f"Generator loaded from {checkpoint_path} (running on {DEVICE})")
    return generator


# ---- Tiled inference for large images ----------------------

def _pad_to_multiple(image: np.ndarray, multiple: int = 32) -> tuple[np.ndarray, tuple]:
    """
    Pad image so H and W are multiples of `multiple` (required by the network).
    Returns padded image and the original (H, W) so we can crop back later.
    """
    h, w = image.shape[:2]
    orig_shape = (h, w)
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return image, orig_shape


def restore_image_array(
    generator: UNetGenerator,
    image_rgb: np.ndarray,
) -> np.ndarray:
    """
    Restore a single image represented as a float32 numpy array.

    Handles images of any size by padding to a multiple of 32.
    The network was trained on 256×256 patches, but can run on
    larger images thanks to its fully-convolutional architecture.

    params:
        generator — loaded UNetGenerator in eval mode
        image_rgb — (H, W, 3) float32 array in [0, 1]
    returns: restored_image — (H, W, 3) float32 array in [0, 1]
    side effects: none beyond GPU memory allocation
    """
    padded, orig_shape = _pad_to_multiple(image_rgb, multiple=32)

    input_tensor = build_four_channel_input_tensor(padded)       # (4, H', W')
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)          # (1, 4, H', W')

    with torch.no_grad():
        output_tensor = generator(input_tensor)                   # (1, 3, H', W')

    output_arr = tensor_to_float_array(output_tensor[0])         # (H', W', 3) in [0,1]

    # Crop back to original size
    h, w = orig_shape
    return output_arr[:h, :w, :]


# ---- Single-image restore ----------------------------------

def restore_single_image(
    input_path: str,
    output_path: str,
    checkpoint_path: str = "checkpoints/best.pt",
    generator: Optional[UNetGenerator] = None,
) -> np.ndarray:
    """
    Load a degraded retina image, restore it, and save the result.

    params:
        input_path      — path to the degraded input image
        output_path     — where to save the restored output (PNG)
        checkpoint_path — model checkpoint to use (default: best.pt)
        generator       — optional pre-loaded generator (avoids reloading on every call)
    returns: restored_image as (H, W, 3) float32 numpy array
    side effects: writes the restored image to output_path
    """
    if generator is None:
        generator = load_generator_for_inference(checkpoint_path)

    print(f"Restoring: {input_path}")
    image = load_image_as_float_array(input_path)
    restored = restore_image_array(generator, image)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_float_array_as_image(restored, output_path)
    print(f"Saved restored image to: {output_path}")
    return restored


# ---- Monte Carlo Dropout uncertainty estimation ------------

def _enable_dropout(model: nn.Module) -> None:
    """
    Set all Dropout layers to training mode (so they DROP activations),
    even though the rest of the model is in eval mode.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()


def restore_with_uncertainty(
    generator: UNetGenerator,
    image_rgb: np.ndarray,
    n_samples: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce a restored image AND a per-pixel uncertainty map using
    Monte Carlo Dropout.

    By enabling dropout at inference time and running the model N times,
    we get N slightly different outputs. The variance across these outputs
    tells us how confident the model is at each pixel.

    HIGH uncertainty → the model is guessing → flag for clinical review.
    LOW uncertainty  → the model is confident.

    params:
        generator — UNetGenerator (must have Dropout layers in the bottleneck)
        image_rgb — (H, W, 3) float32 array in [0, 1]
        n_samples — number of Monte Carlo runs (more = better estimate, slower)
    returns:
        mean_restored  — (H, W, 3) float32, average of N restorations
        uncertainty_map — (H, W) float32, per-pixel standard deviation (higher = less certain)
    """
    generator.eval()
    _enable_dropout(generator)   # Enable dropout stochasticity

    padded, orig_shape = _pad_to_multiple(image_rgb, multiple=32)
    input_tensor = build_four_channel_input_tensor(padded).unsqueeze(0).to(DEVICE)

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = generator(input_tensor)    # (1, 3, H', W')
            samples.append(tensor_to_float_array(out[0]))    # (H', W', 3)

    generator.eval()   # Restore full eval mode (also disables dropout)

    stack          = np.stack(samples, axis=0)             # (N, H', W', 3)
    mean_restored  = stack.mean(axis=0)[:orig_shape[0], :orig_shape[1], :]
    uncertainty_map = stack.std(axis=0).mean(axis=-1)[:orig_shape[0], :orig_shape[1]]

    return mean_restored, uncertainty_map


# ---- Batch inference ---------------------------------------

def restore_batch(
    input_folder: str,
    output_folder: str,
    checkpoint_path: str = "checkpoints/best.pt",
    file_extensions: tuple = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
    compute_uncertainty: bool = False,
    n_mc_samples: int = 10,
) -> list[str]:
    """
    Restore all images in a folder and save results to another folder.

    params:
        input_folder       — folder containing degraded images
        output_folder      — folder where restored images will be saved
        checkpoint_path    — model checkpoint
        file_extensions    — which file types to process
        compute_uncertainty — if True, also save uncertainty maps
        n_mc_samples        — Monte Carlo samples for uncertainty (if enabled)
    returns: list of output file paths
    side effects: writes restored images (and optionally uncertainty maps) to output_folder
    """
    import glob

    generator = load_generator_for_inference(checkpoint_path)
    os.makedirs(output_folder, exist_ok=True)

    input_paths = []
    for ext in file_extensions:
        input_paths.extend(glob.glob(os.path.join(input_folder, f"*{ext}")))
        input_paths.extend(glob.glob(os.path.join(input_folder, f"*{ext.upper()}")))
    input_paths = sorted(set(input_paths))

    if not input_paths:
        print(f"No images found in {input_folder}")
        return []

    print(f"Found {len(input_paths)} images. Restoring...")
    output_paths = []

    from tqdm import tqdm
    for input_path in tqdm(input_paths):
        filename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_folder, f"{filename}_restored.png")

        image = load_image_as_float_array(input_path)

        if compute_uncertainty:
            restored, uncertainty = restore_with_uncertainty(generator, image, n_mc_samples)
            uncertainty_path = os.path.join(output_folder, f"{filename}_uncertainty.png")
            # Normalise uncertainty to [0, 1] for saving as greyscale image
            unc_normalised = np.clip(uncertainty / (uncertainty.max() + 1e-8), 0, 1)
            save_float_array_as_image(np.stack([unc_normalised]*3, axis=-1), uncertainty_path)
        else:
            restored = restore_image_array(generator, image)

        save_float_array_as_image(restored, output_path)
        output_paths.append(output_path)

    print(f"Done. {len(output_paths)} images restored to {output_folder}")
    return output_paths


# ---- ONNX export ------------------------------------------

def export_to_onnx(
    checkpoint_path: str = "checkpoints/best.pt",
    output_path: str = "checkpoints/rrin_generator.onnx",
    image_size: int = 256,
) -> None:
    """
    Export the trained generator to ONNX format for portable deployment.

    ONNX is a universal model format that can run on:
      - ONNX Runtime (fast CPU/GPU inference, no PyTorch needed)
      - TensorRT (NVIDIA GPU, very fast)
      - OpenCV DNN module

    params:
        checkpoint_path — path to the .pt checkpoint
        output_path     — where to save the .onnx file
        image_size      — spatial dimension (must match training crop size)
    side effects: writes the .onnx file to output_path
    """
    generator = load_generator_for_inference(checkpoint_path)
    dummy_input = torch.randn(1, INPUT_CHANNELS, image_size, image_size).to(DEVICE)

    torch.onnx.export(
        generator,
        dummy_input,
        output_path,
        input_names=["degraded_input"],
        output_names=["restored_output"],
        dynamic_axes={
            "degraded_input":   {0: "batch", 2: "height", 3: "width"},
            "restored_output":  {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
    )
    print(f"ONNX model exported to: {output_path}")
