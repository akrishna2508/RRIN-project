"""
src/evaluation/metrics.py
==========================
Final evaluation on the held-out test set.

METRICS COMPUTED (for beginners):
  PSNR  — Peak Signal-to-Noise Ratio. Higher = better.
           Measures raw pixel accuracy in dB. >30 dB is good.

  SSIM  — Structural Similarity Index. Range 0–1, higher = better.
           Measures how similar the structure (edges, patterns) is.
           More correlated with human perception than PSNR.

  LPIPS — Learned Perceptual Image Patch Similarity. Lower = better.
           Uses a deep network trained to match human quality judgments.
           Complements SSIM — measures perceptual quality differently.

  FID   — Fréchet Inception Distance. Lower = better.
           Compares the DISTRIBUTION of restored images to the distribution
           of real clean images (not individual pairs). Measures realism.

  Downstream proxy — vessel segmentation Dice score.
           Runs a pre-trained vessel detector on both restored and original.
           Tests whether restoration helps or hurts a real clinical task.
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import DEVICE
from src.training.train import compute_psnr, compute_ssim_simple
from src.utils.image_utils import tensor_to_float_array, derive_fov_mask_from_input_tensor


# ---- Windowed SSIM (proper, for final evaluation) ----------

def compute_full_ssim(
    output_batch: torch.Tensor,
    target_batch: torch.Tensor,
) -> float:
    """
    Compute proper windowed SSIM using skimage.metrics for the final test report.

    params:
        output_batch — (B, 3, H, W) generator outputs in [-1, 1]
        target_batch — (B, 3, H, W) clean targets in [-1, 1]
    returns: average SSIM over the batch
    """
    from skimage.metrics import structural_similarity as sk_ssim
    ssim_values = []
    B = output_batch.shape[0]
    for i in range(B):
        out_np = tensor_to_float_array(output_batch[i])      # (H, W, 3) in [0,1]
        tgt_np = tensor_to_float_array(target_batch[i])
        ssim   = sk_ssim(out_np, tgt_np, data_range=1.0, channel_axis=2)
        ssim_values.append(ssim)
    return float(np.mean(ssim_values))


def compute_full_psnr(
    output_batch: torch.Tensor,
    target_batch: torch.Tensor,
) -> float:
    """
    Compute PSNR using skimage.metrics for the final test report.
    """
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    psnr_values = []
    B = output_batch.shape[0]
    for i in range(B):
        out_np = tensor_to_float_array(output_batch[i])
        tgt_np = tensor_to_float_array(target_batch[i])
        psnr   = sk_psnr(tgt_np, out_np, data_range=1.0)
        psnr_values.append(psnr)
    return float(np.mean(psnr_values))


# ---- Vessel segmentation proxy metric ----------------------

def load_frozen_vessel_segmentation_model() -> Optional[nn.Module]:
    """
    Load a pre-trained vessel segmentation network (trained on DRIVE/STARE).

    This model is used ONLY for evaluation — it tests whether our restoration
    makes vessel detection easier or harder. Its weights are never updated.

    Returns None if no segmentation model checkpoint is found.
    The checkpoint should be at: checkpoints/vessel_segmenter.pt

    To obtain a vessel segmentation model:
      - Train one on DRIVE/STARE using any standard UNet implementation, OR
      - Download a pre-trained one from the DRIVE challenge page
    """
    model_path = "checkpoints/vessel_segmenter.pt"
    if not os.path.exists(model_path):
        print(
            "  Vessel segmentation model not found at checkpoints/vessel_segmenter.pt\n"
            "  Skipping downstream evaluation.\n"
            "  To enable: place a pre-trained vessel segmentation checkpoint there."
        )
        return None

    try:
        # FIXED: weights_only=False is required here because this optional
        # checkpoint may be a full pickled nn.Module. Under torch >= 2.6 the
        # default weights_only=True would raise on any non-tensor object.
        # This loader is fully optional (returns None if anything goes wrong),
        # so a failure here never blocks the main evaluation.
        model = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        print("  Vessel segmentation model loaded for downstream evaluation.")
        return model
    except Exception as e:
        print(f"  Failed to load vessel segmentation model: {e}")
        return None


def compute_vessel_dice(
    pred_segmentation: np.ndarray,
    gt_segmentation: np.ndarray,
) -> float:
    """
    Compute the Dice coefficient between predicted and ground-truth vessel maps.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Range [0, 1], higher is better. 1 = perfect match.
    """
    pred_binary = (pred_segmentation > 0.5).astype(np.float32)
    gt_binary   = (gt_segmentation   > 0.5).astype(np.float32)
    intersection = (pred_binary * gt_binary).sum()
    return float(2.0 * intersection / (pred_binary.sum() + gt_binary.sum() + 1e-8))


# ---- Full test-set evaluation ------------------------------

def evaluate_on_test_set(
    generator: nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    lpips_model: Optional[nn.Module],
    vessel_segmentation_model: Optional[nn.Module],
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Run the generator on the entire test set and compute all metrics.

    params:
        generator                — trained UNetGenerator in eval mode
        test_dataloader          — yields (degraded_input, real_target) batches
        lpips_model              — pre-trained LPIPS network (or None to skip)
        vessel_segmentation_model — pre-trained vessel segmenter (or None to skip)
        logger                   — Python logger
    returns: DataFrame with one row per image, columns = metric names
    side effects: none (no weight updates)
    """
    generator.eval()
    all_results = []

    print("Evaluating on test set...")

    with torch.no_grad():
        for batch_idx, (degraded_input, real_target) in enumerate(
            tqdm(test_dataloader, desc="Test evaluation")
        ):
            degraded_input = degraded_input.to(DEVICE)
            real_target    = real_target.to(DEVICE)

            restored_output = generator(degraded_input)

            B = degraded_input.shape[0]
            for i in range(B):
                out_single = restored_output[i:i+1]   # (1, 3, H, W)
                tgt_single = real_target[i:i+1]

                result = {
                    "batch":    batch_idx,
                    "sample":   i,
                    "psnr":     compute_full_psnr(out_single, tgt_single),
                    "ssim":     compute_full_ssim(out_single, tgt_single),
                }

                # LPIPS (if available)
                if lpips_model is not None:
                    try:
                        lpips_val = lpips_model(out_single, tgt_single)
                        result["lpips"] = float(lpips_val.mean())
                    except Exception:
                        result["lpips"] = float("nan")

                # Vessel segmentation Dice (if available)
                if vessel_segmentation_model is not None:
                    try:
                        out_np  = tensor_to_float_array(out_single[0])   # (H, W, 3)
                        tgt_np  = tensor_to_float_array(tgt_single[0])

                        # Convert to grayscale (H, W, 1) tensor for segmenter
                        out_gray = torch.from_numpy(np.mean(out_np, axis=2, keepdims=True)).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                        tgt_gray = torch.from_numpy(np.mean(tgt_np, axis=2, keepdims=True)).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

                        seg_out = (vessel_segmentation_model(out_gray) > 0.5).float().cpu().numpy()[0, 0]
                        seg_tgt = (vessel_segmentation_model(tgt_gray) > 0.5).float().cpu().numpy()[0, 0]
                        result["vessel_dice"] = compute_vessel_dice(seg_out, seg_tgt)
                    except Exception:
                        result["vessel_dice"] = float("nan")

                all_results.append(result)

    results_df = pd.DataFrame(all_results)

    # Print summary
    summary = results_df[["psnr", "ssim"]].agg(["mean", "std"])
    logger.info(f"Test set results:\n{summary.to_string()}")
    print(f"\n{'='*50}")
    print("FINAL TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"  PSNR:  {results_df['psnr'].mean():.2f} ± {results_df['psnr'].std():.2f} dB")
    print(f"  SSIM:  {results_df['ssim'].mean():.4f} ± {results_df['ssim'].std():.4f}")
    if "lpips" in results_df.columns:
        print(f"  LPIPS: {results_df['lpips'].mean():.4f} ± {results_df['lpips'].std():.4f}  (lower=better)")
    if "vessel_dice" in results_df.columns:
        print(f"  Vessel Dice: {results_df['vessel_dice'].mean():.4f}")
    print(f"{'='*50}\n")

    return results_df
