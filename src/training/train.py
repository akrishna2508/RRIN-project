"""
src/training/train.py
=====================
The core training and validation loops.

WHAT HAPPENS IN ONE TRAINING EPOCH? (for beginners)
  1. The DataLoader gives us a mini-batch of (degraded_input, clean_target) pairs
  2. The generator produces a restored image from the degraded input
  3. We update the DISCRIMINATOR:
       - Feed it the real (input, target) pair → should output ~1
       - Feed it the fake (input, generator_output) pair → should output ~0
       - Compute LSGAN loss, backpropagate, step the discriminator's optimizer
  4. We update the GENERATOR:
       - Compute all 4 loss terms (adversarial + L1 + SSIM + perceptual)
       - Backpropagate, step the generator's optimizer
  5. Log the average losses for this batch
  Repeat for every batch in the training set → one epoch complete.

VALIDATION:
  After each epoch, run the generator on the validation set with no updates.
  Compute PSNR and SSIM to measure how good the restoration is.
  If SSIM improves, save a new "best" checkpoint.
"""

import collections
import logging
import math
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import DEVICE
from src.models.losses import (
    compute_discriminator_loss,
    compute_generator_loss,
    VGGPerceptualLoss,
    SSIMLoss,
)
from src.utils.image_utils import derive_fov_mask_from_input_tensor, tensor_to_float_array


# ---- PSNR / SSIM metric computation -----------------------

def compute_psnr(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between output and target.

    PSNR = 10 * log10(MAX² / MSE)
    Higher is better. Values above 30 dB are generally good.

    Both tensors should be in [-1, 1]; we rescale to [0, 1] for computation.

    params: output, target — (B, C, H, W) tensors in [-1, 1]
    returns: average PSNR over the batch (float)
    """
    # Rescale to [0, 1]
    out_01    = (output + 1.0) / 2.0
    target_01 = (target + 1.0) / 2.0

    mse = torch.mean((out_01 - target_01) ** 2)
    if mse < 1e-10:
        return 100.0    # Perfect match
    return float(10.0 * torch.log10(torch.tensor(1.0) / mse))


def compute_ssim_simple(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute a simplified SSIM using global (not windowed) statistics.
    Used during training for fast approximate validation.

    The full windowed SSIM is used in the final evaluation (metrics.py).

    params: output, target — (B, C, H, W) in [-1, 1]
    returns: mean SSIM value over batch (float, 0..1)
    """
    out_01    = (output + 1.0) / 2.0
    target_01 = (target + 1.0) / 2.0

    mu_x   = out_01.mean()
    mu_y   = target_01.mean()
    sigma_x = out_01.var()
    sigma_y = target_01.var()
    sigma_xy = ((out_01 - mu_x) * (target_01 - mu_y)).mean()

    C1, C2 = 0.0001, 0.0009   # (k1·L)² and (k2·L)² with L=1

    numerator   = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / (denominator + 1e-8))


def compute_psnr_ssim_batch(
    output_batch: torch.Tensor,
    target_batch: torch.Tensor,
) -> tuple[float, float]:
    """
    Compute average PSNR and SSIM for a batch of images.

    params:
        output_batch — (B, 3, H, W) generator outputs in [-1, 1]
        target_batch — (B, 3, H, W) clean targets in [-1, 1]
    returns: (mean_psnr, mean_ssim) floats
    side effects: none
    """
    psnr_values = []
    ssim_values = []
    B = output_batch.shape[0]

    for i in range(B):
        out  = output_batch[i:i+1]
        tgt  = target_batch[i:i+1]
        psnr_values.append(compute_psnr(out, tgt))
        ssim_values.append(compute_ssim_simple(out, tgt))

    return (
        sum(psnr_values) / len(psnr_values),
        sum(ssim_values) / len(ssim_values),
    )


# ---- One training epoch ------------------------------------

def train_one_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    perceptual_loss_module: VGGPerceptualLoss,
    ssim_loss_module: SSIMLoss,
    train_dataloader: torch.utils.data.DataLoader,
    epoch_index: int,
    logger: logging.Logger,
) -> dict[str, float]:
    """
    Run one full pass over the training dataloader, updating both
    discriminator and generator on every mini-batch.

    Training step order (from Section 9.3 of the project plan):
      1. Forward: generate fake output from degraded input
      2. Update D: compute D loss with DETACHED fake output; backprop; step D
      3. Update G: compute composite G loss (not detached); backprop; step G
      4. Log losses

    params:
        generator, discriminator       — the two networks
        generator_optimizer            — Adam for generator
        discriminator_optimizer        — Adam for discriminator
        perceptual_loss_module         — frozen VGG16 loss module
        ssim_loss_module               — SSIM loss module
        train_dataloader               — yields (degraded_input, real_target) batches
        epoch_index                    — current epoch (for logging)
        logger                         — Python logger
    returns: dict of average loss component values for this epoch
    side effects: updates generator's and discriminator's weights in-place
    """
    generator.train()
    discriminator.train()
    accumulated_losses: dict = collections.defaultdict(float)
    num_batches = 0

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch_index:03d} [train]", leave=False)

    for degraded_input, real_target in progress_bar:
        # FIXED: entire training step wrapped so a transient CUDA OOM on one
        # batch clears the cache and skips that batch instead of killing the
        # whole (multi-hour) run. Non-OOM errors are re-raised normally.
        try:
            degraded_input = degraded_input.to(DEVICE, non_blocking=True)  # (B, 4, H, W)
            real_target    = real_target.to(DEVICE, non_blocking=True)     # (B, 3, H, W)

            # FOV mask — only compute loss on pixels inside the fundus disc
            fov_mask = derive_fov_mask_from_input_tensor(degraded_input)  # (B, 1, H, W)

            # ---- Step 1: Generate fake output ----
            fake_output = generator(degraded_input)     # (B, 3, H, W)

            # ---- Step 2: Update Discriminator ----
            # DETACH fake_output so gradients don't flow into the generator here
            discriminator_optimizer.zero_grad(set_to_none=True)
            disc_loss = compute_discriminator_loss(
                discriminator, degraded_input, real_target, fake_output.detach()
            )
            disc_loss.backward()
            discriminator_optimizer.step()

            # ---- Step 3: Update Generator ----
            # NOT detached — gradients flow from discriminator's judgment into G
            generator_optimizer.zero_grad(set_to_none=True)
            total_gen_loss, loss_dict = compute_generator_loss(
                discriminator, perceptual_loss_module, ssim_loss_module,
                degraded_input, fake_output, real_target, fov_mask
            )
            total_gen_loss.backward()
            generator_optimizer.step()

            # ---- Step 4: Accumulate for logging ----
            loss_dict["discriminator"] = float(disc_loss)
            for key, val in loss_dict.items():
                accumulated_losses[key] += val
            num_batches += 1

            # Update progress bar with live loss values
            progress_bar.set_postfix(
                total=f"{loss_dict['total']:.3f}",
                l1=f"{loss_dict['l1']:.3f}",
                disc=f"{loss_dict['discriminator']:.3f}",
            )

        except RuntimeError as runtime_error:
            # FIXED: graceful recovery from CUDA out-of-memory on a single batch.
            if "out of memory" in str(runtime_error).lower():
                logger.warning(
                    f"CUDA OOM on a batch in epoch {epoch_index} — "
                    f"clearing cache and skipping this batch. "
                    f"If this recurs, lower batch_size in config.yaml."
                )
                # Drop references and free the cached allocator memory
                for _v in ("fake_output", "total_gen_loss", "disc_loss"):
                    if _v in dict(locals()):
                        del locals()[_v]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise   # any non-OOM RuntimeError is a real bug — re-raise it

    avg_losses = {k: v / max(1, num_batches) for k, v in accumulated_losses.items()}
    logger.info(f"Epoch {epoch_index:03d} [train] " +
                " | ".join(f"{k}={v:.4f}" for k, v in avg_losses.items()))
    return avg_losses


# ---- One validation epoch ----------------------------------

def validate_one_epoch(
    generator: nn.Module,
    validation_dataloader: torch.utils.data.DataLoader,
    epoch_index: int,
    logger: logging.Logger,
) -> dict[str, float]:
    """
    Run the generator in evaluation mode over the full validation dataloader.
    Compute PSNR and SSIM. No weight updates happen here.

    params:
        generator             — UNetGenerator in eval mode
        validation_dataloader — yields (degraded_input, real_target) batches
        epoch_index           — current epoch (for logging)
        logger                — Python logger
    returns: dict with keys "psnr" and "ssim"
    side effects: none (no weight updates; torch.no_grad() context is used)
    """
    generator.eval()
    accumulated_psnr = 0.0
    accumulated_ssim = 0.0
    num_batches = 0

    progress_bar = tqdm(validation_dataloader, desc=f"Epoch {epoch_index:03d} [val]", leave=False)

    with torch.no_grad():
        for degraded_input, real_target in progress_bar:
            degraded_input = degraded_input.to(DEVICE)
            real_target    = real_target.to(DEVICE)

            restored_output = generator(degraded_input)

            batch_psnr, batch_ssim = compute_psnr_ssim_batch(restored_output, real_target)
            accumulated_psnr += batch_psnr
            accumulated_ssim += batch_ssim
            num_batches += 1

            progress_bar.set_postfix(
                psnr=f"{batch_psnr:.2f}",
                ssim=f"{batch_ssim:.4f}"
            )

    avg_metrics = {
        "psnr": accumulated_psnr / max(1, num_batches),
        "ssim": accumulated_ssim / max(1, num_batches),
    }
    logger.info(f"Epoch {epoch_index:03d} [val] psnr={avg_metrics['psnr']:.2f} ssim={avg_metrics['ssim']:.4f}")
    return avg_metrics
