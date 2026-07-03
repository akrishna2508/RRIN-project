"""
src/training/domain_adaptation.py
==================================
Tier-3: CycleGAN-style unpaired domain adaptation fine-tuning.

WHAT PROBLEM DOES THIS SOLVE? (for beginners)
  Our synthetic degradation pipeline (Tier 1) generates realistic-looking
  corruptions, but it doesn't perfectly match every real camera artifact.
  If we only train on synthetic data, the network might:
    - Work perfectly on "clean → synthetic-dirty → restored"
    - Fail on real clinical images where artifacts are slightly different

  This stage closes that gap by training the network on REAL degraded images,
  even though we have no matching clean images for them.

HOW CYCLE CONSISTENCY WORKS:
  We have two domains:
    A = "synthetically degraded" images (from our Tier-1 pipeline)
    B = "real degraded" images (from the clinic, no clean pair)

  Two generators:
    G:  A → restored (our main generator, already pre-trained)
    F:  restored + noise → re-degraded (a new degradation generator)

  Two discriminators:
    D_B: "does this look like a REAL degraded image?"
    D_A: "does this look like a SYNTHETICALLY degraded image?"

  Loss: L_cycle = ||G(F(G(x))) - G(x)||₁   (forward cycle)
               + ||F(G(F(y))) - F(y)||₁   (backward cycle)

  The cycle loss forces: if you restore then re-degrade, you get back
  to roughly where you started. This prevents G from collapsing to
  a generic "average clean retina" that ignores the input.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.config import DEVICE, LAMBDA_CYCLE, LEARNING_RATE, ADAM_BETA1, ADAM_BETA2
from src.utils.image_utils import derive_fov_mask_from_input_tensor


def run_domain_adaptation_finetuning(
    generator: nn.Module,
    degradation_generator: nn.Module,
    real_domain_discriminator: nn.Module,
    synthetic_domain_discriminator: nn.Module,
    synthetic_paired_dataloader: torch.utils.data.DataLoader,
    real_unpaired_dataloader: torch.utils.data.DataLoader,
    num_finetune_epochs: int = 20,
    logger: logging.Logger = None,
) -> nn.Module:
    """
    Fine-tune the main generator using CycleGAN-style cycle-consistency.

    This function is called AFTER the main Tier-1 training has converged
    (i.e., after the best checkpoint has been loaded).

    params:
        generator                    — pre-trained UNetGenerator (G: degraded → clean)
        degradation_generator        — new UNetGenerator (F: clean+noise → degraded)
        real_domain_discriminator    — PatchGANDiscriminator for real degraded domain
        synthetic_domain_discriminator — PatchGANDiscriminator for synthetic domain
        synthetic_paired_dataloader  — Tier-1 dataloader (yields paired batches)
        real_unpaired_dataloader     — Tier-3 dataloader (yields single real images)
        num_finetune_epochs          — number of fine-tuning epochs (default 20)
        logger                       — Python logger
    returns: fine-tuned generator
    side effects: updates generator, degradation_generator, and both discriminator weights

    KEY DIFFERENCE FROM STANDARD CycleGAN:
      We initialise G from the pre-trained Tier-1 weights (not random).
      This means cycle-consistency acts as a REGULARISER rather than the
      sole training signal — we preserve what was learned in Tier 1.
    """
    if logger is None:
        import logging as _logging
        logger = _logging.getLogger("domain_adaptation")

    # Separate optimizers for all four networks
    # Lower LR for G since it's already pre-trained
    g_lr  = LEARNING_RATE * 0.1   # Fine-tuning uses 10x lower LR
    f_lr  = LEARNING_RATE
    d_lr  = LEARNING_RATE

    opt_G  = torch.optim.Adam(generator.parameters(),              lr=g_lr, betas=(ADAM_BETA1, ADAM_BETA2))
    opt_F  = torch.optim.Adam(degradation_generator.parameters(),  lr=f_lr, betas=(ADAM_BETA1, ADAM_BETA2))
    opt_DB = torch.optim.Adam(real_domain_discriminator.parameters(),     lr=d_lr, betas=(ADAM_BETA1, ADAM_BETA2))
    opt_DA = torch.optim.Adam(synthetic_domain_discriminator.parameters(), lr=d_lr, betas=(ADAM_BETA1, ADAM_BETA2))

    # Iterators — we zip the two dataloaders (real and synthetic)
    # When one is exhausted, we restart it
    def _cycle_dataloader(dl):
        while True:
            for batch in dl:
                yield batch

    real_iter = _cycle_dataloader(real_unpaired_dataloader)
    steps_per_epoch = len(synthetic_paired_dataloader)

    generator.train()
    degradation_generator.train()
    real_domain_discriminator.train()
    synthetic_domain_discriminator.train()

    for epoch in range(num_finetune_epochs):
        running_losses = {"G": 0.0, "F": 0.0, "D_A": 0.0, "D_B": 0.0, "cycle": 0.0}

        progress_bar = tqdm(
            synthetic_paired_dataloader,
            desc=f"DA Epoch {epoch+1}/{num_finetune_epochs}",
            leave=False
        )

        for synthetic_batch in progress_bar:
            # Unpack synthetic paired batch (input=degraded, target=clean)
            degraded_synth, clean_synth = synthetic_batch
            degraded_synth = degraded_synth.to(DEVICE)   # (B, 4, H, W)
            clean_synth    = clean_synth.to(DEVICE)       # (B, 3, H, W)

            # Get a batch of real degraded images (no target)
            real_degraded = next(real_iter).to(DEVICE)    # (B, 4, H, W)

            # ---- Restore both synthetic and real degraded images ----
            restored_synth = generator(degraded_synth)    # G(x_synth) → (B, 3, H, W)
            restored_real  = generator(real_degraded)     # G(x_real)  → (B, 3, H, W)

            # ---- Re-degrade the restored outputs ----
            # F takes a 3-channel restored image; we pad it to 4ch for compatibility
            restored_synth_4ch = torch.cat([restored_synth, restored_synth[:, 1:2, :, :]], dim=1)
            restored_real_4ch  = torch.cat([restored_real,  restored_real[:, 1:2, :, :]],  dim=1)

            re_degraded_synth = degradation_generator(restored_synth_4ch)[:, :3, :, :]  # F(G(x_synth))
            re_degraded_real  = degradation_generator(restored_real_4ch)[:, :3, :, :]   # F(G(x_real))

            # ---- Update Discriminator D_B (real vs restored_real) ----
            opt_DB.zero_grad()
            real_patch_logits     = real_domain_discriminator(real_degraded, real_degraded[:, :3, :, :])
            restored_patch_logits = real_domain_discriminator(real_degraded, restored_real.detach())
            d_b_loss = 0.5 * (
                F.mse_loss(real_patch_logits,     torch.ones_like(real_patch_logits) * 0.9)
                + F.mse_loss(restored_patch_logits, torch.zeros_like(restored_patch_logits))
            )
            d_b_loss.backward()
            opt_DB.step()

            # ---- Update Discriminator D_A (synth vs re-degraded_synth) ----
            opt_DA.zero_grad()
            synth_patch_logits    = synthetic_domain_discriminator(degraded_synth, degraded_synth[:, :3, :, :])
            redeg_patch_logits    = synthetic_domain_discriminator(degraded_synth, re_degraded_synth.detach())
            d_a_loss = 0.5 * (
                F.mse_loss(synth_patch_logits, torch.ones_like(synth_patch_logits) * 0.9)
                + F.mse_loss(redeg_patch_logits,  torch.zeros_like(redeg_patch_logits))
            )
            d_a_loss.backward()
            opt_DA.step()

            # ---- Update Generator G and F jointly ----
            opt_G.zero_grad()
            opt_F.zero_grad()

            # Adversarial: restored_real should fool D_B
            adv_g_logits = real_domain_discriminator(real_degraded, restored_real)
            adv_g_loss   = F.mse_loss(adv_g_logits, torch.ones_like(adv_g_logits))

            # Adversarial: re_degraded_synth should fool D_A
            adv_f_logits = synthetic_domain_discriminator(degraded_synth, re_degraded_synth)
            adv_f_loss   = F.mse_loss(adv_f_logits, torch.ones_like(adv_f_logits))

            # Cycle consistency for SYNTHETIC images: x_synth → restore → re-degrade ≈ x_synth
            synth_input_3ch = degraded_synth[:, :3, :, :]
            cycle_loss_synth = F.l1_loss(re_degraded_synth, synth_input_3ch)

            # Identity loss: if you feed a clean image into G, it should stay clean
            identity_loss = F.l1_loss(restored_synth, clean_synth)

            # Pixel-fidelity: keep G close to its Tier-1 pre-trained solution
            fov_mask = derive_fov_mask_from_input_tensor(degraded_synth)
            pixel_loss = F.l1_loss(
                restored_synth * fov_mask,
                clean_synth    * fov_mask
            )

            total_G_F_loss = (
                adv_g_loss
                + adv_f_loss
                + LAMBDA_CYCLE * cycle_loss_synth
                + 5.0 * identity_loss
                + 100.0 * pixel_loss   # pixel fidelity dominates — preserve Tier-1 learning
            )
            total_G_F_loss.backward()
            opt_G.step()
            opt_F.step()

            running_losses["G"]     += float(adv_g_loss)
            running_losses["F"]     += float(adv_f_loss)
            running_losses["D_A"]   += float(d_a_loss)
            running_losses["D_B"]   += float(d_b_loss)
            running_losses["cycle"] += float(cycle_loss_synth)

        avg = {k: v / steps_per_epoch for k, v in running_losses.items()}
        logger.info(
            f"DA Epoch {epoch+1}/{num_finetune_epochs} | "
            + " | ".join(f"{k}={v:.4f}" for k, v in avg.items())
        )

    return generator
