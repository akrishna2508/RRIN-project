"""
src/training/early_stopping.py
src/training/schedulers.py
src/training/checkpoints.py
(Combined into one file for brevity — split as you see fit)

EARLY STOPPING (for beginners):
  If the model stops improving on the validation set for 15 epochs in a row,
  training stops automatically. This prevents wasting time on a model that
  is already overfitting (memorising training data instead of learning).

LR SCHEDULER:
  The learning rate controls how big each update step is.
  Too high → unstable training; too low → very slow.
  We keep it constant for the first 100 epochs, then linearly reduce
  it to zero over the next 100 epochs (standard Pix2Pix schedule).

CHECKPOINTS:
  A checkpoint = snapshot of the model weights saved to disk.
  If training crashes, you can resume from the last checkpoint.
  The "best.pt" checkpoint is always the one with the highest
  validation SSIM seen so far.
"""

import os
import logging
from typing import Optional

import torch
import torch.optim

from src.config import EARLY_STOPPING_PATIENCE, LR_CONSTANT_EPOCHS, LR_DECAY_EPOCHS


# ---- Early Stopping ----------------------------------------

class EarlyStopping:
    """
    Tracks the best validation metric and counts epochs without improvement.
    Signals when training should stop early.
    """

    def __init__(
        self,
        patience: int = EARLY_STOPPING_PATIENCE,
        metric_should_increase: bool = True,
    ) -> None:
        """
        params:
            patience               — max consecutive epochs without improvement
            metric_should_increase — True for SSIM/PSNR (higher=better),
                                     False for loss values (lower=better)
        """
        self.patience               = patience
        self.metric_should_increase = metric_should_increase
        self.best_value             = -float("inf") if metric_should_increase else float("inf")
        self.epochs_without_improvement = 0

    def step(self, current_value: float) -> tuple[bool, bool]:
        """
        Update state with the current epoch's validation metric.

        params: current_value — validation SSIM (or PSNR, or loss)
        returns:
            should_stop — True if patience has been exceeded → stop training
            is_new_best — True if this epoch is a new best → save checkpoint

        side effects: mutates self.best_value and self.epochs_without_improvement
        """
        if self.metric_should_increase:
            improved = current_value > self.best_value
        else:
            improved = current_value < self.best_value

        if improved:
            self.best_value = current_value
            self.epochs_without_improvement = 0
            return False, True    # don't stop; IS a new best

        self.epochs_without_improvement += 1
        should_stop = (self.epochs_without_improvement >= self.patience)
        return should_stop, False


# ---- Learning Rate Scheduler ------------------------------

def get_linear_decay_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    constant_epochs: int = LR_CONSTANT_EPOCHS,
    decay_epochs: int = LR_DECAY_EPOCHS,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Build a LambdaLR scheduler that:
      - Keeps the LR constant for the first `constant_epochs` epochs
      - Linearly decays it to zero over the next `decay_epochs` epochs

    This is the standard Pix2Pix / CycleGAN learning-rate schedule.

    params:
        optimizer       — the Adam optimiser to schedule
        constant_epochs — how many epochs to hold LR at initial value
        decay_epochs    — how many epochs to spend linearly decaying to 0
    returns: LambdaLR scheduler (call scheduler.step() once per epoch)
    side effects: none beyond creating the scheduler
    """
    def lr_lambda(epoch_index: int) -> float:
        if epoch_index < constant_epochs:
            return 1.0
        progress = (epoch_index - constant_epochs) / max(1, decay_epochs)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---- Checkpoint Save / Load --------------------------------

def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    validation_metrics: dict,
    is_best: bool,
) -> None:
    """
    Save the complete training state to disk.

    WHY SAVE OPTIMIZER STATE? (for beginners)
      Adam maintains internal momentum variables (m_t, v_t) per parameter.
      If you save only the model weights and resume training, Adam loses
      those accumulated statistics and training becomes unstable for a few
      epochs. Saving the optimizer state fixes this.

    params:
        checkpoint_dir      — folder where checkpoints are saved
        epoch               — current epoch number (used in filename)
        generator           — UNetGenerator module
        discriminator       — PatchGANDiscriminator module
        generator_optimizer — Adam optimizer for generator
        discriminator_optimizer — Adam optimizer for discriminator
        validation_metrics  — dict with 'psnr' and 'ssim' values
        is_best             — if True, also save as "best.pt"
    side effects: writes one or two .pt files to checkpoint_dir
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    payload = {
        "epoch":                       epoch,
        "generator_state":             generator.state_dict(),
        "discriminator_state":         discriminator.state_dict(),
        "generator_optimizer_state":   generator_optimizer.state_dict(),
        "discriminator_optimizer_state": discriminator_optimizer.state_dict(),
        "validation_metrics":          validation_metrics,
    }

    # Save epoch-specific checkpoint
    epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pt")
    torch.save(payload, epoch_path)

    # Overwrite best.pt if this is a new best
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        torch.save(payload, best_path)
        print(f"  ★ New best checkpoint saved → {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    generator_optimizer: Optional[torch.optim.Optimizer] = None,
    discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[int, dict]:
    """
    Restore model (and optionally optimizer) state from a checkpoint file.

    params:
        checkpoint_path         — path to the .pt checkpoint file
        generator               — UNetGenerator (weights restored in-place)
        discriminator           — PatchGANDiscriminator (weights restored in-place)
        generator_optimizer     — if provided, optimizer state is also restored
        discriminator_optimizer — same for discriminator optimizer
    returns:
        starting_epoch     — epoch number stored in the checkpoint
        validation_metrics — metrics dict stored in the checkpoint
    side effects: mutates model and optimizer weight tensors in-place
    """
    from src.config import DEVICE

    print(f"Loading checkpoint: {checkpoint_path}")
    # FIXED: weights_only=False is required for torch >= 2.6, where the default
    # flipped to True. Our checkpoint payload is a dict containing optimizer
    # states and a metrics dict (not just tensors), so it must load in full.
    # These checkpoints are produced by this same trusted codebase.
    payload = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    generator.load_state_dict(payload["generator_state"])
    discriminator.load_state_dict(payload["discriminator_state"])

    if generator_optimizer is not None and "generator_optimizer_state" in payload:
        generator_optimizer.load_state_dict(payload["generator_optimizer_state"])

    if discriminator_optimizer is not None and "discriminator_optimizer_state" in payload:
        discriminator_optimizer.load_state_dict(payload["discriminator_optimizer_state"])

    epoch   = payload.get("epoch", 0)
    metrics = payload.get("validation_metrics", {})
    print(f"  Resumed from epoch {epoch}, metrics: {metrics}")
    return epoch, metrics
