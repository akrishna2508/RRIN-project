"""
src/models/losses.py
====================
All loss functions used to train the generator.

WHAT IS A LOSS FUNCTION? (for beginners)
  A loss function measures "how wrong" the network's output is.
  The lower the loss, the better the restoration.
  Training = repeatedly tweaking the network to reduce the loss.

FOUR LOSSES COMBINED:
  1. L1 (pixel accuracy)    — are pixel values close to the target?
  2. SSIM (structure)       — do edges, contrast, local patterns match?
  3. Perceptual (texture)   — do deep VGG features match? (human-like quality)
  4. Adversarial (realism)  — does it fool the discriminator?

Weights (from config): L1=100, SSIM=10, Perceptual=10, Adversarial=1
L1 dominates to keep the network faithful to the specific input.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from src.config import (
    LAMBDA_ADV, LAMBDA_L1, LAMBDA_PERCEPTUAL, LAMBDA_SSIM,
    LABEL_SMOOTHING_REAL_TARGET, VGG_PERCEPTUAL_LAYERS,
)
from src.utils.image_utils import rescale_to_imagenet_normalization


# ---- VGG Perceptual Loss -----------------------------------

def _build_vgg_layer_name_to_index_map() -> dict[str, int]:
    """
    Map named VGG16 layers (relu2_2, relu3_3, etc.) to their integer
    index in the vgg16.features sequential container.

    VGG16 features layout (numbered 0-based):
      0  Conv2d   (3→64)    1  ReLU  ← relu1_1
      2  Conv2d   (64→64)   3  ReLU  ← relu1_2
      4  MaxPool2d
      5  Conv2d   (64→128)  6  ReLU  ← relu2_1
      7  Conv2d  (128→128)  8  ReLU  ← relu2_2  ★
      9  MaxPool2d
     10  Conv2d  (128→256) 11  ReLU  ← relu3_1
     12  Conv2d  (256→256) 13  ReLU  ← relu3_2
     14  Conv2d  (256→256) 15  ReLU  ← relu3_3  ★
     16  MaxPool2d
     ...
    """
    return {
        "relu1_1": 1,
        "relu1_2": 3,
        "relu2_1": 6,
        "relu2_2": 8,   # ← used by default
        "relu3_1": 11,
        "relu3_2": 13,
        "relu3_3": 15,  # ← used by default
        "relu4_1": 18,
        "relu4_2": 20,
        "relu4_3": 22,
    }


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using intermediate activations from a frozen VGG16.

    INTUITION: Two images can have very different pixel values yet look
    identical to a human (e.g., a 1-pixel shift of a sharp edge).
    VGG features are much more aligned with human perception — they
    respond to "vessel-like pattern" rather than "exact pixel at (x, y)".

    The VGG16 was trained on ImageNet (millions of natural images).
    Its filters already encode useful texture and edge statistics
    without any retina-specific training.

    Parameters are FROZEN — VGG never updates during our training.
    """

    def __init__(self, layer_names: list = None) -> None:
        super().__init__()
        layer_names = layer_names or VGG_PERCEPTUAL_LAYERS

        # Load pre-trained VGG16, freeze all its weights
        # FIXED: wrapped in try/except so a blocked/offline weights download
        # (e.g. HTTP 403 on some mirrors, or no internet in a committed run)
        # degrades gracefully to un-pretrained VGG instead of hard-crashing
        # the entire training run. Perceptual loss still functions with
        # randomly-initialised features (weaker, but training proceeds).
        try:
            vgg_features = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features
        except Exception as vgg_err:
            print(f"  WARNING: could not download pretrained VGG16 weights ({vgg_err}). "
                  f"Falling back to un-pretrained VGG for perceptual loss.")
            vgg_features = tvm.vgg16(weights=None).features
        self.vgg_layers = vgg_features.eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.layer_names      = layer_names
        self.layer_index_map  = _build_vgg_layer_name_to_index_map()
        self._target_indices  = {
            self.layer_index_map[name]
            for name in layer_names
            if name in self.layer_index_map
        }

    def _extract_named_activations(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Run x through VGG and collect activations at the requested layers.

        params: x — (B, 3, H, W) in ImageNet-normalised space
        returns: dict: layer_name → activation tensor
        side effects: none (VGG weights are frozen)
        """
        activations = {}
        current = x
        for idx, layer in enumerate(self.vgg_layers):
            current = layer(current)
            if idx in self._target_indices:
                # Find the name for this index
                name = next(k for k, v in self.layer_index_map.items() if v == idx)
                if name in self.layer_names:
                    activations[name] = current
            if len(activations) == len(self.layer_names):
                break   # Stop early once we have all needed activations
        return activations

    def forward(
        self,
        candidate_output: torch.Tensor,
        target_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the summed, per-layer MSE between VGG activations.

        params:
            candidate_output — (B, 3, H, W) generator output in [-1, 1]
            target_output    — (B, 3, H, W) clean target in [-1, 1]
        returns: perceptual_loss — scalar tensor
        side effects: none
        """
        # Rescale from [-1, 1] to ImageNet-normalised space before VGG
        cand_norm   = rescale_to_imagenet_normalization(candidate_output)
        target_norm = rescale_to_imagenet_normalization(target_output)

        cand_acts = self._extract_named_activations(cand_norm)

        # Target activations don't need gradients (we're not updating VGG or target)
        with torch.no_grad():
            target_acts = self._extract_named_activations(target_norm)

        total_loss = sum(
            F.mse_loss(cand_acts[name], target_acts[name])
            for name in self.layer_names
            if name in cand_acts and name in target_acts
        )
        return total_loss


# ---- SSIM Loss --------------------------------------------

def _build_gaussian_window(window_size: int, sigma: float = 1.5) -> torch.Tensor:
    """
    Build a 1D Gaussian kernel and outer-product it into a 2D window.
    The window is used for computing local statistics in the SSIM formula.

    params: window_size — side length in pixels (typically 11)
            sigma       — Gaussian standard deviation
    returns: (1, 1, window_size, window_size) float tensor
    side effects: none
    """
    half = window_size // 2
    coords = torch.arange(window_size, dtype=torch.float32) - half
    gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()

    # Outer product → 2D Gaussian window
    gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)   # (W, W)
    return gauss_2d.unsqueeze(0).unsqueeze(0)                    # (1, 1, W, W)


def _compute_windowed_ssim_map(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    dynamic_range: float = 2.0,
) -> torch.Tensor:
    """
    Compute the per-window SSIM map between tensors x and y.

    SSIM formula (Wang et al. 2004):
        SSIM(x, y) = (2μxμy + C1)(2σxy + C2) / [(μx² + μy² + C1)(σx² + σy² + C2)]

    Where μ and σ are local (windowed) mean and variance/covariance.

    Dynamic range L=2 because our tensors are in [-1, 1] range.
    C1 = (k1·L)²  with k1=0.01 → C1 = (0.02)² = 0.0004
    C2 = (k2·L)²  with k2=0.03 → C2 = (0.06)² = 0.0036

    params:
        x, y          — (B, C, H, W) image tensors
        window        — (1, 1, ws, ws) Gaussian window
        dynamic_range — value range, 2.0 for [-1, 1] tensors
    returns: SSIM map — (B, C, H', W') values in [-1, 1], 1 = perfect match
    side effects: none
    """
    C1 = (0.01 * dynamic_range) ** 2
    C2 = (0.03 * dynamic_range) ** 2

    B, C, H, W = x.shape
    ws = window.shape[-1]
    pad = ws // 2

    # Apply grouped convolution so each channel is processed independently
    win = window.repeat(C, 1, 1, 1).to(x.device)   # (C, 1, ws, ws)

    mu_x  = F.conv2d(x, win, padding=pad, groups=C)
    mu_y  = F.conv2d(y, win, padding=pad, groups=C)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy   = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, win, padding=pad, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, win, padding=pad, groups=C) - mu_y_sq
    sigma_xy   = F.conv2d(x * y, win, padding=pad, groups=C) - mu_xy

    numerator   = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    return numerator / (denominator + 1e-8)


class SSIMLoss(nn.Module):
    """
    SSIM-based loss: L_SSIM = 1 - mean(SSIM(candidate, target))

    SSIM=1 means perfect structural match.
    SSIM=0 means completely different structure.
    Loss=0 is perfect; loss=1 is worst.

    Unlike pixel L1, SSIM also penalises:
      - Contrast differences (σ comparison)
      - Structural misalignment (σxy covariance comparison)
    This makes it sensitive to whether vessel patterns and edges
    are in the right place, not just the right average brightness.
    """

    def __init__(self, window_size: int = 11) -> None:
        super().__init__()
        self.window_size = window_size
        # Register as a buffer so it moves to the correct device with .to(device)
        self.register_buffer(
            "gaussian_window",
            _build_gaussian_window(window_size)
        )

    def forward(
        self,
        candidate_output: torch.Tensor,
        target_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        params: candidate, target — (B, 3, H, W) in [-1, 1]
        returns: ssim_loss — scalar tensor in [0, 1]
        """
        ssim_map = _compute_windowed_ssim_map(
            candidate_output, target_output,
            self.gaussian_window, dynamic_range=2.0
        )
        return 1.0 - ssim_map.mean()


# ---- LSGAN Discriminator Loss ------------------------------

def compute_discriminator_loss(
    discriminator: nn.Module,
    degraded_input: torch.Tensor,
    real_target: torch.Tensor,
    fake_output_detached: torch.Tensor,
) -> torch.Tensor:
    """
    Compute LSGAN (Least-Squares GAN) discriminator loss.

    LSGAN replaces the original log-loss with squared error, keeping
    gradients informative even when the discriminator is very confident:

        L_D = 0.5 * E[(D(X, Y) - 0.9)²]   ← real pairs → target 0.9 (label smoothing)
            + 0.5 * E[(D(X, G(X)))²]        ← fake pairs → target 0.0

    Label smoothing on real targets (0.9 not 1.0) prevents the discriminator
    from becoming overconfident, which would kill gradient flow to the generator.

    IMPORTANT: fake_output must be DETACHED from the generator's computation
    graph here. We're updating the discriminator only — gradients must not
    flow back into the generator in this step.

    params:
        discriminator        — PatchGANDiscriminator
        degraded_input       — (B, 4, H, W)
        real_target          — (B, 3, H, W) ground-truth clean image
        fake_output_detached — (B, 3, H, W) generator output, .detach() already called
    returns: discriminator_loss — scalar tensor
    side effects: none
    """
    real_logits = discriminator(degraded_input, real_target)
    fake_logits = discriminator(degraded_input, fake_output_detached)

    real_label  = torch.full_like(real_logits, LABEL_SMOOTHING_REAL_TARGET)
    real_loss   = F.mse_loss(real_logits, real_label)
    fake_loss   = F.mse_loss(fake_logits, torch.zeros_like(fake_logits))

    return 0.5 * (real_loss + fake_loss)


# ---- Composite Generator Loss ------------------------------

def compute_generator_loss(
    discriminator: nn.Module,
    perceptual_loss_module: VGGPerceptualLoss,
    ssim_loss_module: SSIMLoss,
    degraded_input: torch.Tensor,
    fake_output: torch.Tensor,
    real_target: torch.Tensor,
    fov_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the combined generator loss (Section 8.6 of the project plan):

        L_G = λ_adv * L_adv  +  λ_L1 * L_L1  +  λ_ssim * L_SSIM  +  λ_perc * L_perc

    The L1 term is computed only over FOV (fundus disc) pixels by masking.
    Pixels outside the circular disc are black background — penalising
    errors there would confuse the network.

    NOTE: fake_output must NOT be detached here. Gradients must flow
    all the way from the discriminator's judgment back through the
    generator's parameters to train it.

    params:
        discriminator        — PatchGANDiscriminator
        perceptual_loss_module — VGGPerceptualLoss
        ssim_loss_module     — SSIMLoss
        degraded_input       — (B, 4, H, W)
        fake_output          — (B, 3, H, W)  NOT detached
        real_target          — (B, 3, H, W)
        fov_mask             — (B, 1, H, W) float tensor of 0/1 values
    returns:
        total_loss  — scalar tensor (for .backward())
        loss_dict   — {name: float} for logging
    side effects: builds computation graph (gradients computed on .backward())
    """
    # Adversarial: fool the discriminator into scoring fake as "real" (target=1)
    fake_logits      = discriminator(degraded_input, fake_output)
    adversarial_loss = F.mse_loss(fake_logits, torch.ones_like(fake_logits))

    # Masked L1: measure pixel accuracy only inside the fundus disc
    masked_l1_loss   = F.l1_loss(
        fake_output * fov_mask,
        real_target * fov_mask
    )

    # SSIM structural similarity
    ssim_loss_value  = ssim_loss_module(fake_output, real_target)

    # VGG perceptual (deep feature matching)
    perceptual_value = perceptual_loss_module(fake_output, real_target)

    total = (
        LAMBDA_ADV        * adversarial_loss
        + LAMBDA_L1       * masked_l1_loss
        + LAMBDA_SSIM     * ssim_loss_value
        + LAMBDA_PERCEPTUAL * perceptual_value
    )

    loss_dict = {
        "adversarial": float(adversarial_loss),
        "l1":          float(masked_l1_loss),
        "ssim":        float(ssim_loss_value),
        "perceptual":  float(perceptual_value),
        "total":       float(total),
    }
    return total, loss_dict
