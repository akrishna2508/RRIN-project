"""
src/models/discriminator.py
===========================
The PatchGAN discriminator — the "critic" that judges whether the
generator's output looks like a real clean retina image.

HOW IT WORKS (for beginners):
  Instead of judging the WHOLE image at once (which gives a very coarse
  signal), PatchGAN divides the image into overlapping ~70×70 pixel patches
  and independently decides "is this patch real or fake?" for each one.

  This gives the generator much richer feedback:
    "The top-left patch looks great, but the patch over the optic disc
     has an artefact that doesn't look natural."

  CONDITIONAL: The discriminator sees BOTH the degraded input AND the
  candidate output (concatenated along channels). So it's not asking
  "is this a realistic retina?" in the abstract — it's asking
  "given this degraded input, does this output look like a plausible restoration?"
  This prevents the generator from cheating by producing any generic
  pretty retina image that ignores the actual input.

SPECTRAL NORMALISATION:
  Divides each weight matrix by its largest singular value before use.
  This constrains the discriminator to be a Lipschitz function,
  which stabilises GAN training significantly.
"""

import torch
import torch.nn as nn
import torch.nn.utils

from src.config import BASE_FILTERS, INPUT_CHANNELS, OUTPUT_CHANNELS
from src.models.generator import ConvInstanceNormActivation


def apply_spectral_norm(module: nn.Module) -> nn.Module:
    """
    Wrap a module with spectral normalisation.
    Applied to every convolutional layer in the discriminator.

    params: module — any nn.Module (usually a ConvInstanceNormActivation)
    returns: the same module with spectral norm hooks attached
    side effects: modifies the module's weight handling in-place
    """
    # Apply spectral norm only to the internal conv layer inside the block
    if isinstance(module, ConvInstanceNormActivation):
        torch.nn.utils.spectral_norm(module.conv)
        return module
    return torch.nn.utils.spectral_norm(module)


class PatchGANDiscriminator(nn.Module):
    """
    5-layer PatchGAN discriminator.

    Input:  channel-concatenated (degraded_input, candidate_output)
            shape = (B, INPUT_CHANNELS + OUTPUT_CHANNELS, H, W) = (B, 7, 256, 256)

    Output: grid of real/fake logit scores
            shape = (B, 1, H', W') where each score covers a ~70×70 patch

    Each stride-2 layer halves the spatial size, giving us 5 downsampling
    operations before the final stride-1 head that produces the score grid.

    Spectral normalisation is applied to all conv layers for GAN stability.
    """

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS + OUTPUT_CHANNELS,
        base_filters: int = BASE_FILTERS,
    ) -> None:
        super().__init__()
        F = base_filters

        # Layer 1: No InstanceNorm (standard for the first discriminator layer)
        self.layer_1 = ConvInstanceNormActivation(
            input_channels, F, use_norm=False, activation="leaky_relu"
        )
        apply_spectral_norm(self.layer_1)

        # Layers 2–4: stride-2 downsampling
        self.layer_2 = ConvInstanceNormActivation(F,     F * 2, activation="leaky_relu")
        apply_spectral_norm(self.layer_2)

        self.layer_3 = ConvInstanceNormActivation(F * 2, F * 4, activation="leaky_relu")
        apply_spectral_norm(self.layer_3)

        self.layer_4 = ConvInstanceNormActivation(F * 4, F * 8, activation="leaky_relu")
        apply_spectral_norm(self.layer_4)

        # Layer 5: stride-1, no norm, no activation — produces raw logit scores
        self.layer_5 = nn.Conv2d(F * 8, 1, kernel_size=4, stride=1, padding=1)
        torch.nn.utils.spectral_norm(self.layer_5)

    def forward(
        self,
        degraded_input: torch.Tensor,
        candidate_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate a (degraded_input, candidate_output) pair.

        params:
            degraded_input   — (B, 4, H, W) in [-1, 1]  (the noisy input given to generator)
            candidate_output — (B, 3, H, W) in [-1, 1]  (the generator's output or real target)
        returns:
            patch_logits — (B, 1, H', W') unnormalised real/fake scores per patch

        The two tensors are concatenated along the channel axis before
        being fed into the first layer. This is what makes it "conditional":
        each patch score is conditioned on the corresponding input patch.
        side effects: none
        """
        combined = torch.cat([degraded_input, candidate_output], dim=1)  # (B, 7, H, W)
        x = self.layer_1(combined)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        return self.layer_5(x)   # raw logits, not probabilities
