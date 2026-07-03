"""
src/models/generator.py
=======================
The U-Net generator with residual blocks and attention gates.

BEGINNER EXPLANATION:
  The generator is the MAIN AI that restores retina images.
  It works like a "camera that removes noise":
    - The ENCODER squeezes the image down to a small abstract representation
      (like summarising what's in the image)
    - The BOTTLENECK applies residual blocks to understand complex patterns
    - The DECODER expands it back to full size, using the original details
      via skip connections (like keeping notes from the encoder to fill in fine details)
    - ATTENTION GATES tell the decoder which skip-connection details matter
      (e.g., focus on blood vessels, not on noise artefacts)

ARCHITECTURE:
  Input:  (B, 4, 256, 256) — 4 channels (3 RGB + green copy)
  Output: (B, 3, 256, 256) — 3 channels (RGB restored image)
  Values: output is in range [-1, 1] (tanh activation)

  Encoder:  5 downsampling stages, channels: 4→64→128→256→512→512
  Bottleneck: 6 residual blocks at 512 channels
  Decoder: 5 upsampling stages with attention-gated skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    ATTENTION_GATE_ENABLED,
    BASE_FILTERS,
    INPUT_CHANNELS,
    NUM_RESIDUAL_BLOCKS,
    OUTPUT_CHANNELS,
)


# ---- Activation factory ------------------------------------

def build_activation_module(activation: str) -> nn.Module:
    """
    Return the requested activation function as an nn.Module.

    params: activation — one of "leaky_relu", "relu", "tanh", "none"
    returns: nn.Module
    side effects: none
    """
    if activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation: {activation}")


# ---- Atomic building block ---------------------------------

class ConvInstanceNormActivation(nn.Module):
    """
    Single Conv(orConvTranspose) → InstanceNorm → Activation block.

    This is the reusable building block for BOTH encoder (downsampling, transposed=False)
    and decoder (upsampling, transposed=True) stages.

    InstanceNorm normalises each feature map independently per sample,
    which works better than BatchNorm for style/texture tasks (like image restoration).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        activation: str = "leaky_relu",
        use_norm: bool = True,
        transposed: bool = False,
    ) -> None:
        super().__init__()

        conv_class = nn.ConvTranspose2d if transposed else nn.Conv2d
        self.conv = conv_class(
            in_channels, out_channels, kernel_size, stride, padding,
            bias=not use_norm,   # bias is redundant when InstanceNorm follows
        )
        self.norm       = nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity()
        self.activation = build_activation_module(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


# ---- Residual block ----------------------------------------

class ResidualBlock(nn.Module):
    """
    Residual block: y = x + F(x)
    Two 3×3 conv-norm-relu layers with an identity shortcut.

    The identity shortcut allows gradients to flow directly backwards
    through deep networks without vanishing, and lets the block
    learn to add small corrections rather than full transformations.

    Used in the bottleneck (the deepest, most abstract part of the network).
    """

    def __init__(self, channels: int, dropout_probability: float = 0.0) -> None:
        super().__init__()
        self.conv_a   = ConvInstanceNormActivation(
            channels, channels, kernel_size=3, stride=1, padding=1, activation="relu"
        )
        self.dropout  = nn.Dropout2d(dropout_probability) if dropout_probability > 0 else nn.Identity()
        self.conv_b   = ConvInstanceNormActivation(
            channels, channels, kernel_size=3, stride=1, padding=1, activation="none"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F(x) = conv_b(dropout(conv_a(x)))
        # Output = x + F(x)   ← the residual shortcut
        residual = self.conv_b(self.dropout(self.conv_a(x)))
        return x + residual


# ---- Attention gate ----------------------------------------

class AttentionGate(nn.Module):
    """
    Additive soft attention gate applied to encoder skip connections.

    Produces a per-pixel attention map α ∈ [0, 1] that up-weights
    spatially relevant parts of the skip features (e.g., vessel regions)
    and down-weights irrelevant or noisy parts (e.g., reflection artefacts).

    Mathematics (from Section 7.6 of the project plan):
        q  = ReLU( W_x(skip) + W_g(gating) )    ← combine signals
        α  = sigmoid( ψ(q) )                      ← per-pixel gate
        output = skip × α                          ← gated features
    """

    def __init__(
        self,
        skip_channels: int,
        gating_channels: int,
        intermediate_channels: int,
    ) -> None:
        super().__init__()
        self.w_x  = nn.Conv2d(skip_channels,    intermediate_channels, kernel_size=1, bias=True)
        self.w_g  = nn.Conv2d(gating_channels,  intermediate_channels, kernel_size=1, bias=True)
        self.psi  = nn.Conv2d(intermediate_channels, 1, kernel_size=1, bias=True)

    def forward(
        self,
        skip_features: torch.Tensor,
        gating_signal: torch.Tensor,
    ) -> torch.Tensor:
        """
        params:
            skip_features — encoder feature map (B, skip_channels, H, W)
            gating_signal — decoder signal at the same spatial size (B, gating_channels, H, W)
        returns: gated skip features — same shape as skip_features
        """
        # Upsample gating signal if spatial sizes don't match
        if gating_signal.shape[-2:] != skip_features.shape[-2:]:
            gating_signal = F.interpolate(
                gating_signal, size=skip_features.shape[-2:], mode="bilinear", align_corners=False
            )

        q     = F.relu(self.w_x(skip_features) + self.w_g(gating_signal))
        alpha = torch.sigmoid(self.psi(q))     # (B, 1, H, W)
        return skip_features * alpha           # broadcast across all channels


# ---- Full U-Net Generator -----------------------------------

class UNetGenerator(nn.Module):
    """
    The complete generator network.

    Architecture (Section 7.5 of the project plan):
      Encoder:    5 downsampling Conv-InstanceNorm-LeakyReLU blocks
                  channels: in_ch → F → 2F → 4F → 8F → 8F
                  spatial:  256 → 128 → 64 → 32 → 16 → 8

      Bottleneck: N residual blocks at 8F channels (spatial: 8×8)

      Decoder:    5 upsampling ConvTranspose-InstanceNorm-ReLU blocks
                  Skip connection at each level passes encoder features
                  (optionally gated by AttentionGate) to the corresponding
                  decoder stage (concatenated on channel axis)

      Output:     ConvTranspose to out_channels with tanh activation
                  (produces values in [-1, 1])

    F = base_filters = 64 by default
    N = num_residual_blocks = 6 by default
    """

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        out_channels: int = OUTPUT_CHANNELS,
        base_filters: int = BASE_FILTERS,
        num_residual_blocks: int = NUM_RESIDUAL_BLOCKS,
        use_attention: bool = ATTENTION_GATE_ENABLED,
    ) -> None:
        super().__init__()
        F = base_filters

        # ---- Encoder: 5 downsampling stages ----
        # First encoder has no InstanceNorm (common practice in Pix2Pix/DCGAN)
        self.encoder_1 = ConvInstanceNormActivation(in_channels, F,     use_norm=False, activation="leaky_relu")
        self.encoder_2 = ConvInstanceNormActivation(F,           F * 2, activation="leaky_relu")
        self.encoder_3 = ConvInstanceNormActivation(F * 2,       F * 4, activation="leaky_relu")
        self.encoder_4 = ConvInstanceNormActivation(F * 4,       F * 8, activation="leaky_relu")
        self.encoder_5 = ConvInstanceNormActivation(F * 8,       F * 8, activation="leaky_relu")

        # ---- Bottleneck: stack of residual blocks ----
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(F * 8, dropout_probability=0.15) for _ in range(num_residual_blocks)]
        )

        # ---- Attention gates (optional) ----
        if use_attention:
            self.attn_4 = AttentionGate(F * 8, F * 8, F * 4)
            self.attn_3 = AttentionGate(F * 4, F * 4, F * 2)
            self.attn_2 = AttentionGate(F * 2, F * 2, F)
            self.attn_1 = AttentionGate(F,     F,     F // 2)
        else:
            self.attn_4 = self.attn_3 = self.attn_2 = self.attn_1 = None

        # ---- Decoder: 5 upsampling stages ----
        # After each ConvTranspose, the corresponding (attention-gated) encoder
        # skip features are concatenated, doubling the channel count before
        # the next decoder block.  That's why decoder_2 takes F*16 = F*8 + F*8.
        self.decoder_1 = ConvInstanceNormActivation(F * 8,  F * 8, activation="relu", transposed=True)
        self.decoder_2 = ConvInstanceNormActivation(F * 16, F * 4, activation="relu", transposed=True)
        self.decoder_3 = ConvInstanceNormActivation(F * 8,  F * 2, activation="relu", transposed=True)
        self.decoder_4 = ConvInstanceNormActivation(F * 4,  F,     activation="relu", transposed=True)

        # Final output layer: tanh so output is in [-1, 1]
        self.output_layer = ConvInstanceNormActivation(
            F * 2, out_channels,
            kernel_size=4, stride=2, padding=1,
            activation="tanh", use_norm=False, transposed=True
        )

    def _maybe_gate(
        self,
        gate: AttentionGate,
        skip: torch.Tensor,
        decoder_out: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention gate if available; otherwise return skip unchanged."""
        return gate(skip, decoder_out) if gate is not None else skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: encode → bottleneck → attention-gated decode.

        params: x — (B, 4, 256, 256) degraded input tensor in [-1, 1]
        returns: restored image — (B, 3, 256, 256) in [-1, 1]
        side effects: none
        """
        # Encoder (save outputs for skip connections)
        e1 = self.encoder_1(x)     # (B, F,   H/2,  W/2)
        e2 = self.encoder_2(e1)    # (B, 2F,  H/4,  W/4)
        e3 = self.encoder_3(e2)    # (B, 4F,  H/8,  W/8)
        e4 = self.encoder_4(e3)    # (B, 8F,  H/16, W/16)
        e5 = self.encoder_5(e4)    # (B, 8F,  H/32, W/32)

        # Bottleneck
        b  = self.bottleneck(e5)   # (B, 8F,  H/32, W/32)

        # Decoder with attention-gated skip connections
        d1 = self.decoder_1(b)                                         # (B, 8F, H/16, W/16)
        d2 = self.decoder_2(torch.cat([d1, self._maybe_gate(self.attn_4, e4, d1)], dim=1))  # (B, 4F, H/8, W/8)
        d3 = self.decoder_3(torch.cat([d2, self._maybe_gate(self.attn_3, e3, d2)], dim=1))  # (B, 2F, H/4, W/4)
        d4 = self.decoder_4(torch.cat([d3, self._maybe_gate(self.attn_2, e2, d3)], dim=1))  # (B, F,  H/2, W/2)

        out = self.output_layer(torch.cat([d4, self._maybe_gate(self.attn_1, e1, d4)], dim=1))  # (B, 3, H, W)
        return out


# ---- Weight initialisation ---------------------------------

def initialize_network_weights(module: nn.Module) -> None:
    """
    Apply DCGAN-style weight initialisation to every Conv and InstanceNorm layer.

    Convolution weights: Normal distribution, mean=0, std=0.02
    InstanceNorm weights: Normal around 1.0 (scale), bias=0

    This is called ONCE after constructing each network via:
        generator.apply(initialize_network_weights)

    params: module — any nn.Module (applied recursively to the whole network tree)
    returns: None
    side effects: mutates weight/bias tensors in-place
    """
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.InstanceNorm2d) and module.affine:
        nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(module.bias.data, 0.0)
