"""Convolutional autoencoder for reconstruction-based anomaly detection (Regime A).

Trained on normal images only. Inputs are expected in [0, 1] (no mean/std
normalisation) so the sigmoid output lives in the same space.
"""

from __future__ import annotations

import torch
from torch import nn

MAX_CHANNELS = 256


def _conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class ConvAutoencoder(nn.Module):
    """Symmetric conv encoder/decoder with a spatial latent bottleneck.

    Each of ``num_downsamples`` stages halves the resolution, so input height
    and width must be divisible by ``2 ** num_downsamples``. The latent tensor
    has shape (``latent_channels``, H / 2**n, W / 2**n); ``latent_channels``
    is the main capacity knob (too large and anomalies are reconstructed too).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_downsamples: int = 4,
        latent_channels: int = 32,
    ) -> None:
        super().__init__()
        self.num_downsamples = num_downsamples
        widths = [min(base_channels * 2**i, MAX_CHANNELS) for i in range(num_downsamples)]

        encoder: list[nn.Module] = []
        prev = in_channels
        for w in widths:
            encoder += [_conv_block(prev, w, stride=2), _conv_block(w, w)]
            prev = w
        encoder.append(nn.Conv2d(prev, latent_channels, kernel_size=1))
        self.encoder = nn.Sequential(*encoder)

        decoder: list[nn.Module] = [_conv_block(latent_channels, widths[-1])]
        for i in reversed(range(num_downsamples)):
            out = widths[i - 1] if i > 0 else base_channels
            decoder += [nn.Upsample(scale_factor=2, mode="nearest"), _conv_block(widths[i], out), _conv_block(out, out)]
        decoder += [nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        factor = 2**self.num_downsamples
        if x.shape[-2] % factor or x.shape[-1] % factor:
            raise ValueError(f"Input size {tuple(x.shape[-2:])} must be divisible by {factor}")
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """Per-pixel squared reconstruction error averaged over channels, shape (B, 1, H, W)."""
        return (self(x) - x).pow(2).mean(dim=1, keepdim=True)
