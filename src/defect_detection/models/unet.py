"""Lightweight U-Net with a pretrained ResNet encoder for supervised segmentation (Regime C).

Outputs per-pixel defect logits. ``anomaly_map`` returns the sigmoid of the
logits: a model-estimated defect probability, not a calibrated one.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import get_model


def _decoder_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    """ResNet-18/34 encoder (strides 2..32) with a skip-connected upsampling decoder.

    Input height and width must be divisible by 32.
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, decoder_channels: int = 64) -> None:
        super().__init__()
        if backbone not in ("resnet18", "resnet34"):
            raise ValueError("backbone must be resnet18 or resnet34")
        net = get_model(backbone, weights="DEFAULT" if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)  # stride 2, 64 ch
        self.pool = net.maxpool
        self.encoder = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])  # 64,128,256,512

        d = decoder_channels
        self.up4 = _decoder_block(512 + 256, 4 * d)
        self.up3 = _decoder_block(4 * d + 128, 2 * d)
        self.up2 = _decoder_block(2 * d + 64, d)
        self.up1 = _decoder_block(d + 64, d)
        self.up0 = _decoder_block(d, d // 2)
        self.head = nn.Conv2d(d // 2, 1, kernel_size=1)

    @staticmethod
    def _up_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 32 or x.shape[-1] % 32:
            raise ValueError(f"Input size {tuple(x.shape[-2:])} must be divisible by 32")
        s0 = self.stem(x)  # /2
        e1 = self.encoder[0](self.pool(s0))  # /4
        e2 = self.encoder[1](e1)  # /8
        e3 = self.encoder[2](e2)  # /16
        e4 = self.encoder[3](e3)  # /32
        d = self.up4(self._up_cat(e4, e3))
        d = self.up3(self._up_cat(d, e2))
        d = self.up2(self._up_cat(d, e1))
        d = self.up1(self._up_cat(d, s0))
        d = self.up0(F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.head(d)

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(x))
