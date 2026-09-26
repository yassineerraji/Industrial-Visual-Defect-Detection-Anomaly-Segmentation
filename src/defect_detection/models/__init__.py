"""Model construction from config."""

from __future__ import annotations

from typing import Any

from torch import nn

from defect_detection.models.autoencoder import ConvAutoencoder
from defect_detection.models.patchcore import PatchCore
from defect_detection.models.unet import ResNetUNet


def build_model(model_cfg: dict[str, Any]) -> nn.Module:
    params = {k: v for k, v in model_cfg.items() if k != "name"}
    name = model_cfg["name"]
    if name == "autoencoder":
        return ConvAutoencoder(**params)
    if name == "patchcore":
        return PatchCore(**params)
    if name == "unet":
        return ResNetUNet(**params)
    raise ValueError(f"Unknown model '{name}'")
