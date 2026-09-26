"""Compute-device selection: MPS -> CUDA -> CPU."""

from __future__ import annotations

import torch

from defect_detection.utils.logging import get_logger

logger = get_logger(__name__)

_VALID = ("auto", "mps", "cuda", "cpu")


def get_device(preference: str = "auto") -> torch.device:
    """Return the requested device, or the best available one for ``"auto"``.

    Raises ``RuntimeError`` if an explicitly requested accelerator is unavailable,
    rather than silently falling back.
    """
    preference = preference.lower()
    if preference not in _VALID:
        raise ValueError(f"Unknown device '{preference}'. Expected one of {_VALID}.")

    if preference == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        available = {
            "mps": torch.backends.mps.is_available(),
            "cuda": torch.cuda.is_available(),
            "cpu": True,
        }[preference]
        if not available:
            raise RuntimeError(f"Requested device '{preference}' is not available.")
        device = torch.device(preference)

    logger.info("Using device: %s", device)
    return device
