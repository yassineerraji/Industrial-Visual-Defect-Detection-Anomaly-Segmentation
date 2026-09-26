"""MVTec AD 2 sample indexing and PyTorch dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from defect_detection.data.preprocessing import PairTransform, load_image, load_mask
from defect_detection.utils.config import DatasetConfig

NORMAL = 0
ANOMALOUS = 1


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path | None  # None for normal samples
    label: int
    split: str


def list_images(directory: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions)


def expected_mask_path(cfg: DatasetConfig, split: str, image_path: Path) -> Path:
    mask_name = f"{image_path.stem}{cfg.layout.mask_suffix}{image_path.suffix}"
    return cfg.split_dir(split) / cfg.layout.mask_dir / mask_name


def index_split(cfg: DatasetConfig, split: str) -> list[Sample]:
    """List all samples of a split, sorted deterministically (normal first).

    Raises if the split directory is missing, holds no images, or any
    anomalous image lacks its ground-truth mask.
    """
    split_dir = cfg.split_dir(split)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    exts = cfg.layout.image_extensions
    samples = [
        Sample(p, None, NORMAL, split) for p in list_images(split_dir / cfg.layout.good_dir, exts)
    ]
    missing: list[Path] = []
    for p in list_images(split_dir / cfg.layout.bad_dir, exts):
        mask = expected_mask_path(cfg, split, p)
        if not mask.is_file():
            missing.append(mask)
        samples.append(Sample(p, mask, ANOMALOUS, split))

    if missing:
        raise FileNotFoundError(f"{len(missing)} ground-truth masks missing in '{split}', e.g. {missing[0]}")
    if not samples:
        raise FileNotFoundError(f"No images found under {split_dir}")
    return samples


class MVTecAD2Dataset(Dataset):
    """Yields dicts with ``image`` (3,H,W), ``mask`` (1,H,W), ``label`` and ``path``."""

    def __init__(self, samples: list[Sample], transform: PairTransform) -> None:
        self.samples = samples
        self.transform = transform

    @classmethod
    def from_config(cls, cfg: DatasetConfig, split: str, transform: PairTransform) -> "MVTecAD2Dataset":
        return cls(index_split(cfg, split), transform)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = load_image(sample.image_path)
        mask = load_mask(sample.mask_path, image.size)
        image_t, mask_t = self.transform(image, mask)
        return {"image": image_t, "mask": mask_t, "label": sample.label, "path": str(sample.image_path)}
