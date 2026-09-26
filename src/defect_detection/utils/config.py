"""YAML configuration loading into typed, immutable dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DatasetLayout:
    splits: dict[str, str]
    good_dir: str = "good"
    bad_dir: str = "bad"
    mask_dir: str = "ground_truth/bad"
    mask_suffix: str = "_mask"
    image_extensions: tuple[str, ...] = (".png",)


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    category: str
    layout: DatasetLayout

    @property
    def category_dir(self) -> Path:
        return self.root / self.category

    def split_dir(self, split: str) -> Path:
        if split not in self.layout.splits:
            raise KeyError(f"Unknown split '{split}'. Configured: {list(self.layout.splits)}")
        return self.category_dir / self.layout.splits[split]


def _as_hw(size: int | list[int] | tuple[int, ...]) -> tuple[int, int]:
    """Normalise an int (square) or [height, width] to a (height, width) tuple."""
    if isinstance(size, int):
        return (size, size)
    if len(size) != 2:
        raise ValueError(f"Size must be an int or [height, width], got {size}")
    return (int(size[0]), int(size[1]))


@dataclass(frozen=True)
class PreprocessingConfig:
    """``image_size`` / ``crop_size`` are (height, width); an int means square."""

    image_size: tuple[int, int] = (256, 256)
    crop_size: tuple[int, int] | None = None
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_size", _as_hw(self.image_size))
        if self.crop_size is not None:
            object.__setattr__(self, "crop_size", _as_hw(self.crop_size))
            if any(c > s for c, s in zip(self.crop_size, self.image_size)):
                raise ValueError("crop_size must not exceed image_size")

    @property
    def output_size(self) -> tuple[int, int]:
        return self.crop_size or self.image_size


@dataclass(frozen=True)
class AugmentationConfig:
    hflip_p: float = 0.0
    vflip_p: float = 0.0
    rotation_degrees: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    blur_sigma: tuple[float, float] | None = None
    noise_std: float = 0.0


@dataclass(frozen=True)
class DataConfig:
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    seed: int = 42


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping at the top level.")
    return data


def parse_data_config(raw: dict[str, Any], data_root: str | Path | None = None) -> DataConfig:
    """Build a :class:`DataConfig` from a raw config mapping.

    ``data_root`` overrides ``dataset.root`` (e.g. from a ``--data-root`` CLI flag).
    """
    ds = raw["dataset"]
    layout_raw = dict(ds["layout"])
    layout_raw["image_extensions"] = tuple(e.lower() for e in layout_raw.get("image_extensions", [".png"]))
    dataset = DatasetConfig(
        root=Path(data_root if data_root is not None else ds["root"]).expanduser(),
        category=ds["category"],
        layout=DatasetLayout(**layout_raw),
    )

    pre_raw = dict(raw.get("preprocessing") or {})
    for key in ("mean", "std"):
        if key in pre_raw:
            pre_raw[key] = tuple(pre_raw[key])

    aug_raw = dict(raw.get("augmentation") or {})
    if aug_raw.get("blur_sigma") is not None:
        aug_raw["blur_sigma"] = tuple(aug_raw["blur_sigma"])

    return DataConfig(
        dataset=dataset,
        preprocessing=PreprocessingConfig(**pre_raw),
        augmentation=AugmentationConfig(**aug_raw),
        seed=int(raw.get("seed", 42)),
    )


def load_data_config(path: str | Path, data_root: str | Path | None = None) -> DataConfig:
    return parse_data_config(load_yaml(path), data_root=data_root)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config into a single self-contained mapping.

    An experiment config names a shared data config (``data_config``, relative
    to the experiment file) and may override parts of it under ``data``. The
    result holds the fully merged data section under ``data``, so it can be
    saved alongside a run and re-parsed without the original files.
    """
    path = Path(path)
    raw = load_yaml(path)
    data_raw: dict[str, Any] = {}
    if "data_config" in raw:
        data_raw = load_yaml(path.parent / raw.pop("data_config"))
    raw["data"] = deep_merge(data_raw, raw.get("data") or {})
    return raw
