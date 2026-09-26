"""DataLoader construction with reproducible shuffling and worker seeding."""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from defect_detection.utils.reproducibility import make_generator, seed_worker


def build_loader(
    dataset: Dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_generator(seed),
        persistent_workers=num_workers > 0,
    )
