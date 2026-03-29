from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .common import DEFAULT_DATASET_ROOT, DEFAULT_MEAN, DEFAULT_STD, build_dataloader, list_image_files, load_image_tensor


@dataclass
class CLEVRDatasetConfig:
    root: Path = field(default_factory=lambda: DEFAULT_DATASET_ROOT / "clevr")
    split: str = "train"
    image_size: int = 224
    max_samples: Optional[int] = None
    normalize: bool = False
    mean: Sequence[float] = DEFAULT_MEAN
    std: Sequence[float] = DEFAULT_STD


class CLEVRImageDataset(Dataset):
    def __init__(self, config: CLEVRDatasetConfig) -> None:
        self.config = config
        self.root = resolve_clevr_root(config.root)
        self.image_dir = self.root / "images" / config.split
        if not self.image_dir.exists():
            raise FileNotFoundError(f"CLEVR split directory not found: {self.image_dir}")

        self.image_paths = list_image_files(self.image_dir)
        if config.max_samples is not None:
            self.image_paths = self.image_paths[: config.max_samples]
        if len(self.image_paths) == 0:
            raise ValueError(f"No CLEVR images found under {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        image_path = self.image_paths[index]
        return {
            "images": load_image_tensor(
                image_path=image_path,
                image_size=self.config.image_size,
                normalize=self.config.normalize,
                mean=self.config.mean,
                std=self.config.std,
            ),
            "image_path": str(image_path),
            "image_index": torch.tensor(index, dtype=torch.long),
        }


def resolve_clevr_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = [root, root / "CLEVR_v1.0"]
    for candidate in candidates:
        if (candidate / "images").exists():
            return candidate
    raise FileNotFoundError(
        f"Unable to locate CLEVR images under {root}. Expected {root / 'images'} or {root / 'CLEVR_v1.0' / 'images'}."
    )


def build_clevr_dataloader(
    dataset_config: CLEVRDatasetConfig,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    return build_dataloader(
        dataset=CLEVRImageDataset(dataset_config),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
