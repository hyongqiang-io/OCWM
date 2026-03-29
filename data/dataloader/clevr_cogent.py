from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .common import DEFAULT_DATASET_ROOT, DEFAULT_MEAN, DEFAULT_STD, build_dataloader, list_image_files, load_image_tensor


_SPLIT_ALIASES = {
    "train": "trainA",
    "val": "valA",
    "valid": "valA",
    "test": "testA",
}


@dataclass
class CLEVRCoGenTDatasetConfig:
    root: Path = field(default_factory=lambda: DEFAULT_DATASET_ROOT / "clevr_cogent")
    split: str = "trainA"
    image_size: int = 224
    max_samples: Optional[int] = None
    normalize: bool = False
    mean: Sequence[float] = DEFAULT_MEAN
    std: Sequence[float] = DEFAULT_STD


class CLEVRCoGenTImageDataset(Dataset):
    def __init__(self, config: CLEVRCoGenTDatasetConfig) -> None:
        self.config = config
        self.root = resolve_clevr_cogent_root(config.root)
        split_name = resolve_clevr_cogent_split(self.root, config.split)
        self.image_dir = self.root / "images" / split_name
        self.image_paths = list_image_files(self.image_dir)
        if config.max_samples is not None:
            self.image_paths = self.image_paths[: config.max_samples]
        if len(self.image_paths) == 0:
            raise ValueError(f"No CLEVR-CoGenT images found under {self.image_dir}")
        self.split_name = split_name

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
            "split": self.split_name,
        }


def resolve_clevr_cogent_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = [root, root / "CLEVR_CoGenT_v1.0"]
    for candidate in candidates:
        if (candidate / "images").exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate CLEVR-CoGenT images under "
        f"{root}. Expected {root / 'images'} or {root / 'CLEVR_CoGenT_v1.0' / 'images'}."
    )


def resolve_clevr_cogent_split(root: Path, split: str) -> str:
    split_candidates = [split]
    alias = _SPLIT_ALIASES.get(split)
    if alias is not None:
        split_candidates.append(alias)

    for candidate in split_candidates:
        if (root / "images" / candidate).exists():
            return candidate

    raise FileNotFoundError(f"Unable to locate CLEVR-CoGenT split '{split}' under {root / 'images'}")


def build_clevr_cogent_dataloader(
    dataset_config: CLEVRCoGenTDatasetConfig,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    return build_dataloader(
        dataset=CLEVRCoGenTImageDataset(dataset_config),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
