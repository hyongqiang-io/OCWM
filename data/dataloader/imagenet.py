from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .common import DEFAULT_DATASET_ROOT, DEFAULT_MEAN, DEFAULT_STD, build_dataloader, list_image_files, load_image_tensor, resolve_existing_path


@dataclass
class ImageNetDatasetConfig:
    root: Path = field(default_factory=lambda: DEFAULT_DATASET_ROOT / "imagenet")
    split: str = "train"
    image_size: int = 224
    max_samples: Optional[int] = None
    normalize: bool = False
    mean: Sequence[float] = DEFAULT_MEAN
    std: Sequence[float] = DEFAULT_STD


class ImageNetImageDataset(Dataset):
    def __init__(self, config: ImageNetDatasetConfig) -> None:
        self.config = config
        self.root = resolve_imagenet_root(config.root)
        self.split_dir = resolve_imagenet_split_dir(self.root, config.split)
        self.image_paths = list_image_files(self.split_dir, recursive=True)
        if config.max_samples is not None:
            self.image_paths = self.image_paths[: config.max_samples]
        if len(self.image_paths) == 0:
            raise ValueError(f"No ImageNet images found under {self.split_dir}")

        class_names = sorted(
            {
                relative.parts[0]
                for relative in (image_path.relative_to(self.split_dir) for image_path in self.image_paths)
                if len(relative.parts) > 1
            }
        )
        self.class_to_index = {class_name: index for index, class_name in enumerate(class_names)}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        image_path = self.image_paths[index]
        relative = image_path.relative_to(self.split_dir)
        class_name = relative.parts[0] if len(relative.parts) > 1 else ""
        class_index = self.class_to_index.get(class_name, -1)
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
            "class_index": torch.tensor(class_index, dtype=torch.long),
            "class_name": class_name,
        }


def resolve_imagenet_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = [root, root / "ILSVRC2012", root / "ImageNet"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate ImageNet root under {root}")


def resolve_imagenet_split_dir(root: Path, split: str) -> Path:
    split_name = split.lower()
    return resolve_existing_path(
        [
            root / split_name,
            root / "images" / split_name,
            root / "ILSVRC" / "Data" / "CLS-LOC" / split_name,
        ]
    )


def build_imagenet_dataloader(
    dataset_config: ImageNetDatasetConfig,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    return build_dataloader(
        dataset=ImageNetImageDataset(dataset_config),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
