from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .common import DEFAULT_DATASET_ROOT, DEFAULT_MEAN, DEFAULT_STD, build_dataloader, list_image_files, load_image_tensor, resolve_existing_path


@dataclass
class COCODatasetConfig:
    root: Path = field(default_factory=lambda: DEFAULT_DATASET_ROOT / "coco")
    split: str = "train"
    image_size: int = 224
    max_samples: Optional[int] = None
    normalize: bool = False
    mean: Sequence[float] = DEFAULT_MEAN
    std: Sequence[float] = DEFAULT_STD


class COCOImageDataset(Dataset):
    def __init__(self, config: COCODatasetConfig) -> None:
        self.config = config
        self.root = resolve_coco_root(config.root)
        self.image_dir = resolve_coco_split_dir(self.root, config.split)
        self.image_paths = list_image_files(self.image_dir)
        if config.max_samples is not None:
            self.image_paths = self.image_paths[: config.max_samples]
        if len(self.image_paths) == 0:
            raise ValueError(f"No COCO images found under {self.image_dir}")
        self.annotation_path = resolve_coco_annotation_path(self.root, config.split)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        image_path = self.image_paths[index]
        sample: dict[str, Tensor | str] = {
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
        if self.annotation_path is not None:
            sample["annotation_path"] = str(self.annotation_path)
        return sample


def resolve_coco_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = [root, root / "COCO", root / "coco"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate COCO root under {root}")


def resolve_coco_split_dir(root: Path, split: str) -> Path:
    normalized_split = split.lower()
    split_candidates = [
        root / normalized_split,
        root / f"{normalized_split}2017",
        root / "images" / normalized_split,
        root / "images" / f"{normalized_split}2017",
    ]
    if normalized_split.endswith("2017"):
        base_split = normalized_split[:-4]
        split_candidates.extend([
            root / base_split,
            root / "images" / base_split,
        ])
    return resolve_existing_path(split_candidates)


def resolve_coco_annotation_path(root: Path, split: str) -> Optional[Path]:
    normalized_split = split.lower()
    candidate_names = [
        f"instances_{normalized_split}.json",
        f"instances_{normalized_split}2017.json",
        f"image_info_{normalized_split}.json",
        f"image_info_{normalized_split}2017.json",
    ]
    annotations_dir = root / "annotations"
    if not annotations_dir.exists():
        return None
    for candidate_name in candidate_names:
        candidate_path = annotations_dir / candidate_name
        if candidate_path.exists():
            return candidate_path
    return None


def build_coco_dataloader(
    dataset_config: COCODatasetConfig,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    return build_dataloader(
        dataset=COCOImageDataset(dataset_config),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
