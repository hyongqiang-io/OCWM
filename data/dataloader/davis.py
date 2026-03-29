from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset

from .common import DEFAULT_DATASET_ROOT, DEFAULT_MEAN, DEFAULT_STD, build_dataloader, list_image_files, load_video_clip, resolve_existing_path


@dataclass
class DAVISDatasetConfig:
    root: Path = field(default_factory=lambda: DEFAULT_DATASET_ROOT / "davis")
    split: str = "train"
    image_size: int = 224
    frames_per_clip: Optional[int] = 8
    frame_stride: int = 1
    max_sequences: Optional[int] = None
    random_clip: bool = True
    normalize: bool = False
    mean: Sequence[float] = DEFAULT_MEAN
    std: Sequence[float] = DEFAULT_STD


class DAVISVideoDataset(Dataset):
    def __init__(self, config: DAVISDatasetConfig) -> None:
        self.config = config
        self.root = resolve_davis_root(config.root)
        self.frame_root = resolve_davis_frame_root(self.root)
        self.mask_root = resolve_davis_mask_root(self.root, self.frame_root.name)
        self.sequence_names = resolve_davis_sequences(self.root, self.frame_root, config.split)
        if config.max_sequences is not None:
            self.sequence_names = self.sequence_names[: config.max_sequences]
        if len(self.sequence_names) == 0:
            raise ValueError(f"No DAVIS sequences found for split '{config.split}'")

    def __len__(self) -> int:
        return len(self.sequence_names)

    def __getitem__(self, index: int) -> dict[str, object]:
        sequence_name = self.sequence_names[index]
        sequence_dir = self.frame_root / sequence_name
        frame_paths = list_image_files(sequence_dir)
        if len(frame_paths) == 0:
            raise ValueError(f"No DAVIS frames found under {sequence_dir}")

        frames, frame_indices, sampled_paths = load_video_clip(
            frame_paths=frame_paths,
            image_size=self.config.image_size,
            frames_per_clip=self.config.frames_per_clip,
            frame_stride=self.config.frame_stride,
            random_clip=self.config.random_clip,
            normalize=self.config.normalize,
            mean=self.config.mean,
            std=self.config.std,
        )
        sampled_names = [path.name for path in sampled_paths]
        mask_paths = []
        if self.mask_root is not None:
            for frame_name in sampled_names:
                mask_path = self.mask_root / sequence_name / frame_name.replace(".jpg", ".png")
                if mask_path.exists():
                    mask_paths.append(str(mask_path))

        return {
            "frames": frames,
            "frame_indices": frame_indices,
            "frame_paths": [str(path) for path in sampled_paths],
            "sequence_name": sequence_name,
            "sequence_index": torch.tensor(index, dtype=torch.long),
            "mask_paths": mask_paths,
            "sequence_length": torch.tensor(len(frame_paths), dtype=torch.long),
        }


def resolve_davis_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = [root, root / "DAVIS"]
    for candidate in candidates:
        if (candidate / "JPEGImages").exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate DAVIS root under {root}")


def resolve_davis_frame_root(root: Path) -> Path:
    jpeg_root = root / "JPEGImages"
    candidates = [jpeg_root / "480p", jpeg_root / "Full-Resolution", jpeg_root / "1080p"]
    return resolve_existing_path(candidates)


def resolve_davis_mask_root(root: Path, resolution_name: str) -> Optional[Path]:
    candidate = root / "Annotations" / resolution_name
    if candidate.exists():
        return candidate
    return None


def resolve_davis_sequences(root: Path, frame_root: Path, split: str) -> list[str]:
    split_candidates = [
        root / "ImageSets" / "2017" / f"{split}.txt",
        root / "ImageSets" / "2016" / f"{split}.txt",
    ]
    for split_file in split_candidates:
        if split_file.exists():
            with split_file.open("r", encoding="utf-8") as file:
                return [line.strip() for line in file if line.strip()]
    return sorted(path.name for path in frame_root.iterdir() if path.is_dir())


def build_davis_dataloader(
    dataset_config: DAVISDatasetConfig,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    return build_dataloader(
        dataset=DAVISVideoDataset(dataset_config),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
