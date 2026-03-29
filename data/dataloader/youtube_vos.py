from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset

from .common import DEFAULT_DATASET_ROOT, DEFAULT_MEAN, DEFAULT_STD, build_dataloader, list_image_files, load_video_clip, resolve_existing_path


_SPLIT_ALIASES = {
    "val": "valid",
    "validation": "valid",
}


@dataclass
class YouTubeVOSDatasetConfig:
    root: Path = field(default_factory=lambda: DEFAULT_DATASET_ROOT / "youtube_vos")
    split: str = "train"
    image_size: int = 224
    frames_per_clip: Optional[int] = 8
    frame_stride: int = 1
    max_sequences: Optional[int] = None
    random_clip: bool = True
    normalize: bool = False
    mean: Sequence[float] = DEFAULT_MEAN
    std: Sequence[float] = DEFAULT_STD


class YouTubeVOSVideoDataset(Dataset):
    def __init__(self, config: YouTubeVOSDatasetConfig) -> None:
        self.config = config
        self.root = resolve_youtube_vos_root(config.root)
        self.split_root = resolve_youtube_vos_split_root(self.root, config.split)
        self.frame_root = self.split_root / "JPEGImages"
        self.mask_root = self.split_root / "Annotations"
        self.sequence_names = sorted(path.name for path in self.frame_root.iterdir() if path.is_dir())
        if config.max_sequences is not None:
            self.sequence_names = self.sequence_names[: config.max_sequences]
        if len(self.sequence_names) == 0:
            raise ValueError(f"No YouTube-VOS sequences found for split '{config.split}'")

    def __len__(self) -> int:
        return len(self.sequence_names)

    def __getitem__(self, index: int) -> dict[str, object]:
        sequence_name = self.sequence_names[index]
        sequence_dir = self.frame_root / sequence_name
        frame_paths = list_image_files(sequence_dir)
        if len(frame_paths) == 0:
            raise ValueError(f"No YouTube-VOS frames found under {sequence_dir}")

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
        mask_paths = []
        if self.mask_root.exists():
            for frame_path in sampled_paths:
                mask_path = self.mask_root / sequence_name / frame_path.name.replace(".jpg", ".png")
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


def resolve_youtube_vos_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = [root, root / "YouTubeVOS", root / "YouTube-VOS"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate YouTube-VOS root under {root}")


def resolve_youtube_vos_split_root(root: Path, split: str) -> Path:
    alias = _SPLIT_ALIASES.get(split, split)
    split_candidates = [
        root / split,
        root / alias,
        root / "all_frames" / split,
        root / "all_frames" / alias,
    ]
    candidate = resolve_existing_path([path for path in split_candidates if (path / "JPEGImages").exists()])
    return candidate


def build_youtube_vos_dataloader(
    dataset_config: YouTubeVOSDatasetConfig,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    return build_dataloader(
        dataset=YouTubeVOSVideoDataset(dataset_config),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
