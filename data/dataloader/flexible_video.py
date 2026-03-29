from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset

from .common import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    SequenceSource,
    discover_video_sources,
    load_sequence_source,
)


@dataclass
class FlexibleVideoDatasetConfig:
    root: Path
    split: str = "train"
    image_size: int = 224
    frames_per_clip: Optional[int] = 8
    frame_stride: int = 1
    max_sequences: Optional[int] = None
    random_clip: bool = True
    normalize: bool = False
    mean: Sequence[float] = DEFAULT_MEAN
    std: Sequence[float] = DEFAULT_STD


class FlexibleVideoDataset(Dataset):
    def __init__(
        self,
        config: FlexibleVideoDatasetConfig,
        *,
        dataset_name: str,
        split_name: str,
        frame_root: Path,
        mask_root: Optional[Path] = None,
        annotation_path: Optional[Path] = None,
        sequence_annotation_root: Optional[Path] = None,
        sequence_annotation_suffixes: Sequence[str] = (".json",),
        recursive_sources: bool = True,
        allowed_sequences: Optional[set[str]] = None,
        metadata_path: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.frame_root = frame_root
        self.mask_root = mask_root
        self.annotation_path = annotation_path
        self.sequence_annotation_root = sequence_annotation_root
        self.sequence_annotation_suffixes = tuple(sequence_annotation_suffixes)
        self.metadata_path = metadata_path

        sources = discover_video_sources(frame_root, recursive=recursive_sources)
        if allowed_sequences is not None:
            normalized_allowed = {self.normalize_sequence_name(name) for name in allowed_sequences}
            sources = [
                source
                for source in sources
                if self.normalize_sequence_name(source[0]) in normalized_allowed
                or Path(self.normalize_sequence_name(source[0])).name in normalized_allowed
            ]

        if config.max_sequences is not None:
            sources = sources[: config.max_sequences]
        if len(sources) == 0:
            raise ValueError(
                f"No {dataset_name} sequences found for split '{config.split}' under {frame_root}"
            )
        self.sequence_sources = sources

    def __len__(self) -> int:
        return len(self.sequence_sources)

    def __getitem__(self, index: int) -> dict[str, object]:
        sequence_name, source_path, source_type = self.sequence_sources[index]
        frames, frame_indices, frame_refs, sequence_length = load_sequence_source(
            source=(sequence_name, source_path, source_type),
            image_size=self.config.image_size,
            frames_per_clip=self.config.frames_per_clip,
            frame_stride=self.config.frame_stride,
            random_clip=self.config.random_clip,
            normalize=self.config.normalize,
            mean=self.config.mean,
            std=self.config.std,
        )

        sample: dict[str, object] = {
            "frames": frames,
            "frame_indices": frame_indices,
            "frame_paths": frame_refs,
            "sequence_name": sequence_name,
            "sequence_index": torch.tensor(index, dtype=torch.long),
            "sequence_length": torch.tensor(sequence_length, dtype=torch.long),
            "source_type": source_type,
            "split": self.split_name,
        }

        if self.mask_root is not None and source_type == "frames":
            mask_dir = self.mask_root / Path(sequence_name)
            if mask_dir.exists():
                mask_paths = []
                for frame_ref in frame_refs:
                    mask_path = mask_dir / f"{Path(frame_ref).stem}.png"
                    if mask_path.exists():
                        mask_paths.append(str(mask_path))
                if mask_paths:
                    sample["mask_paths"] = mask_paths

        resolved_annotation_path = self._resolve_sequence_annotation_path(sequence_name, source_path)
        if resolved_annotation_path is not None:
            sample["annotation_path"] = str(resolved_annotation_path)
        elif self.annotation_path is not None:
            sample["annotation_path"] = str(self.annotation_path)

        if self.metadata_path is not None:
            sample["metadata_path"] = str(self.metadata_path)
        return sample

    def _resolve_sequence_annotation_path(
        self,
        sequence_name: str,
        source_path: Path,
    ) -> Optional[Path]:
        if self.sequence_annotation_root is None:
            return None

        candidate_bases = [
            Path(sequence_name),
            Path(Path(sequence_name).name),
            Path(source_path.stem),
        ]
        seen: set[Path] = set()
        for base_path in candidate_bases:
            for suffix in self.sequence_annotation_suffixes:
                candidate = self.sequence_annotation_root / base_path.with_suffix(suffix)
                if candidate in seen:
                    continue
                seen.add(candidate)
                if candidate.exists():
                    return candidate
        return None

    @staticmethod
    def normalize_sequence_name(name: str) -> str:
        return Path(name.replace("\\", "/")).with_suffix("").as_posix()
