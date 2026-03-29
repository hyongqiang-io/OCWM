from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .common import DEFAULT_DATASET_ROOT, maybe_resolve_existing_path, resolve_existing_path, build_dataloader
from .flexible_video import FlexibleVideoDataset, FlexibleVideoDatasetConfig


_SPLIT_ALIASES = {
    "validation": "val",
    "valid": "val",
    "eval": "val",
}


@dataclass
class OPNetDatasetConfig(FlexibleVideoDatasetConfig):
    root: Path = field(default_factory=lambda: DEFAULT_DATASET_ROOT / "opnet")


class OPNetVideoDataset(FlexibleVideoDataset):
    def __init__(self, config: OPNetDatasetConfig) -> None:
        root = resolve_opnet_root(config.root)
        split_name = resolve_opnet_split_name(config.split)
        split_root = resolve_opnet_split_root(root, split_name)
        frame_root = resolve_existing_path([
            split_root / "JPEGImages",
            split_root / "frames",
            split_root / "images",
            split_root / "videos",
            split_root,
        ])
        annotation_root = maybe_resolve_existing_path([
            root / "annotations",
            root / "labels",
            root / "scene_data",
            root / "scenes",
            split_root / "annotations",
        ])
        metadata_path = maybe_resolve_existing_path([
            root / f"{split_name}.txt",
            root / "splits" / f"{split_name}.txt",
            split_root / f"{split_name}.txt",
        ])
        super().__init__(
            config=config,
            dataset_name="OPNet",
            split_name=split_name,
            frame_root=frame_root,
            sequence_annotation_root=annotation_root,
            sequence_annotation_suffixes=(".json", ".pkl"),
            recursive_sources=True,
            allowed_sequences=load_split_entries(root, split_name),
            metadata_path=metadata_path,
        )
        self.root = root


def resolve_opnet_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = [root, root / "OPNet", root / "ObjectPermanence", root / "LA-CATER"]
    for candidate in candidates:
        if any((candidate / name).exists() for name in ("train", "val", "test", "videos", "annotations")):
            return candidate
    raise FileNotFoundError(f"Unable to locate OPNet root under {root}")


def resolve_opnet_split_name(split: str) -> str:
    return _SPLIT_ALIASES.get(split, _SPLIT_ALIASES.get(split.lower(), split))


def resolve_opnet_split_root(root: Path, split_name: str) -> Path:
    candidates = [
        root / split_name,
        root / "videos" / split_name,
        root / "JPEGImages" / split_name,
        root / "frames" / split_name,
        root,
    ]
    return resolve_existing_path(candidates)


def load_split_entries(root: Path, split_name: str) -> set[str] | None:
    candidate_paths = [
        root / f"{split_name}.txt",
        root / "splits" / f"{split_name}.txt",
        root / "lists" / f"{split_name}.txt",
    ]
    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue
        with candidate_path.open("r", encoding="utf-8") as file:
            entries = {
                Path(line.strip()).with_suffix("").as_posix()
                for line in file
                if line.strip()
            }
        if entries:
            return entries
    return None


def build_opnet_dataloader(
    dataset_config: OPNetDatasetConfig,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    return build_dataloader(
        dataset=OPNetVideoDataset(dataset_config),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
