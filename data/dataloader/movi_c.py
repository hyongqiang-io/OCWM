from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .common import DEFAULT_DATASET_ROOT, build_dataloader, maybe_resolve_existing_path, resolve_existing_path
from .flexible_video import FlexibleVideoDataset, FlexibleVideoDatasetConfig


_SPLIT_ALIASES = {
    "val": "validation",
    "valid": "validation",
    "eval": "validation",
}


@dataclass
class MOViCDatasetConfig(FlexibleVideoDatasetConfig):
    root: Path = field(default_factory=lambda: DEFAULT_DATASET_ROOT / "movi_c")


class MOViCVideoDataset(FlexibleVideoDataset):
    def __init__(self, config: MOViCDatasetConfig) -> None:
        root = resolve_movi_c_root(config.root)
        split_name, split_root = resolve_movi_c_split_root(root, config.split)
        frame_root = resolve_existing_path([
            split_root / "JPEGImages",
            split_root / "images",
            split_root / "video",
            split_root,
        ])
        mask_root = maybe_resolve_existing_path([
            split_root / "Annotations",
            split_root / "segmentations",
        ])
        metadata_path = maybe_resolve_existing_path([
            split_root / "metadata.jsonl",
            split_root / "metadata.json",
        ])
        super().__init__(
            config=config,
            dataset_name="MOVi-C",
            split_name=split_name,
            frame_root=frame_root,
            mask_root=mask_root,
            recursive_sources=True,
            metadata_path=metadata_path,
        )
        self.root = root


def resolve_movi_c_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = [root, root / "MOVi-C", root / "movi_c"]
    for candidate in candidates:
        if any((candidate / split_name).exists() for split_name in ("train", "validation", "test")):
            return candidate
        if (candidate / "256x256").exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate MOVi-C root under {root}")


def resolve_movi_c_split_root(root: Path, split: str) -> tuple[str, Path]:
    split_candidates = [split, split.lower()]
    alias = _SPLIT_ALIASES.get(split) or _SPLIT_ALIASES.get(split.lower())
    if alias is not None:
        split_candidates.append(alias)

    for split_name in split_candidates:
        for candidate in (root / split_name, root / "256x256" / split_name):
            if candidate.exists():
                return split_name, candidate

    rendered = ", ".join(split_candidates)
    raise FileNotFoundError(f"Unable to locate MOVi-C split '{split}' ({rendered}) under {root}")


def build_movi_c_dataloader(
    dataset_config: MOViCDatasetConfig,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    return build_dataloader(
        dataset=MOViCVideoDataset(dataset_config),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
