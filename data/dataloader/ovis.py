from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .common import DEFAULT_DATASET_ROOT, maybe_resolve_existing_path, resolve_existing_path, build_dataloader
from .flexible_video import FlexibleVideoDataset, FlexibleVideoDatasetConfig


_SPLIT_ALIASES = {
    "validation": "valid",
    "val": "valid",
    "eval": "valid",
}


@dataclass
class OVISDatasetConfig(FlexibleVideoDatasetConfig):
    root: Path = field(default_factory=lambda: DEFAULT_DATASET_ROOT / "ovis")


class OVISVideoDataset(FlexibleVideoDataset):
    def __init__(self, config: OVISDatasetConfig) -> None:
        root = resolve_ovis_root(config.root)
        split_name, frame_root = resolve_ovis_split_root(root, config.split)
        annotation_path = maybe_resolve_existing_path([
            root / "annotations" / f"annotations_{split_name}.json",
            root / f"annotations_{split_name}.json",
            root / "annotations" / f"{split_name}.json",
        ])
        super().__init__(
            config=config,
            dataset_name="OVIS",
            split_name=split_name,
            frame_root=frame_root,
            annotation_path=annotation_path,
            recursive_sources=True,
        )
        self.root = root


def resolve_ovis_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = [root, root / "OVIS"]
    for candidate in candidates:
        if any((candidate / name).exists() for name in ("train_images", "valid_images", "annotations")):
            return candidate
    raise FileNotFoundError(f"Unable to locate OVIS root under {root}")


def resolve_ovis_split_root(root: Path, split: str) -> tuple[str, Path]:
    split_candidates = [split, split.lower()]
    alias = _SPLIT_ALIASES.get(split) or _SPLIT_ALIASES.get(split.lower())
    if alias is not None:
        split_candidates.append(alias)

    for split_name in split_candidates:
        candidates = [
            root / f"{split_name}_images",
            root / split_name,
            root / "JPEGImages" / split_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return split_name, candidate

    rendered = ", ".join(split_candidates)
    raise FileNotFoundError(f"Unable to locate OVIS split '{split}' ({rendered}) under {root}")


def build_ovis_dataloader(
    dataset_config: OVISDatasetConfig,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    return build_dataloader(
        dataset=OVISVideoDataset(dataset_config),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
