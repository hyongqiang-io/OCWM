from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .common import DEFAULT_DATASET_ROOT, build_dataloader, maybe_resolve_existing_path, resolve_existing_path
from .flexible_video import FlexibleVideoDataset, FlexibleVideoDatasetConfig


_SPLIT_ALIASES = {
    "validation": "val",
    "valid": "val",
    "eval": "val",
}


@dataclass
class LVOSV2DatasetConfig(FlexibleVideoDatasetConfig):
    root: Path = field(default_factory=lambda: DEFAULT_DATASET_ROOT / "lvos_v2")


class LVOSV2VideoDataset(FlexibleVideoDataset):
    def __init__(self, config: LVOSV2DatasetConfig) -> None:
        root = resolve_lvos_v2_root(config.root)
        split_name, split_root = resolve_lvos_v2_split_root(root, config.split)
        frame_root = resolve_existing_path([
            split_root / "JPEGImages",
            split_root / "images",
            split_root,
        ])
        mask_root = maybe_resolve_existing_path([
            split_root / "Annotations",
            split_root / "masks",
        ])
        metadata_path = maybe_resolve_existing_path([
            split_root / f"{split_name}_meta.json",
            split_root / "meta.json",
            root / f"{split_name}_meta.json",
        ])
        super().__init__(
            config=config,
            dataset_name="LVOS v2",
            split_name=split_name,
            frame_root=frame_root,
            mask_root=mask_root,
            recursive_sources=True,
            metadata_path=metadata_path,
        )
        self.root = root


def resolve_lvos_v2_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = [root, root / "LVOS", root / "LVOS_v2", root / "LVOS V2"]
    for candidate in candidates:
        if any((candidate / name).exists() for name in ("train", "val", "test")):
            return candidate
    raise FileNotFoundError(f"Unable to locate LVOS v2 root under {root}")


def resolve_lvos_v2_split_root(root: Path, split: str) -> tuple[str, Path]:
    split_candidates = [split, split.lower()]
    alias = _SPLIT_ALIASES.get(split) or _SPLIT_ALIASES.get(split.lower())
    if alias is not None:
        split_candidates.append(alias)

    for split_name in split_candidates:
        for candidate in (root / split_name, root / split_name.lower()):
            if candidate.exists():
                return split_name, candidate

    rendered = ", ".join(split_candidates)
    raise FileNotFoundError(f"Unable to locate LVOS v2 split '{split}' ({rendered}) under {root}")


def build_lvos_v2_dataloader(
    dataset_config: LVOSV2DatasetConfig,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    return build_dataloader(
        dataset=LVOSV2VideoDataset(dataset_config),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
