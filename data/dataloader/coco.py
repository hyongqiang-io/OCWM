from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .common import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MEAN,
    DEFAULT_STD,
    build_dataloader,
    list_image_files,
    load_image_tensor,
    resolve_existing_path,
)


_SPLIT_ALIASES = {
    "training": "train",
    "validation": "val",
    "valid": "val",
    "eval": "val",
}


@dataclass(frozen=True)
class COCOAnnotationRecord:
    annotation_id: int
    category_id: int
    bbox_xywh: tuple[float, float, float, float]
    area: float
    iscrowd: int


@dataclass(frozen=True)
class COCOImageRecord:
    image_id: int
    file_name: str
    image_path: Path
    width: int
    height: int
    annotations: tuple[COCOAnnotationRecord, ...] = ()
    captions_text: str = ""


@dataclass
class COCODatasetConfig:
    root: Path = field(default_factory=lambda: DEFAULT_DATASET_ROOT / "coco")
    split: str = "train"
    image_size: int = 224
    max_samples: Optional[int] = None
    normalize: bool = False
    mean: Sequence[float] = DEFAULT_MEAN
    std: Sequence[float] = DEFAULT_STD
    use_annotations: bool = True
    include_captions: bool = False
    max_instances: int = 64


class COCOImageDataset(Dataset):
    def __init__(self, config: COCODatasetConfig) -> None:
        if config.max_instances <= 0:
            raise ValueError(f"max_instances must be positive, got {config.max_instances}")

        self.config = config
        self.root = resolve_coco_root(config.root)
        self.split_name = resolve_coco_split_name(config.split)
        self.image_dir = resolve_coco_split_dir(self.root, self.split_name)
        self.annotation_path = resolve_coco_annotation_path(self.root, self.split_name) if config.use_annotations else None
        self.caption_path = resolve_coco_caption_path(self.root, self.split_name) if config.include_captions else None
        self.category_id_to_name: dict[int, str] = {}

        if self.annotation_path is not None:
            self.records, self.category_id_to_name = load_coco_records(
                root=self.root,
                image_dir=self.image_dir,
                annotation_path=self.annotation_path,
                caption_path=self.caption_path,
            )
        else:
            self.records = build_coco_records_from_directory(self.image_dir)

        if config.max_samples is not None:
            self.records = self.records[: config.max_samples]
        if len(self.records) == 0:
            raise ValueError(f"No COCO images found under {self.image_dir}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record = self.records[index]
        sample: dict[str, Tensor | str] = {
            "images": load_image_tensor(
                image_path=record.image_path,
                image_size=self.config.image_size,
                normalize=self.config.normalize,
                mean=self.config.mean,
                std=self.config.std,
            ),
            "image_path": str(record.image_path),
            "image_index": torch.tensor(index, dtype=torch.long),
            "image_id": torch.tensor(record.image_id, dtype=torch.long),
            "original_size": torch.tensor([record.height, record.width], dtype=torch.long),
            "file_name": record.file_name,
        }
        sample.update(pack_coco_annotations(record.annotations, self.config.max_instances))
        if self.annotation_path is not None:
            sample["annotation_path"] = str(self.annotation_path)
        if self.caption_path is not None:
            sample["caption_path"] = str(self.caption_path)
        if record.captions_text:
            sample["captions_text"] = record.captions_text
        return sample


def resolve_coco_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = [root, root / "COCO", root / "coco"]
    for candidate in candidates:
        if any((candidate / name).exists() for name in ("train2017", "val2017", "test2017", "annotations")):
            return candidate
        if any((candidate / "images" / name).exists() for name in ("train2017", "val2017", "test2017")):
            return candidate
    raise FileNotFoundError(f"Unable to locate COCO root under {root}")


def resolve_coco_split_name(split: str) -> str:
    normalized_split = split.lower()
    normalized_split = _SPLIT_ALIASES.get(normalized_split, normalized_split)
    if normalized_split in {"train", "val", "test"}:
        return f"{normalized_split}2017"
    return normalized_split


def resolve_coco_split_dir(root: Path, split: str) -> Path:
    normalized_split = resolve_coco_split_name(split)
    base_split = normalized_split[:-4] if normalized_split.endswith("2017") else normalized_split
    split_candidates = [
        root / normalized_split,
        root / base_split,
        root / "images" / normalized_split,
        root / "images" / base_split,
    ]
    return resolve_existing_path(split_candidates)


def resolve_coco_annotation_path(root: Path, split: str) -> Optional[Path]:
    split_name = resolve_coco_split_name(split)
    base_split = split_name[:-4] if split_name.endswith("2017") else split_name
    annotations_dir = root / "annotations"
    if not annotations_dir.exists():
        return None
    return resolve_optional_coco_annotation_path(
        annotations_dir,
        [
            f"instances_{split_name}.json",
            f"instances_{base_split}.json",
            f"image_info_{split_name}.json",
            f"image_info_{base_split}.json",
        ],
    )


def resolve_coco_caption_path(root: Path, split: str) -> Optional[Path]:
    split_name = resolve_coco_split_name(split)
    base_split = split_name[:-4] if split_name.endswith("2017") else split_name
    annotations_dir = root / "annotations"
    if not annotations_dir.exists():
        return None
    return resolve_optional_coco_annotation_path(
        annotations_dir,
        [
            f"captions_{split_name}.json",
            f"captions_{base_split}.json",
        ],
    )


def resolve_optional_coco_annotation_path(annotations_dir: Path, candidate_names: Sequence[str]) -> Optional[Path]:
    for candidate_name in candidate_names:
        candidate_path = annotations_dir / candidate_name
        if candidate_path.exists():
            return candidate_path
    return None


def load_coco_records(
    root: Path,
    image_dir: Path,
    annotation_path: Path,
    caption_path: Optional[Path],
) -> tuple[list[COCOImageRecord], dict[int, str]]:
    with annotation_path.open("r", encoding="utf-8") as file:
        annotation_data = json.load(file)

    images = annotation_data.get("images") or []
    annotations = annotation_data.get("annotations") or []
    categories = annotation_data.get("categories") or []

    annotations_by_image: dict[int, list[COCOAnnotationRecord]] = defaultdict(list)
    for annotation in annotations:
        image_id = annotation.get("image_id")
        bbox = annotation.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        if image_id is None or len(bbox) != 4:
            continue

        annotation_id = int(annotation.get("id", -1))
        annotations_by_image[int(image_id)].append(
            COCOAnnotationRecord(
                annotation_id=annotation_id,
                category_id=int(annotation.get("category_id", -1)),
                bbox_xywh=tuple(float(value) for value in bbox),
                area=float(annotation.get("area", 0.0)),
                iscrowd=int(annotation.get("iscrowd", 0)),
            )
        )

    captions_by_image = load_coco_captions(caption_path)
    category_id_to_name = {
        int(category["id"]): str(category.get("name", ""))
        for category in categories
        if "id" in category
    }

    records: list[COCOImageRecord] = []
    for image in images:
        file_name = str(image.get("file_name", ""))
        image_id = int(image.get("id", len(records)))
        records.append(
            COCOImageRecord(
                image_id=image_id,
                file_name=file_name,
                image_path=resolve_coco_image_path(root, image_dir, file_name),
                width=int(image.get("width", 0)),
                height=int(image.get("height", 0)),
                annotations=tuple(sorted(annotations_by_image.get(image_id, []), key=lambda item: item.annotation_id)),
                captions_text="\n".join(captions_by_image.get(image_id, [])),
            )
        )

    return records, category_id_to_name


def load_coco_captions(caption_path: Optional[Path]) -> dict[int, list[str]]:
    if caption_path is None or not caption_path.exists():
        return {}

    with caption_path.open("r", encoding="utf-8") as file:
        caption_data = json.load(file)

    captions_by_image: dict[int, list[str]] = defaultdict(list)
    for annotation in caption_data.get("annotations") or []:
        image_id = annotation.get("image_id")
        caption = annotation.get("caption")
        if image_id is None or caption is None:
            continue
        captions_by_image[int(image_id)].append(str(caption))
    return dict(captions_by_image)


def build_coco_records_from_directory(image_dir: Path) -> list[COCOImageRecord]:
    image_paths = list_image_files(image_dir, recursive=True)
    records: list[COCOImageRecord] = []
    for index, image_path in enumerate(image_paths):
        records.append(
            COCOImageRecord(
                image_id=index,
                file_name=image_path.relative_to(image_dir).as_posix(),
                image_path=image_path,
                width=0,
                height=0,
            )
        )
    return records


def resolve_coco_image_path(root: Path, image_dir: Path, file_name: str) -> Path:
    if not file_name:
        raise ValueError("COCO annotation image entry is missing file_name")

    relative_path = Path(file_name)
    candidates = [
        image_dir / relative_path,
        root / relative_path,
        root / "images" / relative_path,
    ]
    return resolve_existing_path(candidates)


def pack_coco_annotations(
    annotations: Sequence[COCOAnnotationRecord],
    max_instances: int,
) -> dict[str, Tensor]:
    boxes_xywh = torch.zeros((max_instances, 4), dtype=torch.float32)
    category_ids = torch.full((max_instances,), -1, dtype=torch.long)
    annotation_ids = torch.full((max_instances,), -1, dtype=torch.long)
    areas = torch.zeros((max_instances,), dtype=torch.float32)
    iscrowd = torch.zeros((max_instances,), dtype=torch.bool)
    instance_mask = torch.zeros((max_instances,), dtype=torch.bool)

    for instance_index, annotation in enumerate(annotations[:max_instances]):
        boxes_xywh[instance_index] = torch.tensor(annotation.bbox_xywh, dtype=torch.float32)
        category_ids[instance_index] = annotation.category_id
        annotation_ids[instance_index] = annotation.annotation_id
        areas[instance_index] = annotation.area
        iscrowd[instance_index] = bool(annotation.iscrowd)
        instance_mask[instance_index] = True

    return {
        "boxes_xywh": boxes_xywh,
        "category_ids": category_ids,
        "annotation_ids": annotation_ids,
        "areas": areas,
        "iscrowd": iscrowd,
        "instance_mask": instance_mask,
        "num_instances": torch.tensor(min(len(annotations), max_instances), dtype=torch.long),
    }


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
