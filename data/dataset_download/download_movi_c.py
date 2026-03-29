from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from common import DEFAULT_DATA_ROOT, ensure_dir


DEFAULT_TFDS_DATASET = "movi_c/256x256"
DEFAULT_TFDS_DATA_DIR = "gs://kubric-public/tfds"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize MOVi-C from TFDS into a local frame directory.")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT / "movi_c")
    parser.add_argument("--dataset-name", default=DEFAULT_TFDS_DATASET)
    parser.add_argument("--tfds-data-dir", default=DEFAULT_TFDS_DATA_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--include-segmentations", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_scalar(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def extract_metadata(example: dict[str, Any]) -> dict[str, Any]:
    metadata = {}
    raw_metadata = example.get("metadata")
    if isinstance(raw_metadata, dict):
        for key, value in raw_metadata.items():
            metadata[key] = normalize_scalar(value)
    return metadata


def resolve_scene_name(example: dict[str, Any], index: int) -> str:
    metadata = extract_metadata(example)
    candidates = [
        example.get("video_name"),
        example.get("scene_name"),
        metadata.get("video_name"),
        metadata.get("scene_name"),
    ]
    for candidate in candidates:
        normalized = normalize_scalar(candidate)
        if normalized not in (None, ""):
            return str(normalized)
    return f"scene_{index:06d}"


def save_rgb_frames(video: Any, output_dir: Path, overwrite: bool) -> int:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        return len(list(output_dir.iterdir()))

    ensure_dir(output_dir)
    for frame_path in output_dir.glob("*.png"):
        frame_path.unlink()
    for frame_index, frame in enumerate(video):
        Image.fromarray(frame).save(output_dir / f"{frame_index:06d}.png")
    return int(len(video))


def save_segmentation_frames(segmentations: Any, output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        return

    ensure_dir(output_dir)
    for frame_path in output_dir.glob("*.png"):
        frame_path.unlink()
    for frame_index, frame in enumerate(segmentations):
        mask = frame
        if getattr(mask, "ndim", 0) == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        mask = mask.astype("uint16")
        Image.fromarray(mask, mode="I;16").save(output_dir / f"{frame_index:06d}.png")


def main() -> None:
    args = parse_args()

    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise SystemExit(
            "MOVi-C download requires tensorflow_datasets. Install TensorFlow + TFDS first."
        ) from exc

    ensure_dir(args.root)
    for split in args.splits:
        split_root = ensure_dir(args.root / split)
        frame_root = ensure_dir(split_root / "JPEGImages")
        mask_root = ensure_dir(split_root / "Annotations") if args.include_segmentations else None
        metadata_path = split_root / "metadata.jsonl"

        dataset = tfds.load(args.dataset_name, split=split, data_dir=args.tfds_data_dir)
        iterator = tfds.as_numpy(dataset)
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            for index, example in enumerate(iterator):
                if args.max_scenes is not None and index >= args.max_scenes:
                    break
                scene_name = resolve_scene_name(example, index)
                video = example["video"]
                num_frames = save_rgb_frames(video, frame_root / scene_name, overwrite=args.overwrite)

                if args.include_segmentations and "segmentations" in example and mask_root is not None:
                    save_segmentation_frames(
                        example["segmentations"],
                        mask_root / scene_name,
                        overwrite=args.overwrite,
                    )

                payload = {
                    "split": split,
                    "scene_name": scene_name,
                    "index": index,
                    "num_frames": num_frames,
                    "metadata": extract_metadata(example),
                }
                metadata_file.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
