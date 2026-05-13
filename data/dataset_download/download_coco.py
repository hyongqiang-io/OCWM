from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_DATA_ROOT, download_and_prepare


DEFAULT_TRAIN_URL = "http://images.cocodataset.org/zips/train2017.zip"
DEFAULT_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
DEFAULT_TEST_URL = "http://images.cocodataset.org/zips/test2017.zip"
DEFAULT_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
DEFAULT_TEST_INFO_URL = "http://images.cocodataset.org/annotations/image_info_test2017.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download COCO 2017 archives into a local data directory.")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT / "coco")
    parser.add_argument("--train-url", action="append", default=None, help="COCO train2017 image archive URL.")
    parser.add_argument("--val-url", action="append", default=None, help="COCO val2017 image archive URL.")
    parser.add_argument("--annotations-url", action="append", default=None, help="COCO train/val annotation archive URL.")
    parser.add_argument("--test-url", action="append", default=None, help="COCO test2017 image archive URL.")
    parser.add_argument("--test-info-url", action="append", default=None, help="COCO test2017 image-info archive URL.")
    parser.add_argument("--skip-train", action="store_true", help="Skip downloading train2017 images.")
    parser.add_argument("--skip-val", action="store_true", help="Skip downloading val2017 images.")
    parser.add_argument("--skip-annotations", action="store_true", help="Skip downloading train/val annotations.")
    parser.add_argument("--include-test", action="store_true", help="Also download test2017 images and image_info_test2017.")
    parser.add_argument("--no-extract", action="store_true", help="Only download archives, do not extract.")
    parser.add_argument("--remove-archive", action="store_true", help="Delete archives after extraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resources: list[tuple[str, str]] = []

    if not args.skip_train:
        resources.extend((url, "train2017.zip") for url in (args.train_url or [DEFAULT_TRAIN_URL]))
    if not args.skip_val:
        resources.extend((url, "val2017.zip") for url in (args.val_url or [DEFAULT_VAL_URL]))
    if not args.skip_annotations:
        resources.extend(
            (url, "annotations_trainval2017.zip")
            for url in (args.annotations_url or [DEFAULT_ANNOTATIONS_URL])
        )
    if args.include_test:
        resources.extend((url, "test2017.zip") for url in (args.test_url or [DEFAULT_TEST_URL]))
        resources.extend(
            (url, "image_info_test2017.zip")
            for url in (args.test_info_url or [DEFAULT_TEST_INFO_URL])
        )

    if not resources:
        raise SystemExit("No COCO resources selected. Remove skip flags or pass --include-test.")

    default_resource_count = 3 + (2 if args.include_test else 0)
    for resource_index, (url, output_name) in enumerate(resources):
        unique_output_name = output_name if len(resources) == default_resource_count else f"{resource_index:02d}_{output_name}"
        download_and_prepare(
            url=url,
            root=args.root,
            extract=not args.no_extract,
            remove_archive=args.remove_archive,
            output_name=unique_output_name,
        )


if __name__ == "__main__":
    main()
