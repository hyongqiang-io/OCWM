from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_DATA_ROOT, download_and_prepare, download_from_google_drive


DEFAULT_TRAIN_URL = "https://drive.google.com/file/d/1-ehpl5s0Fd14WwtT-GmWtIWa_BxZl9D6/view?usp=share_link"
DEFAULT_VAL_URL = "https://drive.google.com/file/d/17Hwc__6i2rpF5e2s5OPqoywNxG5bzlcO/view?usp=share_link"
DEFAULT_TEST_URL = "https://drive.google.com/file/d/1Vp_y8dSUO4ktYmeBFkIQnmAxK6bl3Eyf/view?usp=share_link"
DEFAULT_META_FOLDER_URL = "https://drive.google.com/drive/folders/1EtTW57QfSkUK3Jl1A_m9D120muA7NG4L?usp=share_link"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LVOS v2 archives into a local data directory.")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT / "lvos_v2")
    parser.add_argument("--train-url", action="append", default=None)
    parser.add_argument("--val-url", action="append", default=None)
    parser.add_argument("--test-url", action="append", default=None)
    parser.add_argument("--meta-folder-url", default=DEFAULT_META_FOLDER_URL)
    parser.add_argument("--skip-meta", action="store_true", help="Skip downloading the public LVOS metadata folder.")
    parser.add_argument("--no-extract", action="store_true", help="Only download archives, do not extract.")
    parser.add_argument("--remove-archive", action="store_true", help="Delete archives after extraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_urls = args.train_url or [DEFAULT_TRAIN_URL]
    val_urls = args.val_url or [DEFAULT_VAL_URL]
    test_urls = args.test_url or [DEFAULT_TEST_URL]
    resources = [
        (url, "train.zip") for url in train_urls
    ] + [
        (url, "val.zip") for url in val_urls
    ] + [
        (url, "test.zip") for url in test_urls
    ]

    for resource_index, (url, default_name) in enumerate(resources):
        output_name = default_name if len(resources) == 3 else f"{resource_index:02d}_{default_name}"
        download_and_prepare(
            url=url,
            root=args.root,
            extract=not args.no_extract,
            remove_archive=args.remove_archive,
            output_name=output_name,
            use_gdown=True,
        )

    if not args.skip_meta and args.meta_folder_url:
        download_from_google_drive(args.meta_folder_url, args.root / "meta", folder=True)


if __name__ == "__main__":
    main()
