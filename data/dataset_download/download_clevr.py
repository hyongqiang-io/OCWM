from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_DATA_ROOT, download_and_prepare


DEFAULT_CLEVR_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CLEVR archive into a local data directory.")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT / "clevr")
    parser.add_argument("--url", default=DEFAULT_CLEVR_URL, help="CLEVR archive URL.")
    parser.add_argument("--no-extract", action="store_true", help="Only download archive, do not extract.")
    parser.add_argument("--remove-archive", action="store_true", help="Delete archive after extraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_and_prepare(
        url=args.url,
        root=args.root,
        extract=not args.no_extract,
        remove_archive=args.remove_archive,
    )


if __name__ == "__main__":
    main()
