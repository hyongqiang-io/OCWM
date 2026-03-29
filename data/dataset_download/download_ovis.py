from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_DATA_ROOT, download_and_prepare


OFFICIAL_PROJECT_URL = "https://songbai.site/ovis/"
OFFICIAL_COMPETITION_URL = "https://codalab.lisn.upsaclay.fr/competitions/4763#participate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OVIS archives into a local data directory.")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT / "ovis")
    parser.add_argument(
        "--images-url",
        action="append",
        default=[],
        help="OVIS frame archive URL. Repeatable. The official page currently points to the competition page rather than a stable direct archive.",
    )
    parser.add_argument(
        "--annotations-url",
        action="append",
        default=[],
        help="OVIS annotation archive URL. Repeatable.",
    )
    parser.add_argument("--no-extract", action="store_true", help="Only download archives, do not extract.")
    parser.add_argument("--remove-archive", action="store_true", help="Delete archives after extraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resources = [
        (url, f"images_{index:02d}.zip")
        for index, url in enumerate(args.images_url)
    ] + [
        (url, f"annotations_{index:02d}.zip")
        for index, url in enumerate(args.annotations_url)
    ]
    if not resources:
        raise SystemExit(
            "No OVIS URLs provided. Pass --images-url / --annotations-url with your accessible source links. "
            f"Official pages: {OFFICIAL_PROJECT_URL} and {OFFICIAL_COMPETITION_URL}"
        )

    for url, output_name in resources:
        download_and_prepare(
            url=url,
            root=args.root,
            extract=not args.no_extract,
            remove_archive=args.remove_archive,
            output_name=output_name,
        )


if __name__ == "__main__":
    main()
