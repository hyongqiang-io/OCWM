from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_DATA_ROOT, download_and_prepare


DEFAULT_VIDEO_URL = "https://cmu.box.com/shared/static/yvhx9p5haip5abzh9i2fofssjpq34zwz.zip"
DEFAULT_SCENES_URL = "https://cmu.box.com/shared/static/zfau8j1e6n7ylobf0g1d2wjdgdu86j2e.zip"
DEFAULT_LISTS_URL = "https://cmu.box.com/shared/static/i9kexj33if00t338esnw93uzm5f6sfar.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CATER archives into a local data directory.")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT / "cater")
    parser.add_argument("--video-url", action="append", default=None, help="CATER video archive URL.")
    parser.add_argument("--scene-url", action="append", default=None, help="CATER scene archive URL.")
    parser.add_argument("--lists-url", action="append", default=None, help="CATER split-list archive URL.")
    parser.add_argument("--no-extract", action="store_true", help="Only download archives, do not extract.")
    parser.add_argument("--remove-archive", action="store_true", help="Delete archives after extraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_urls = args.video_url or [DEFAULT_VIDEO_URL]
    scene_urls = args.scene_url or [DEFAULT_SCENES_URL]
    lists_urls = args.lists_url or [DEFAULT_LISTS_URL]
    resources = [
        (url, "videos.zip") for url in video_urls
    ] + [
        (url, "scenes.zip") for url in scene_urls
    ] + [
        (url, "lists.zip") for url in lists_urls
    ]

    for resource_index, (url, default_name) in enumerate(resources):
        output_name = default_name if len(resources) == 3 else f"{resource_index:02d}_{default_name}"
        download_and_prepare(
            url=url,
            root=args.root,
            extract=not args.no_extract,
            remove_archive=args.remove_archive,
            output_name=output_name,
        )


if __name__ == "__main__":
    main()
