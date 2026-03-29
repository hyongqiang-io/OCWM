from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_DATA_ROOT, download_and_prepare


DEFAULT_TRAIN_PART_1_URL = "https://livebiuac-my.sharepoint.com/:u:/g/personal/aviv_shamsian_live_biu_ac_il/EXpwyuwQvhdMrnOW3byMznQBFLKo8YR7C4dGD8iIzqxe1Q?e=iKCh0d"
DEFAULT_TRAIN_PART_2_URL = "https://livebiuac-my.sharepoint.com/:u:/g/personal/aviv_shamsian_live_biu_ac_il/ETkMqR9YBAVIq1gYq5g1C_QB_kp4VeD-FcZKPoJR_TtKVA?e=51mZfW"
DEFAULT_VAL_URL = "https://livebiuac-my.sharepoint.com/:u:/g/personal/aviv_shamsian_live_biu_ac_il/EfRaEebOq_RIsJ5_3fiY1T0BMrCZ82lcqPQkNpBr0vqPpw?e=yu5klZ"
DEFAULT_TEST_URL = "https://livebiuac-my.sharepoint.com/:u:/g/personal/aviv_shamsian_live_biu_ac_il/EbpIL4SqxARAmQLiPP1vs30BQbg38_t09iQvoHvVUFxd0Q?e=9mYDrI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OPNet / LA-CATER archives into a local data directory.")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT / "opnet")
    parser.add_argument("--train-part-1-url", action="append", default=None)
    parser.add_argument("--train-part-2-url", action="append", default=None)
    parser.add_argument("--val-url", action="append", default=None)
    parser.add_argument("--test-url", action="append", default=None)
    parser.add_argument("--no-extract", action="store_true", help="Only download archives, do not extract.")
    parser.add_argument("--remove-archive", action="store_true", help="Delete archives after extraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_part_1_urls = args.train_part_1_url or [DEFAULT_TRAIN_PART_1_URL]
    train_part_2_urls = args.train_part_2_url or [DEFAULT_TRAIN_PART_2_URL]
    val_urls = args.val_url or [DEFAULT_VAL_URL]
    test_urls = args.test_url or [DEFAULT_TEST_URL]
    resources = [
        (url, "train_part_1.zip") for url in train_part_1_urls
    ] + [
        (url, "train_part_2.zip") for url in train_part_2_urls
    ] + [
        (url, "validation.zip") for url in val_urls
    ] + [
        (url, "test.zip") for url in test_urls
    ]

    for resource_index, (url, default_name) in enumerate(resources):
        output_name = default_name if len(resources) == 4 else f"{resource_index:02d}_{default_name}"
        download_and_prepare(
            url=url,
            root=args.root,
            extract=not args.no_extract,
            remove_archive=args.remove_archive,
            output_name=output_name,
        )


if __name__ == "__main__":
    main()
