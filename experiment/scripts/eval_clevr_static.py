from __future__ import annotations

import argparse
from pathlib import Path

from run_experiment import main as run_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for unified CLEVR static evaluation.")
    parser.add_argument("--data-root", type=Path, default=Path("data/dataset/clevr"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--use-real-dino", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    argv = [
        "--data-root", str(args.data_root),
        "--eval-split", args.split,
        "--image-size", str(args.image_size),
        "--max-items", str(args.max_samples),
        "--eval-batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
    ]
    if args.device is not None:
        argv.extend(["--device", args.device])
    if args.output_dir is not None:
        argv.extend(["--output-root", str(args.output_dir)])
    if args.run_name is not None:
        argv.extend(["--run-name", args.run_name])
    if args.checkpoint is not None:
        argv.extend(["--checkpoint", str(args.checkpoint)])
    if args.use_real_dino:
        argv.append("--use-real-dino")
    if args.normalize:
        argv.append("--normalize")
    run_main(argv=argv, default_mode="eval", default_dataset="clevr")


if __name__ == "__main__":
    main()
