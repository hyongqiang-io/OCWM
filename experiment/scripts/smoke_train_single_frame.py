from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import build_model, ensure_result_dir, resolve_device, save_json, timestamped_run_name

from script import TKDISATrainingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-frame TK-DISA smoke training experiment.")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--use-real-dino", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = build_model(use_real_dino=args.use_real_dino)
    pipeline = TKDISATrainingPipeline(model, device=device)

    metrics_history = []
    for step in range(args.steps):
        batch = {
            "images": torch.randn(
                args.batch_size,
                3,
                args.image_size,
                args.image_size,
            )
        }
        result = pipeline.train_step(batch)
        metrics_history.append({"step": step, "loss": result.loss, **result.metrics})

    run_name = args.run_name or timestamped_run_name("smoke_single_frame")
    run_dir = ensure_result_dir(run_name, args.output_dir)
    save_json(
        {
            "run_name": run_name,
            "device": device,
            "use_real_dino": args.use_real_dino,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "metrics": metrics_history,
        },
        run_dir / "metrics.json",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
