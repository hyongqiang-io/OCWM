from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import build_model, ensure_result_dir, resolve_device, save_json, timestamped_run_name

from script import TKDISADynamicInferencePipeline, TKDISAInferencePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TK-DISA smoke inference.")
    parser.add_argument("--frames", type=int, default=4)
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
    image_pipeline = TKDISAInferencePipeline(model, device=device)
    video_pipeline = TKDISADynamicInferencePipeline(model, device=device)

    image_result = image_pipeline.infer_image(
        torch.randn(1, 3, args.image_size, args.image_size),
        return_reconstruction=True,
    )
    video_result = video_pipeline.infer_video(
        torch.randn(args.frames, 3, args.image_size, args.image_size),
        return_reconstruction=False,
    )

    run_name = args.run_name or timestamped_run_name("smoke_infer")
    run_dir = ensure_result_dir(run_name, args.output_dir)
    save_json(
        {
            "run_name": run_name,
            "device": device,
            "use_real_dino": args.use_real_dino,
            "frames": args.frames,
            "image_size": args.image_size,
            "image_detection_count": len(image_result.detections[0]),
            "video_frame_count": len(video_result.frames),
            "video_track_counts": [len(frame.tracks) for frame in video_result.frames],
        },
        run_dir / "metrics.json",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
