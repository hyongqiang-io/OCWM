from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch

from common import DEFAULT_RESULTS_ROOT, REPO_ROOT, build_model, collect_runtime_info, resolve_device, timestamped_run_name
from data.dataloader import (
    CATERDatasetConfig,
    CLEVRCoGenTDatasetConfig,
    CLEVRDatasetConfig,
    COCODatasetConfig,
    DAVISDatasetConfig,
    ImageNetDatasetConfig,
    LVOSV2DatasetConfig,
    MOViCDatasetConfig,
    OPNetDatasetConfig,
    OVISDatasetConfig,
    YouTubeVOSDatasetConfig,
    build_cater_dataloader,
    build_clevr_cogent_dataloader,
    build_clevr_dataloader,
    build_coco_dataloader,
    build_davis_dataloader,
    build_imagenet_dataloader,
    build_lvos_v2_dataloader,
    build_movi_c_dataloader,
    build_opnet_dataloader,
    build_ovis_dataloader,
    build_youtube_vos_dataloader,
)
from module.dynamic import TKDISADynamicConfig
from module.slot_encoder import TKDISAConfig
from script import TKDISADynamicLoss, TKDISADynamicTrainingPipeline, TKDISALoss, TKDISATrainingPipeline
from run_utils import (
    append_jsonl,
    ensure_run_package,
    evaluate_dynamic,
    evaluate_static,
    load_checkpoint,
    save_checkpoint,
    save_metric_curves,
    save_summary,
)


DATASET_KINDS = {
    "clevr": "static",
    "clevr_cogent": "static",
    "coco": "static",
    "imagenet": "static",
    "cater": "dynamic",
    "davis": "dynamic",
    "lvos_v2": "dynamic",
    "movi_c": "dynamic",
    "opnet": "dynamic",
    "ovis": "dynamic",
    "youtube_vos": "dynamic",
}

DEFAULT_DATA_ROOTS = {
    "clevr": REPO_ROOT / "data" / "dataset" / "clevr",
    "clevr_cogent": REPO_ROOT / "data" / "dataset" / "clevr_cogent",
    "coco": REPO_ROOT / "data" / "dataset" / "coco",
    "imagenet": REPO_ROOT / "data" / "dataset" / "imagenet",
    "cater": REPO_ROOT / "data" / "dataset" / "cater",
    "davis": REPO_ROOT / "data" / "dataset" / "davis",
    "lvos_v2": REPO_ROOT / "data" / "dataset" / "lvos_v2",
    "movi_c": REPO_ROOT / "data" / "dataset" / "movi_c",
    "opnet": REPO_ROOT / "data" / "dataset" / "opnet",
    "ovis": REPO_ROOT / "data" / "dataset" / "ovis",
    "youtube_vos": REPO_ROOT / "data" / "dataset" / "youtube_vos",
}

DEFAULT_SPLITS = {
    "clevr": {"train": "train", "eval": "val"},
    "clevr_cogent": {"train": "trainA", "eval": "valA"},
    "coco": {"train": "train", "eval": "val"},
    "imagenet": {"train": "train", "eval": "val"},
    "cater": {"train": "train", "eval": "val"},
    "davis": {"train": "train", "eval": "val"},
    "lvos_v2": {"train": "train", "eval": "val"},
    "movi_c": {"train": "train", "eval": "validation"},
    "opnet": {"train": "train", "eval": "val"},
    "ovis": {"train": "train", "eval": "valid"},
    "youtube_vos": {"train": "train", "eval": "valid"},
}


def parse_args(
    argv: Optional[list[str]] = None,
    default_mode: Optional[str] = None,
    default_dataset: Optional[str] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified TK-DISA experiment entrypoint.")
    parser.add_argument("--mode", choices=("train", "eval"), default=default_mode or "train")
    parser.add_argument("--dataset", choices=tuple(sorted(DATASET_KINDS)), default=default_dataset or "clevr")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--val-every-steps", type=int, default=50)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-real-dino", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--visualize-samples", type=int, default=8)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--amp-dtype", default="float16")
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--dino-model-name", default="dinov2_vits14")
    parser.add_argument("--dino-layers", type=int, nargs="+", default=[4, 11])
    parser.add_argument("--dino-patch-size", type=int, default=14)
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--d-vis", type=int, default=256)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--slot-iterations", type=int, default=3)
    parser.add_argument("--vq-decay", type=float, default=0.99)
    parser.add_argument("--commitment-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--vq-weight", type=float, default=1.0)
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--temporal-visual-weight", type=float, default=1.0)
    parser.add_argument("--temporal-velocity-weight", type=float, default=0.1)
    parser.add_argument("--max-temporal-match-distance", type=float, default=0.25)
    parser.add_argument("--max-occlusion", type=int, default=30)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--position-threshold", type=float, default=0.2)
    parser.add_argument("--cost-threshold", type=float, default=2.0)
    parser.add_argument("--velocity-decay", type=float, default=0.95)
    parser.add_argument("--measurement-alpha", type=float, default=0.7)
    args = parser.parse_args(argv)

    if args.resume and args.checkpoint is None:
        parser.error("--resume requires --checkpoint.")
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if args.max_eval_items is None:
        args.max_eval_items = args.max_items
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_epoch_seed(seed: int, epoch: int) -> None:
    random.seed(seed + epoch)
    torch.manual_seed(seed + epoch)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + epoch)


def normalize_split(split: Optional[str]) -> Optional[str]:
    if split is None:
        return None
    if split.strip().lower() in {"", "none", "null", "skip", "off", "false"}:
        return None
    return split


def sync_model_config(config: TKDISAConfig) -> TKDISAConfig:
    feature_dim = config.encoder.resolved_output_dim()
    config.position.d_feature = feature_dim
    config.visual.d_feature = feature_dim
    config.visual.k_max = config.position.k_max
    config.decoder.d_feature = feature_dim
    config.decoder.d_vis = config.visual.d_vis
    return config


def build_model_config(args: argparse.Namespace) -> TKDISAConfig:
    config = TKDISAConfig()
    config.encoder.model_name = args.dino_model_name
    config.encoder.layers = tuple(args.dino_layers)
    config.encoder.patch_size = args.dino_patch_size

    config.position.k_max = args.k_max
    config.position.num_iterations = args.slot_iterations

    config.visual.k_max = args.k_max
    config.visual.d_vis = args.d_vis
    config.visual.num_iterations = args.slot_iterations
    config.visual.codebook_size = args.codebook_size
    config.visual.vq_decay = args.vq_decay
    config.visual.commitment_weight = args.commitment_weight

    config.loss.reconstruction_weight = args.reconstruction_weight
    config.loss.vq_weight = args.vq_weight
    config.loss.position_weight = args.position_weight

    config.train.learning_rate = args.learning_rate
    config.train.weight_decay = args.weight_decay
    config.train.grad_clip_norm = args.grad_clip_norm
    config.train.amp = not args.disable_amp
    config.train.amp_dtype = args.amp_dtype

    config.inference.confidence_threshold = args.confidence_threshold
    config.inference.return_reconstruction = True
    config.inference.amp = not args.disable_amp
    config.inference.amp_dtype = args.amp_dtype
    return sync_model_config(config)


def build_dynamic_config(args: argparse.Namespace) -> TKDISADynamicConfig:
    config = TKDISADynamicConfig()
    config.loss.temporal_visual_weight = args.temporal_visual_weight
    config.loss.temporal_velocity_weight = args.temporal_velocity_weight
    config.loss.max_temporal_match_distance = args.max_temporal_match_distance

    config.slot_manager.max_occlusion = args.max_occlusion
    config.slot_manager.min_confidence = args.min_confidence
    config.slot_manager.position_threshold = args.position_threshold
    config.slot_manager.cost_threshold = args.cost_threshold
    config.slot_manager.velocity_decay = args.velocity_decay
    config.slot_manager.measurement_alpha = args.measurement_alpha

    config.train.learning_rate = args.learning_rate
    config.train.weight_decay = args.weight_decay
    config.train.grad_clip_norm = args.grad_clip_norm
    config.train.amp = not args.disable_amp
    config.train.amp_dtype = args.amp_dtype

    config.inference.confidence_threshold = args.confidence_threshold
    config.inference.return_reconstruction = True
    config.inference.amp = not args.disable_amp
    config.inference.amp_dtype = args.amp_dtype
    return config


def build_dataloader(
    args: argparse.Namespace,
    split: str,
    device: torch.device,
    training: bool,
) -> Any:
    dataset = args.dataset
    dataset_kind = DATASET_KINDS[dataset]
    data_root = (args.data_root or DEFAULT_DATA_ROOTS[dataset]).expanduser().resolve()
    max_items = args.max_items if training else args.max_eval_items
    batch_size = args.batch_size if training else args.eval_batch_size
    pin_memory = device.type == "cuda"

    if dataset == "clevr":
        return build_clevr_dataloader(
            CLEVRDatasetConfig(
                root=data_root,
                split=split,
                image_size=args.image_size,
                max_samples=max_items,
                normalize=args.normalize,
            ),
            batch_size=batch_size,
            shuffle=training,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    if dataset == "clevr_cogent":
        return build_clevr_cogent_dataloader(
            CLEVRCoGenTDatasetConfig(
                root=data_root,
                split=split,
                image_size=args.image_size,
                max_samples=max_items,
                normalize=args.normalize,
            ),
            batch_size=batch_size,
            shuffle=training,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    if dataset == "coco":
        return build_coco_dataloader(
            COCODatasetConfig(
                root=data_root,
                split=split,
                image_size=args.image_size,
                max_samples=max_items,
                normalize=args.normalize,
            ),
            batch_size=batch_size,
            shuffle=training,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    if dataset == "imagenet":
        return build_imagenet_dataloader(
            ImageNetDatasetConfig(
                root=data_root,
                split=split,
                image_size=args.image_size,
                max_samples=max_items,
                normalize=args.normalize,
            ),
            batch_size=batch_size,
            shuffle=training,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    frames_per_clip = args.frames_per_clip if args.frames_per_clip > 0 else None
    if dataset_kind != "dynamic":
        raise ValueError(f"Unsupported dataset kind for dataloader construction: {dataset_kind}")

    if dataset == "cater":
        return build_cater_dataloader(
            CATERDatasetConfig(
                root=data_root,
                split=split,
                image_size=args.image_size,
                frames_per_clip=frames_per_clip,
                frame_stride=args.frame_stride,
                max_sequences=max_items,
                random_clip=training,
                normalize=args.normalize,
            ),
            batch_size=batch_size,
            shuffle=training,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    if dataset == "davis":
        return build_davis_dataloader(
            DAVISDatasetConfig(
                root=data_root,
                split=split,
                image_size=args.image_size,
                frames_per_clip=frames_per_clip,
                frame_stride=args.frame_stride,
                max_sequences=max_items,
                random_clip=training,
                normalize=args.normalize,
            ),
            batch_size=batch_size,
            shuffle=training,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    if dataset == "lvos_v2":
        return build_lvos_v2_dataloader(
            LVOSV2DatasetConfig(
                root=data_root,
                split=split,
                image_size=args.image_size,
                frames_per_clip=frames_per_clip,
                frame_stride=args.frame_stride,
                max_sequences=max_items,
                random_clip=training,
                normalize=args.normalize,
            ),
            batch_size=batch_size,
            shuffle=training,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    if dataset == "movi_c":
        return build_movi_c_dataloader(
            MOViCDatasetConfig(
                root=data_root,
                split=split,
                image_size=args.image_size,
                frames_per_clip=frames_per_clip,
                frame_stride=args.frame_stride,
                max_sequences=max_items,
                random_clip=training,
                normalize=args.normalize,
            ),
            batch_size=batch_size,
            shuffle=training,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    if dataset == "opnet":
        return build_opnet_dataloader(
            OPNetDatasetConfig(
                root=data_root,
                split=split,
                image_size=args.image_size,
                frames_per_clip=frames_per_clip,
                frame_stride=args.frame_stride,
                max_sequences=max_items,
                random_clip=training,
                normalize=args.normalize,
            ),
            batch_size=batch_size,
            shuffle=training,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    if dataset == "ovis":
        return build_ovis_dataloader(
            OVISDatasetConfig(
                root=data_root,
                split=split,
                image_size=args.image_size,
                frames_per_clip=frames_per_clip,
                frame_stride=args.frame_stride,
                max_sequences=max_items,
                random_clip=training,
                normalize=args.normalize,
            ),
            batch_size=batch_size,
            shuffle=training,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    if dataset == "youtube_vos":
        return build_youtube_vos_dataloader(
            YouTubeVOSDatasetConfig(
                root=data_root,
                split=split,
                image_size=args.image_size,
                frames_per_clip=frames_per_clip,
                frame_stride=args.frame_stride,
                max_sequences=max_items,
                random_clip=training,
                normalize=args.normalize,
            ),
            batch_size=batch_size,
            shuffle=training,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    raise ValueError(f"Unsupported dataset: {dataset}")


def prepare_run_package(
    args: argparse.Namespace,
    dataset_kind: str,
    train_split: Optional[str],
    eval_split: Optional[str],
    model_config: TKDISAConfig,
    dynamic_config: TKDISADynamicConfig,
    device: str,
) -> tuple[str, dict[str, Path]]:
    run_name = args.run_name or timestamped_run_name(f"{args.dataset}_{args.mode}_{dataset_kind}")
    paths = ensure_run_package(args.output_root, args.dataset, run_name)
    data_root = (args.data_root or DEFAULT_DATA_ROOTS[args.dataset]).expanduser().resolve()

    save_summary(paths["meta"] / "args.json", vars(args))
    save_summary(
        paths["meta"] / "config.json",
        {
            "model": asdict(model_config),
            "dynamic": asdict(dynamic_config),
            "dataset": {
                "name": args.dataset,
                "kind": dataset_kind,
                "root": data_root,
                "train_split": train_split,
                "eval_split": eval_split,
                "image_size": args.image_size,
                "max_items": args.max_items,
                "max_eval_items": args.max_eval_items,
                "frames_per_clip": args.frames_per_clip,
                "frame_stride": args.frame_stride,
                "normalize": args.normalize,
            },
        },
    )
    save_summary(paths["meta"] / "runtime.json", collect_runtime_info(device))
    save_summary(
        paths["meta"] / "status.json",
        {
            "state": "running",
            "run_name": run_name,
            "mode": args.mode,
            "dataset": args.dataset,
            "dataset_kind": dataset_kind,
        },
    )
    return run_name, paths


def write_run_summary(
    args: argparse.Namespace,
    run_name: str,
    paths: dict[str, Path],
    dataset_kind: str,
    train_split: Optional[str],
    eval_split: Optional[str],
    train_history: list[dict[str, Any]],
    validation_history: list[dict[str, Any]],
    evaluation_history: list[dict[str, Any]],
    best_metric: Optional[float],
    checkpoint_loaded: Optional[Path],
    state: str,
) -> None:
    save_summary(
        paths["meta"] / "summary.json",
        {
            "run_name": run_name,
            "mode": args.mode,
            "dataset": args.dataset,
            "dataset_kind": dataset_kind,
            "data_root": (args.data_root or DEFAULT_DATA_ROOTS[args.dataset]).expanduser().resolve(),
            "train_split": train_split,
            "eval_split": eval_split,
            "best_metric": best_metric,
            "checkpoint_loaded": checkpoint_loaded,
            "resume": args.resume,
            "train_records": len(train_history),
            "validation_records": len(validation_history),
            "evaluation_records": len(evaluation_history),
            "last_train": train_history[-1] if train_history else None,
            "last_validation": validation_history[-1] if validation_history else None,
            "last_evaluation": evaluation_history[-1] if evaluation_history else None,
            "artifacts": {
                "meta": paths["meta"],
                "ckpt": paths["ckpt"],
                "logs": paths["logs"],
                "curves": paths["curves"],
                "validation_outputs": paths["validation"],
                "evaluation_outputs": paths["evaluation"],
            },
        },
    )
    save_summary(
        paths["meta"] / "status.json",
        {
            "state": state,
            "run_name": run_name,
            "mode": args.mode,
            "dataset": args.dataset,
            "dataset_kind": dataset_kind,
        },
    )


def maybe_load_checkpoint(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> tuple[int, int, int, Optional[float], Optional[Path]]:
    start_epoch = 0
    start_batch_in_epoch = 0
    global_step = 0
    best_metric: Optional[float] = None
    checkpoint_loaded: Optional[Path] = None

    if args.checkpoint is None:
        return start_epoch, start_batch_in_epoch, global_step, best_metric, checkpoint_loaded

    checkpoint_path = args.checkpoint.expanduser().resolve()
    payload = load_checkpoint(
        checkpoint_path,
        model,
        optimizer=optimizer if args.resume else None,
        map_location=device,
    )
    checkpoint_loaded = checkpoint_path
    if args.resume:
        start_epoch = int(payload.get("epoch", 0))
        global_step = int(payload.get("step", 0))
        best_metric = payload.get("best_metric")
        extra = payload.get("extra") or {}
        start_batch_in_epoch = int(extra.get("batch_in_epoch", 0))
    return start_epoch, start_batch_in_epoch, global_step, best_metric, checkpoint_loaded


def build_evaluation_record(
    args: argparse.Namespace,
    split: str,
    epoch: int,
    global_step: int,
    record_type: str,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "mode": record_type,
        "split": split,
        "epoch": epoch,
        "global_step": global_step,
        **metrics,
    }


def evaluate_once(
    args: argparse.Namespace,
    dataset_kind: str,
    model: torch.nn.Module,
    loss_module: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    output_dir: Path,
) -> dict[str, float]:
    if dataset_kind == "static":
        return evaluate_static(
            model=model,
            loss_module=loss_module,
            dataloader=dataloader,
            device=device,
            use_amp=not args.disable_amp and device.type == "cuda",
            amp_dtype=torch.float16 if args.amp_dtype.lower() in {"float16", "fp16"} else (
                torch.bfloat16 if args.amp_dtype.lower() in {"bfloat16", "bf16"} else torch.float32
            ),
            visualization_dir=output_dir,
            max_visualizations=args.visualize_samples,
        )
    return evaluate_dynamic(
        model=model,
        loss_module=loss_module,
        dataloader=dataloader,
        device=device,
        use_amp=not args.disable_amp and device.type == "cuda",
        amp_dtype=torch.float16 if args.amp_dtype.lower() in {"float16", "fp16"} else (
            torch.bfloat16 if args.amp_dtype.lower() in {"bfloat16", "bf16"} else torch.float32
        ),
        visualization_dir=output_dir,
        max_visualizations=args.visualize_samples,
    )


def run_train(args: argparse.Namespace, dataset_kind: str, device: torch.device) -> Path:
    model_config = build_model_config(args)
    dynamic_config = build_dynamic_config(args)
    train_split = normalize_split(args.train_split) or DEFAULT_SPLITS[args.dataset]["train"]
    eval_split = normalize_split(args.eval_split)
    if eval_split is None:
        eval_split = DEFAULT_SPLITS[args.dataset]["eval"]

    run_name, paths = prepare_run_package(
        args=args,
        dataset_kind=dataset_kind,
        train_split=train_split,
        eval_split=eval_split,
        model_config=model_config,
        dynamic_config=dynamic_config,
        device=str(device),
    )

    model = build_model(use_real_dino=args.use_real_dino, config=model_config)
    model.to(device)
    if dataset_kind == "static":
        train_loss_module = TKDISALoss(model_config.loss)
        training_pipeline = TKDISATrainingPipeline(
            model=model,
            config=model_config.train,
            loss_module=train_loss_module,
            device=str(device),
        )
        evaluation_loss_module = TKDISALoss(model_config.loss).to(device)
    else:
        train_loss_module = TKDISADynamicLoss(frame_loss=TKDISALoss(model_config.loss), config=dynamic_config.loss)
        training_pipeline = TKDISADynamicTrainingPipeline(
            model=model,
            dynamic_config=dynamic_config,
            config=model_config.train,
            loss_module=train_loss_module,
            device=str(device),
        )
        evaluation_loss_module = TKDISADynamicLoss(
            frame_loss=TKDISALoss(model_config.loss),
            config=dynamic_config.loss,
        ).to(device)

    start_epoch, start_batch_in_epoch, global_step, best_metric, checkpoint_loaded = maybe_load_checkpoint(
        args=args,
        model=model,
        optimizer=training_pipeline.optimizer,
        device=device,
    )

    train_loader = build_dataloader(args, split=train_split, device=device, training=True)
    eval_loader = build_dataloader(args, split=eval_split, device=device, training=False) if eval_split else None

    train_history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    evaluation_history: list[dict[str, Any]] = []
    last_validation_step = -1
    last_train_record: Optional[dict[str, Any]] = None
    stop_training = args.steps > 0 and global_step >= args.steps

    for epoch in range(start_epoch, args.epochs):
        if stop_training:
            break
        set_epoch_seed(args.seed, epoch)
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < start_batch_in_epoch:
                continue

            if dataset_kind == "static":
                result = training_pipeline.train_step(batch)
            else:
                result = training_pipeline.train_video_step(batch)

            global_step += 1
            last_train_record = {
                "dataset": args.dataset,
                "mode": "train",
                "split": train_split,
                "epoch": epoch,
                "step": batch_index,
                "global_step": global_step,
                "loss": result.loss,
                "learning_rate": training_pipeline.optimizer.param_groups[0]["lr"],
                **result.metrics,
            }
            train_history.append(last_train_record)
            append_jsonl(paths["logs"] / "train.jsonl", last_train_record)

            if eval_loader is None:
                candidate_metric = float(last_train_record["loss"])
                if best_metric is None or candidate_metric < best_metric:
                    best_metric = candidate_metric
                    save_checkpoint(
                        paths["ckpt"] / "best.pt",
                        model=model,
                        optimizer=training_pipeline.optimizer,
                        epoch=epoch,
                        step=global_step,
                        best_metric=best_metric,
                        extra={"batch_in_epoch": batch_index + 1},
                    )

            if eval_loader is not None and args.val_every_steps > 0 and global_step % args.val_every_steps == 0:
                validation_dir = paths["validation"] / f"step_{global_step:07d}"
                validation_metrics = evaluate_once(
                    args=args,
                    dataset_kind=dataset_kind,
                    model=model,
                    loss_module=evaluation_loss_module,
                    dataloader=eval_loader,
                    device=device,
                    output_dir=validation_dir,
                )
                validation_record = build_evaluation_record(
                    args=args,
                    split=eval_split,
                    epoch=epoch,
                    global_step=global_step,
                    record_type="validation",
                    metrics=validation_metrics,
                )
                validation_history.append(validation_record)
                append_jsonl(paths["logs"] / "validation.jsonl", validation_record)
                save_metric_curves(train_history, validation_history, paths["curves"])
                last_validation_step = global_step
                if best_metric is None or validation_record["loss"] < best_metric:
                    best_metric = float(validation_record["loss"])
                    save_checkpoint(
                        paths["ckpt"] / "best.pt",
                        model=model,
                        optimizer=training_pipeline.optimizer,
                        epoch=epoch,
                        step=global_step,
                        best_metric=best_metric,
                        extra={"batch_in_epoch": batch_index + 1},
                    )
                write_run_summary(
                    args=args,
                    run_name=run_name,
                    paths=paths,
                    dataset_kind=dataset_kind,
                    train_split=train_split,
                    eval_split=eval_split,
                    train_history=train_history,
                    validation_history=validation_history,
                    evaluation_history=evaluation_history,
                    best_metric=best_metric,
                    checkpoint_loaded=checkpoint_loaded,
                    state="running",
                )

            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                step_checkpoint = paths["ckpt"] / f"step_{global_step:07d}.pt"
                save_checkpoint(
                    step_checkpoint,
                    model=model,
                    optimizer=training_pipeline.optimizer,
                    epoch=epoch,
                    step=global_step,
                    best_metric=best_metric,
                    extra={"batch_in_epoch": batch_index + 1},
                )
                save_checkpoint(
                    paths["ckpt"] / "last.pt",
                    model=model,
                    optimizer=training_pipeline.optimizer,
                    epoch=epoch,
                    step=global_step,
                    best_metric=best_metric,
                    extra={"batch_in_epoch": batch_index + 1},
                )

            if args.steps > 0 and global_step >= args.steps:
                stop_training = True
                break

        save_checkpoint(
            paths["ckpt"] / "last.pt",
            model=model,
            optimizer=training_pipeline.optimizer,
            epoch=epoch + 1,
            step=global_step,
            best_metric=best_metric,
            extra={"batch_in_epoch": 0},
        )

    if eval_loader is not None and global_step > 0 and last_validation_step != global_step:
        validation_dir = paths["validation"] / f"step_{global_step:07d}"
        validation_metrics = evaluate_once(
            args=args,
            dataset_kind=dataset_kind,
            model=model,
            loss_module=evaluation_loss_module,
            dataloader=eval_loader,
            device=device,
            output_dir=validation_dir,
        )
        validation_record = build_evaluation_record(
            args=args,
            split=eval_split,
            epoch=max(args.epochs - 1, 0),
            global_step=global_step,
            record_type="validation",
            metrics=validation_metrics,
        )
        validation_history.append(validation_record)
        append_jsonl(paths["logs"] / "validation.jsonl", validation_record)
        save_metric_curves(train_history, validation_history, paths["curves"])
        if best_metric is None or validation_record["loss"] < best_metric:
            best_metric = float(validation_record["loss"])
            save_checkpoint(
                paths["ckpt"] / "best.pt",
                model=model,
                optimizer=training_pipeline.optimizer,
                epoch=args.epochs,
                step=global_step,
                best_metric=best_metric,
                extra={"batch_in_epoch": 0},
            )

    if eval_loader is not None:
        evaluation_dir = paths["evaluation"] / "final"
        evaluation_metrics = evaluate_once(
            args=args,
            dataset_kind=dataset_kind,
            model=model,
            loss_module=evaluation_loss_module,
            dataloader=eval_loader,
            device=device,
            output_dir=evaluation_dir,
        )
        evaluation_record = build_evaluation_record(
            args=args,
            split=eval_split,
            epoch=max(args.epochs - 1, 0),
            global_step=global_step,
            record_type="evaluation",
            metrics=evaluation_metrics,
        )
        evaluation_history.append(evaluation_record)
        append_jsonl(paths["logs"] / "evaluation.jsonl", evaluation_record)

    if best_metric is None and last_train_record is not None:
        best_metric = float(last_train_record["loss"])
        save_checkpoint(
            paths["ckpt"] / "best.pt",
            model=model,
            optimizer=training_pipeline.optimizer,
            epoch=args.epochs,
            step=global_step,
            best_metric=best_metric,
            extra={"batch_in_epoch": 0},
        )

    save_metric_curves(train_history, validation_history, paths["curves"])
    write_run_summary(
        args=args,
        run_name=run_name,
        paths=paths,
        dataset_kind=dataset_kind,
        train_split=train_split,
        eval_split=eval_split,
        train_history=train_history,
        validation_history=validation_history,
        evaluation_history=evaluation_history,
        best_metric=best_metric,
        checkpoint_loaded=checkpoint_loaded,
        state="completed",
    )
    return paths["run"]


def run_eval(args: argparse.Namespace, dataset_kind: str, device: torch.device) -> Path:
    model_config = build_model_config(args)
    dynamic_config = build_dynamic_config(args)
    train_split = normalize_split(args.train_split) or DEFAULT_SPLITS[args.dataset]["train"]
    eval_split = normalize_split(args.eval_split) or DEFAULT_SPLITS[args.dataset]["eval"]

    run_name, paths = prepare_run_package(
        args=args,
        dataset_kind=dataset_kind,
        train_split=train_split,
        eval_split=eval_split,
        model_config=model_config,
        dynamic_config=dynamic_config,
        device=str(device),
    )

    model = build_model(use_real_dino=args.use_real_dino, config=model_config)
    model.to(device)
    if dataset_kind == "static":
        evaluation_loss_module = TKDISALoss(model_config.loss).to(device)
    else:
        evaluation_loss_module = TKDISADynamicLoss(
            frame_loss=TKDISALoss(model_config.loss),
            config=dynamic_config.loss,
        ).to(device)

    _, _, _, _, checkpoint_loaded = maybe_load_checkpoint(
        args=args,
        model=model,
        optimizer=None,
        device=device,
    )
    eval_loader = build_dataloader(args, split=eval_split, device=device, training=False)

    evaluation_metrics = evaluate_once(
        args=args,
        dataset_kind=dataset_kind,
        model=model,
        loss_module=evaluation_loss_module,
        dataloader=eval_loader,
        device=device,
        output_dir=paths["evaluation"] / "standalone",
    )
    evaluation_record = build_evaluation_record(
        args=args,
        split=eval_split,
        epoch=0,
        global_step=0,
        record_type="evaluation",
        metrics=evaluation_metrics,
    )
    append_jsonl(paths["logs"] / "evaluation.jsonl", evaluation_record)
    write_run_summary(
        args=args,
        run_name=run_name,
        paths=paths,
        dataset_kind=dataset_kind,
        train_split=train_split,
        eval_split=eval_split,
        train_history=[],
        validation_history=[],
        evaluation_history=[evaluation_record],
        best_metric=float(evaluation_record["loss"]),
        checkpoint_loaded=checkpoint_loaded,
        state="completed",
    )
    return paths["run"]


def main(
    argv: Optional[list[str]] = None,
    default_mode: Optional[str] = None,
    default_dataset: Optional[str] = None,
) -> None:
    args = parse_args(argv=argv, default_mode=default_mode, default_dataset=default_dataset)
    set_seed(args.seed)
    device_name = resolve_device(args.device)
    device = torch.device(device_name)
    dataset_kind = DATASET_KINDS[args.dataset]

    if args.mode == "train":
        run_dir = run_train(args, dataset_kind=dataset_kind, device=device)
    else:
        run_dir = run_eval(args, dataset_kind=dataset_kind, device=device)
    print(run_dir)


if __name__ == "__main__":
    main()
