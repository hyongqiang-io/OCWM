from __future__ import annotations

import json
import math
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


_IMAGE_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
from torch import Tensor, nn


_PLOT_COLORS = [
    (30, 136, 229),
    (67, 160, 71),
    (239, 83, 80),
    (251, 192, 45),
    (141, 110, 99),
    (126, 87, 194),
]
_MASK_COLORS = [
    (244, 67, 54),
    (33, 150, 243),
    (76, 175, 80),
    (255, 152, 0),
    (156, 39, 176),
    (0, 150, 136),
    (121, 85, 72),
    (63, 81, 181),
    (205, 220, 57),
    (255, 87, 34),
]
_RESERVED_LOG_KEYS = {"epoch", "step", "global_step", "num_batches", "mode", "split", "dataset"}


def _amp_context(use_amp: bool, amp_dtype: torch.dtype):
    if not use_amp:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    return value


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(to_serializable(dict(record)), ensure_ascii=False) + "\n")


def ensure_run_package(results_root: Path, dataset_name: str, run_name: str) -> dict[str, Path]:
    run_root = results_root / dataset_name / run_name
    paths = {
        "run": run_root,
        "meta": run_root / "meta",
        "ckpt": run_root / "ckpt",
        "logs": run_root / "logs",
        "curves": run_root / "curves",
        "outputs": run_root / "outputs",
        "validation": run_root / "outputs" / "validation",
        "evaluation": run_root / "outputs" / "evaluation",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    step: int,
    best_metric: Optional[float],
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "extra": to_serializable(dict(extra or {})),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def save_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_serializable(dict(summary)), file, indent=2, ensure_ascii=False)


def _normalize_tensor(tensor: Tensor) -> Tensor:
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] >= 3:
        tensor = tensor[:3]
    else:
        tensor = torch.cat([tensor, tensor[:1].repeat(3 - tensor.shape[0], 1, 1)], dim=0)

    min_value = tensor.amin(dim=(1, 2), keepdim=True)
    max_value = tensor.amax(dim=(1, 2), keepdim=True)
    scale = (max_value - min_value).clamp(min=1e-6)

    if tensor.min().item() >= 0.0 and tensor.max().item() <= 1.0:
        return tensor.clamp(0.0, 1.0)
    return ((tensor - min_value) / scale).clamp(0.0, 1.0)


def tensor_to_pil_image(tensor: Tensor, image_size: Optional[tuple[int, int]] = None) -> Image.Image:
    tensor = _normalize_tensor(tensor)
    image = (tensor.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    pil_image = Image.fromarray(image)
    if image_size is not None:
        pil_image = pil_image.resize(image_size, resample=_IMAGE_BILINEAR)
    return pil_image


def feature_map_to_pil_image(feature_map: Tensor, image_size: Optional[tuple[int, int]] = None) -> Image.Image:
    return tensor_to_pil_image(feature_map, image_size=image_size)


def mask_map_to_pil_image(
    masks: Tensor,
    image_size: Optional[tuple[int, int]] = None,
    base_image: Optional[Tensor] = None,
) -> Image.Image:
    mask_tensor = masks.detach().float().cpu()
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    height, width = mask_tensor.shape[-2:]
    color_tensor = torch.zeros(3, height, width)

    for slot_index in range(mask_tensor.shape[0]):
        color = torch.tensor(_MASK_COLORS[slot_index % len(_MASK_COLORS)], dtype=torch.float32).view(3, 1, 1) / 255.0
        color_tensor = color_tensor + mask_tensor[slot_index].clamp(0.0, 1.0).unsqueeze(0) * color

    color_tensor = color_tensor.clamp(0.0, 1.0)
    if base_image is not None:
        base_tensor = _normalize_tensor(base_image)
        if base_tensor.shape[-2:] != color_tensor.shape[-2:]:
            color_tensor = F.interpolate(
                color_tensor.unsqueeze(0),
                size=base_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        color_tensor = (0.6 * base_tensor + 0.4 * color_tensor).clamp(0.0, 1.0)
    return tensor_to_pil_image(color_tensor, image_size=image_size)


def _label_image(image: Image.Image, label: str) -> Image.Image:
    labeled = image.copy()
    draw = ImageDraw.Draw(labeled)
    draw.rectangle((0, 0, labeled.width, 18), fill=(255, 255, 255))
    draw.text((4, 2), label, fill=(0, 0, 0))
    return labeled


def make_horizontal_panel(images: Sequence[Image.Image], labels: Sequence[str]) -> Image.Image:
    if len(images) == 0:
        raise ValueError("Panel requires at least one image.")
    labeled_images = [_label_image(image, label) for image, label in zip(images, labels)]
    width = sum(image.width for image in labeled_images)
    height = max(image.height for image in labeled_images)
    panel = Image.new("RGB", (width, height), color=(255, 255, 255))
    x_offset = 0
    for image in labeled_images:
        panel.paste(image, (x_offset, 0))
        x_offset += image.width
    return panel


def make_vertical_stack(images: Sequence[Image.Image]) -> Image.Image:
    if len(images) == 0:
        raise ValueError("Stack requires at least one image.")
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    stacked = Image.new("RGB", (width, height), color=(255, 255, 255))
    y_offset = 0
    for image in images:
        stacked.paste(image, (0, y_offset))
        y_offset += image.height
    return stacked


def save_static_visualizations(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    output_dir: Path,
    max_samples: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch.get("images")
            if images is None:
                continue
            images_device = images.to(device, non_blocking=device.type == "cuda")
            with _amp_context(use_amp, amp_dtype):
                outputs = model(images_device)
            image_size = (images.shape[-1], images.shape[-2])
            for sample_index in range(images.shape[0]):
                panel = make_horizontal_panel(
                    images=[
                        tensor_to_pil_image(images[sample_index], image_size=image_size),
                        feature_map_to_pil_image(outputs.reconstructed_features[sample_index], image_size=image_size),
                        mask_map_to_pil_image(
                            outputs.decoder_masks[sample_index],
                            image_size=image_size,
                            base_image=images[sample_index],
                        ),
                    ],
                    labels=["ground_truth", "reconstruction_proxy", "slot_masks"],
                )
                panel.save(output_dir / f"sample_{saved:04d}.png")
                saved += 1
                if saved >= max_samples:
                    return


def save_dynamic_visualizations(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    output_dir: Path,
    max_samples: int,
    max_frames: int = 4,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            frames = batch.get("frames")
            if frames is None:
                continue
            frames_device = frames.to(device, non_blocking=device.type == "cuda")
            with _amp_context(use_amp, amp_dtype):
                outputs_per_frame = [model(frames_device[:, frame_index]) for frame_index in range(frames_device.shape[1])]
            for sample_index in range(frames.shape[0]):
                rows = []
                image_size = (frames.shape[-1], frames.shape[-2])
                num_frames = min(frames.shape[1], max_frames)
                for frame_index in range(num_frames):
                    rows.append(
                        make_horizontal_panel(
                            images=[
                                tensor_to_pil_image(frames[sample_index, frame_index], image_size=image_size),
                                feature_map_to_pil_image(
                                    outputs_per_frame[frame_index].reconstructed_features[sample_index],
                                    image_size=image_size,
                                ),
                                mask_map_to_pil_image(
                                    outputs_per_frame[frame_index].decoder_masks[sample_index],
                                    image_size=image_size,
                                    base_image=frames[sample_index, frame_index],
                                ),
                            ],
                            labels=[
                                f"frame_{frame_index:02d}_gt",
                                f"frame_{frame_index:02d}_reconstruction_proxy",
                                f"frame_{frame_index:02d}_slot_masks",
                            ],
                        )
                    )
                make_vertical_stack(rows).save(output_dir / f"sample_{saved:04d}.png")
                saved += 1
                if saved >= max_samples:
                    return


def evaluate_static(
    model: nn.Module,
    loss_module: nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    visualization_dir: Optional[Path] = None,
    max_visualizations: int = 0,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    total_loss = 0.0
    batch_count = 0

    model.eval()
    loss_module.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch.get("images")
            if images is None:
                raise KeyError("Static evaluation expects 'images' in the batch.")
            images = images.to(device, non_blocking=device.type == "cuda")
            position_targets = batch.get("position_targets")
            target_features = batch.get("target_features")
            if position_targets is not None:
                position_targets = position_targets.to(device, non_blocking=device.type == "cuda")
            if target_features is not None:
                target_features = target_features.to(device, non_blocking=device.type == "cuda")

            with _amp_context(use_amp, amp_dtype):
                outputs = model(images)
                loss_output = loss_module(
                    outputs,
                    position_targets=position_targets,
                    target_features=target_features,
                )

            total_loss += float(loss_output.total_loss.detach().cpu().item())
            for name, value in loss_output.metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu().item())
            batch_count += 1

    if batch_count == 0:
        raise ValueError("Static evaluation requires at least one batch.")

    if visualization_dir is not None and max_visualizations > 0:
        save_static_visualizations(
            model=model,
            dataloader=dataloader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            output_dir=visualization_dir,
            max_samples=max_visualizations,
        )

    metrics = {name: value / batch_count for name, value in totals.items()}
    metrics["loss"] = total_loss / batch_count
    metrics["num_batches"] = float(batch_count)
    return metrics


def evaluate_dynamic(
    model: nn.Module,
    loss_module: Any,
    dataloader: Iterable[Mapping[str, Any]],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    visualization_dir: Optional[Path] = None,
    max_visualizations: int = 0,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    total_loss = 0.0
    batch_count = 0

    model.eval()
    loss_module.eval()
    with torch.no_grad():
        for batch in dataloader:
            frames = batch.get("frames")
            if frames is None:
                raise KeyError("Dynamic evaluation expects 'frames' in the batch.")
            frames = frames.to(device, non_blocking=device.type == "cuda")
            position_targets = batch.get("position_targets")
            target_features = batch.get("target_features")
            track_ids = batch.get("track_ids")
            if position_targets is not None:
                position_targets = position_targets.to(device, non_blocking=device.type == "cuda")
            if target_features is not None:
                target_features = target_features.to(device, non_blocking=device.type == "cuda")
            if track_ids is not None:
                track_ids = track_ids.to(device, non_blocking=device.type == "cuda")

            with _amp_context(use_amp, amp_dtype):
                outputs = [model(frames[:, frame_index]) for frame_index in range(frames.shape[1])]
                loss_output = loss_module.compute_video_loss(
                    outputs,
                    position_targets=position_targets,
                    target_features=target_features,
                    track_ids=track_ids,
                )

            total_loss += float(loss_output.total_loss.detach().cpu().item())
            for name, value in loss_output.metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu().item())
            batch_count += 1

    if batch_count == 0:
        raise ValueError("Dynamic evaluation requires at least one batch.")

    if visualization_dir is not None and max_visualizations > 0:
        save_dynamic_visualizations(
            model=model,
            dataloader=dataloader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            output_dir=visualization_dir,
            max_samples=max_visualizations,
        )

    metrics = {name: value / batch_count for name, value in totals.items()}
    metrics["loss"] = total_loss / batch_count
    metrics["num_batches"] = float(batch_count)
    return metrics


def _collect_metric_names(*histories: Sequence[Mapping[str, Any]]) -> list[str]:
    metric_names = set()
    for history in histories:
        for record in history:
            for key, value in record.items():
                if key in _RESERVED_LOG_KEYS:
                    continue
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    metric_names.add(key)
    return sorted(metric_names)


def _extract_series(records: Sequence[Mapping[str, Any]], metric_name: str) -> list[tuple[float, float]]:
    series = []
    for record in records:
        if metric_name not in record:
            continue
        x_value = record.get("global_step", record.get("step", record.get("epoch", 0)))
        value = record[metric_name]
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        series.append((float(x_value), float(value)))
    return series


def draw_curve(path: Path, title: str, series_map: Mapping[str, Sequence[tuple[float, float]]]) -> None:
    width, height = 960, 540
    margin_left, margin_top, margin_right, margin_bottom = 70, 40, 20, 60
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    all_points = [point for series in series_map.values() for point in series]
    if len(all_points) == 0:
        draw.text((20, 20), f"{title}: no data", fill=(0, 0, 0))
        image.save(path)
        return

    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0

    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom

    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(0, 0, 0), width=1)
    draw.text((plot_left, 10), title, fill=(0, 0, 0))

    for grid_index in range(1, 5):
        y = plot_top + (plot_bottom - plot_top) * grid_index / 5.0
        draw.line((plot_left, y, plot_right, y), fill=(220, 220, 220), width=1)

    def project(point: tuple[float, float]) -> tuple[int, int]:
        x_value, y_value = point
        x = plot_left + (x_value - x_min) / (x_max - x_min) * (plot_right - plot_left)
        y = plot_bottom - (y_value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
        return int(x), int(y)

    for color_index, (label, series) in enumerate(series_map.items()):
        if len(series) == 0:
            continue
        color = _PLOT_COLORS[color_index % len(_PLOT_COLORS)]
        projected = [project(point) for point in series]
        if len(projected) == 1:
            x_value, y_value = projected[0]
            draw.ellipse((x_value - 2, y_value - 2, x_value + 2, y_value + 2), fill=color)
        else:
            draw.line(projected, fill=color, width=3)
        legend_y = plot_top + color_index * 18
        draw.rectangle((plot_right - 170, legend_y, plot_right - 158, legend_y + 12), fill=color)
        draw.text((plot_right - 152, legend_y - 1), label, fill=(0, 0, 0))

    draw.text((plot_left, plot_bottom + 8), f"step: {x_min:.0f} -> {x_max:.0f}", fill=(0, 0, 0))
    draw.text((plot_left + 250, plot_bottom + 8), f"value: {y_min:.4f} -> {y_max:.4f}", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_metric_curves(
    train_history: Sequence[Mapping[str, Any]],
    validation_history: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = _collect_metric_names(train_history, validation_history)
    for metric_name in metric_names:
        series_map = {
            "train": _extract_series(train_history, metric_name),
            "validation": _extract_series(validation_history, metric_name),
        }
        if len(series_map["train"]) == 0 and len(series_map["validation"]) == 0:
            continue
        draw_curve(
            path=output_dir / f"{metric_name}.png",
            title=metric_name,
            series_map=series_map,
        )
