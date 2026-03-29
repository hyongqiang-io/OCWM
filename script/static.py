from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW, Optimizer

from module.slot_encoder.config import InferencePipelineConfig, LossConfig, TrainPipelineConfig
from module.slot_encoder.model import TKDISAModel, TKDISAModelOutput

from .common import amp_context, metrics_to_float, resolve_amp_dtype


@dataclass
class LossOutput:
    total_loss: Tensor
    metrics: Dict[str, Tensor]


class TKDISALoss(nn.Module):
    def __init__(self, config: Optional[LossConfig] = None) -> None:
        super().__init__()
        self.config = config or LossConfig()

    def forward(
        self,
        outputs: TKDISAModelOutput,
        position_targets: Optional[Tensor] = None,
        target_features: Optional[Tensor] = None,
    ) -> LossOutput:
        target_features = target_features if target_features is not None else outputs.features.detach()
        reconstruction_loss = F.mse_loss(outputs.reconstructed_features, target_features)

        if position_targets is None and self.config.use_pseudo_position_targets:
            position_targets = self.compute_attention_centroids(outputs.position_attention).detach()
        elif position_targets is not None and position_targets.shape[-1] > 2:
            position_targets = position_targets[..., :2]

        if position_targets is None:
            position_loss = outputs.position_slots.new_zeros(())
        else:
            position_targets = position_targets.to(outputs.position_slots.device, outputs.position_slots.dtype)
            position_loss = F.mse_loss(outputs.position_slots[..., :2], position_targets)

        vq_loss = outputs.commitment_loss
        total_loss = (
            self.config.reconstruction_weight * reconstruction_loss
            + self.config.vq_weight * vq_loss
            + self.config.position_weight * position_loss
        )

        metrics = {
            "reconstruction_loss": reconstruction_loss.detach(),
            "vq_loss": vq_loss.detach(),
            "position_loss": position_loss.detach(),
            "total_loss": total_loss.detach(),
        }
        return LossOutput(total_loss=total_loss, metrics=metrics)

    def compute_attention_centroids(self, attention_masks: Tensor) -> Tensor:
        batch_size, num_slots, height, width = attention_masks.shape
        weights = attention_masks.flatten(2)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

        y_coords = torch.linspace(0.0, 1.0, height, device=attention_masks.device, dtype=attention_masks.dtype)
        x_coords = torch.linspace(0.0, 1.0, width, device=attention_masks.device, dtype=attention_masks.dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1).reshape(1, 1, height * width, 2)
        centroids = (weights.unsqueeze(-1) * grid).sum(dim=2)
        return centroids.reshape(batch_size, num_slots, 2)


@dataclass
class TrainStepResult:
    loss: float
    metrics: Dict[str, float]


@dataclass
class InferenceImageResult:
    detections: list[list[dict]]
    reconstructed_features: Optional[Tensor] = None
    decoder_masks: Optional[Tensor] = None
    background_mask: Optional[Tensor] = None


@dataclass
class StaticEvaluationResult:
    num_batches: int
    loss: float
    metrics: Dict[str, float]


class TKDISATrainingPipeline:
    def __init__(
        self,
        model: TKDISAModel,
        config: Optional[TrainPipelineConfig] = None,
        loss_module: Optional[TKDISALoss] = None,
        optimizer: Optional[Optimizer] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.config = config or model.config.train or TrainPipelineConfig()
        self.loss_module = loss_module or TKDISALoss(model.config.loss)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.loss_module.to(self.device)
        self.optimizer = optimizer or AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.use_amp = self.config.amp and self.device.type == "cuda"
        self.amp_dtype = resolve_amp_dtype(self.config.amp_dtype)
        self.scaler = GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)

    def train_step(self, batch: Dict[str, Tensor]) -> TrainStepResult:
        images = self._get_images(batch)
        position_targets = self._maybe_to_device(batch.get("position_targets"))
        target_features = self._maybe_to_device(batch.get("target_features"))

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with amp_context(self.use_amp, self.amp_dtype):
            outputs = self.model(images)
            loss_output = self.loss_module(
                outputs,
                position_targets=position_targets,
                target_features=target_features,
            )
        self._optimize(loss_output.total_loss)
        return TrainStepResult(
            loss=float(loss_output.total_loss.detach().cpu().item()),
            metrics=metrics_to_float(loss_output.metrics),
        )

    def _get_images(self, batch: Dict[str, Tensor]) -> Tensor:
        images = batch.get("images")
        if images is None:
            raise KeyError("Batch must include 'images' for static slot encoder training.")
        if not images.ndim == 4:
            raise ValueError("Static slot encoder expects image batches with shape [B, C, H, W].")
        return images.to(self.device, non_blocking=self.device.type == "cuda")

    def _maybe_to_device(self, tensor: Optional[Tensor]) -> Optional[Tensor]:
        if tensor is None:
            return None
        return tensor.to(self.device, non_blocking=self.device.type == "cuda")

    def _optimize(self, loss: Tensor) -> None:
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self._clip_gradients()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self._clip_gradients()
            self.optimizer.step()

    def _clip_gradients(self) -> None:
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)


class TKDISAInferencePipeline:
    def __init__(
        self,
        model: TKDISAModel,
        config: Optional[InferencePipelineConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.config = config or model.config.inference or InferencePipelineConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.use_amp = self.config.amp and self.device.type == "cuda"
        self.amp_dtype = resolve_amp_dtype(self.config.amp_dtype)

    @torch.no_grad()
    def infer_image(
        self,
        images: Tensor,
        confidence_threshold: Optional[float] = None,
        return_reconstruction: Optional[bool] = None,
    ) -> InferenceImageResult:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if not images.ndim == 4:
            raise ValueError("infer_image expects [C, H, W] or [B, C, H, W] tensors.")

        threshold = self.config.confidence_threshold if confidence_threshold is None else confidence_threshold
        reconstruct = self.config.return_reconstruction if return_reconstruction is None else return_reconstruction

        self.model.eval()
        with amp_context(self.use_amp, self.amp_dtype):
            outputs = self.model(images.to(self.device, non_blocking=self.device.type == "cuda"))
        detections = self.model.extract_detections(outputs, confidence_threshold=threshold)

        return InferenceImageResult(
            detections=detections,
            reconstructed_features=outputs.reconstructed_features.detach().cpu() if reconstruct else None,
            decoder_masks=outputs.decoder_masks.detach().cpu() if reconstruct else None,
            background_mask=(
                outputs.background_mask.detach().cpu()
                if reconstruct and outputs.background_mask is not None
                else None
            ),
        )


class TKDISAEvaluationPipeline:
    def __init__(
        self,
        model: TKDISAModel,
        config: Optional[InferencePipelineConfig] = None,
        loss_module: Optional[TKDISALoss] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.config = config or model.config.inference or InferencePipelineConfig()
        self.loss_module = loss_module or TKDISALoss(model.config.loss)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.loss_module.to(self.device)
        self.use_amp = self.config.amp and self.device.type == "cuda"
        self.amp_dtype = resolve_amp_dtype(self.config.amp_dtype)

    @torch.no_grad()
    def evaluate_batches(self, batches: Iterable[Dict[str, Tensor]]) -> StaticEvaluationResult:
        self.model.eval()
        metric_totals: Dict[str, float] = {}
        total_loss = 0.0
        batch_count = 0

        for batch in batches:
            images = self._get_images(batch)
            position_targets = self._maybe_to_device(batch.get("position_targets"))
            target_features = self._maybe_to_device(batch.get("target_features"))
            with amp_context(self.use_amp, self.amp_dtype):
                outputs = self.model(images)
                loss_output = self.loss_module(
                    outputs,
                    position_targets=position_targets,
                    target_features=target_features,
                )

            scalar_metrics = metrics_to_float(loss_output.metrics)
            total_loss += float(loss_output.total_loss.detach().cpu().item())
            for name, value in scalar_metrics.items():
                metric_totals[name] = metric_totals.get(name, 0.0) + value
            batch_count += 1

        if batch_count == 0:
            raise ValueError("Static evaluation requires at least one batch.")

        averaged_metrics = {name: value / batch_count for name, value in metric_totals.items()}
        return StaticEvaluationResult(
            num_batches=batch_count,
            loss=total_loss / batch_count,
            metrics=averaged_metrics,
        )

    def _get_images(self, batch: Dict[str, Tensor]) -> Tensor:
        images = batch.get("images")
        if images is None:
            raise KeyError("Batch must include 'images' for static slot encoder evaluation.")
        if not images.ndim == 4:
            raise ValueError("Static evaluation expects image batches with shape [B, C, H, W].")
        return images.to(self.device, non_blocking=self.device.type == "cuda")

    def _maybe_to_device(self, tensor: Optional[Tensor]) -> Optional[Tensor]:
        if tensor is None:
            return None
        return tensor.to(self.device, non_blocking=self.device.type == "cuda")
