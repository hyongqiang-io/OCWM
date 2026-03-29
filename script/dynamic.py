from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW, Optimizer

from module.dynamic.config import SlotManagerConfig, TKDISADynamicConfig, TemporalLossConfig
from module.dynamic.slot_manager import SlotManager
from module.slot_encoder.config import InferencePipelineConfig, TrainPipelineConfig
from module.slot_encoder.model import TKDISAModel, TKDISAModelOutput

from .common import amp_context, metrics_to_float, resolve_amp_dtype
from .static import LossOutput, TKDISALoss


class TKDISADynamicLoss(nn.Module):
    def __init__(
        self,
        frame_loss: Optional[TKDISALoss] = None,
        config: Optional[TemporalLossConfig] = None,
    ) -> None:
        super().__init__()
        self.frame_loss = frame_loss or TKDISALoss()
        self.config = config or TemporalLossConfig()

    def compute_video_loss(
        self,
        outputs: Sequence[TKDISAModelOutput],
        position_targets: Optional[Tensor] = None,
        target_features: Optional[Tensor] = None,
        track_ids: Optional[Tensor] = None,
    ) -> LossOutput:
        if len(outputs) == 0:
            raise ValueError("Video loss requires at least one frame output.")

        total_loss = outputs[0].position_slots.new_zeros(())
        reconstruction_loss = outputs[0].position_slots.new_zeros(())
        vq_loss = outputs[0].position_slots.new_zeros(())
        position_loss = outputs[0].position_slots.new_zeros(())
        temporal_visual_loss = outputs[0].position_slots.new_zeros(())
        temporal_velocity_loss = outputs[0].position_slots.new_zeros(())
        temporal_code_disagreement = outputs[0].position_slots.new_zeros(())
        temporal_pairs = outputs[0].position_slots.new_zeros(())

        for frame_index, frame_output in enumerate(outputs):
            frame_position_targets = None
            frame_target_features = None

            if position_targets is not None:
                frame_position_targets = position_targets[:, frame_index]
            if target_features is not None:
                frame_target_features = target_features[:, frame_index]

            frame_loss = self.frame_loss(
                frame_output,
                position_targets=frame_position_targets,
                target_features=frame_target_features,
            )
            total_loss = total_loss + frame_loss.total_loss
            reconstruction_loss = reconstruction_loss + frame_loss.metrics["reconstruction_loss"]
            vq_loss = vq_loss + frame_loss.metrics["vq_loss"]
            position_loss = position_loss + frame_loss.metrics["position_loss"]

        frame_count = float(len(outputs))
        reconstruction_loss = reconstruction_loss / frame_count
        vq_loss = vq_loss / frame_count
        position_loss = position_loss / frame_count

        if len(outputs) > 1:
            for frame_index in range(len(outputs) - 1):
                previous_track_ids = track_ids[:, frame_index] if track_ids is not None else None
                current_track_ids = track_ids[:, frame_index + 1] if track_ids is not None else None
                temporal_loss = self.compute_temporal_loss(
                    outputs[frame_index],
                    outputs[frame_index + 1],
                    previous_track_ids=previous_track_ids,
                    current_track_ids=current_track_ids,
                )
                total_loss = total_loss + temporal_loss.total_loss
                temporal_visual_loss = temporal_visual_loss + temporal_loss.metrics["temporal_visual_loss"]
                temporal_velocity_loss = temporal_velocity_loss + temporal_loss.metrics["temporal_velocity_loss"]
                temporal_code_disagreement = (
                    temporal_code_disagreement + temporal_loss.metrics["temporal_code_disagreement"]
                )
                temporal_pairs = temporal_pairs + temporal_loss.metrics["temporal_pairs"]

            pair_frame_count = float(len(outputs) - 1)
            temporal_visual_loss = temporal_visual_loss / pair_frame_count
            temporal_velocity_loss = temporal_velocity_loss / pair_frame_count
            temporal_code_disagreement = temporal_code_disagreement / pair_frame_count
            temporal_pairs = temporal_pairs / pair_frame_count

        metrics = {
            "reconstruction_loss": reconstruction_loss.detach(),
            "vq_loss": vq_loss.detach(),
            "position_loss": position_loss.detach(),
            "temporal_visual_loss": temporal_visual_loss.detach(),
            "temporal_velocity_loss": temporal_velocity_loss.detach(),
            "temporal_code_disagreement": temporal_code_disagreement.detach(),
            "temporal_pairs": temporal_pairs.detach(),
            "total_loss": total_loss.detach(),
        }
        return LossOutput(total_loss=total_loss, metrics=metrics)

    def compute_temporal_loss(
        self,
        previous_outputs: TKDISAModelOutput,
        current_outputs: TKDISAModelOutput,
        previous_track_ids: Optional[Tensor] = None,
        current_track_ids: Optional[Tensor] = None,
    ) -> LossOutput:
        device = previous_outputs.position_slots.device
        temporal_visual_loss = previous_outputs.position_slots.new_zeros(())
        temporal_velocity_loss = previous_outputs.position_slots.new_zeros(())
        temporal_code_disagreement = previous_outputs.position_slots.new_zeros(())
        temporal_pairs = previous_outputs.position_slots.new_zeros(())

        for batch_index in range(previous_outputs.position_slots.shape[0]):
            pairs = self._match_slots(
                previous_outputs.position_slots[batch_index],
                current_outputs.position_slots[batch_index],
                previous_track_ids[batch_index] if previous_track_ids is not None else None,
                current_track_ids[batch_index] if current_track_ids is not None else None,
            )
            if not pairs:
                continue

            prev_indices = torch.tensor([pair[0] for pair in pairs], device=device, dtype=torch.long)
            curr_indices = torch.tensor([pair[1] for pair in pairs], device=device, dtype=torch.long)

            prev_visual = previous_outputs.quantized_visual_slots[batch_index, prev_indices]
            curr_visual = current_outputs.quantized_visual_slots[batch_index, curr_indices]
            prev_velocity = previous_outputs.position_slots[batch_index, prev_indices, 2:]
            curr_velocity = current_outputs.position_slots[batch_index, curr_indices, 2:]
            prev_codes = previous_outputs.visual_code_indices[batch_index, prev_indices]
            curr_codes = current_outputs.visual_code_indices[batch_index, curr_indices]

            temporal_visual_loss = temporal_visual_loss + F.mse_loss(prev_visual, curr_visual)
            temporal_velocity_loss = temporal_velocity_loss + F.mse_loss(prev_velocity, curr_velocity)
            temporal_code_disagreement = temporal_code_disagreement + torch.ne(prev_codes, curr_codes).float().mean()
            temporal_pairs = temporal_pairs + previous_outputs.position_slots.new_tensor(float(len(pairs)))

        matched_batches = max(previous_outputs.position_slots.shape[0], 1)
        temporal_visual_loss = temporal_visual_loss / matched_batches
        temporal_velocity_loss = temporal_velocity_loss / matched_batches
        temporal_code_disagreement = temporal_code_disagreement / matched_batches
        temporal_pairs = temporal_pairs / matched_batches

        total_loss = (
            self.config.temporal_visual_weight * temporal_visual_loss
            + self.config.temporal_velocity_weight * temporal_velocity_loss
        )
        metrics = {
            "temporal_visual_loss": temporal_visual_loss.detach(),
            "temporal_velocity_loss": temporal_velocity_loss.detach(),
            "temporal_code_disagreement": temporal_code_disagreement.detach(),
            "temporal_pairs": temporal_pairs.detach(),
            "total_loss": total_loss.detach(),
        }
        return LossOutput(total_loss=total_loss, metrics=metrics)

    def _match_slots(
        self,
        previous_positions: Tensor,
        current_positions: Tensor,
        previous_track_ids: Optional[Tensor] = None,
        current_track_ids: Optional[Tensor] = None,
    ) -> list[tuple[int, int]]:
        if previous_track_ids is not None and current_track_ids is not None:
            pairs: list[tuple[int, int]] = []
            current_id_to_index = {
                int(slot_id.item()): index
                for index, slot_id in enumerate(current_track_ids)
                if int(slot_id.item()) >= 0
            }
            for previous_index, slot_id in enumerate(previous_track_ids):
                slot_id_value = int(slot_id.item())
                if slot_id_value < 0 or slot_id_value not in current_id_to_index:
                    continue
                pairs.append((previous_index, current_id_to_index[slot_id_value]))
            return pairs

        distances = torch.cdist(previous_positions[:, :2], current_positions[:, :2])
        num_current = current_positions.shape[0]
        flat_indices = torch.argsort(distances.flatten())
        used_previous: set[int] = set()
        used_current: set[int] = set()
        pairs = []

        for flat_index in flat_indices.tolist():
            previous_index = flat_index // num_current
            current_index = flat_index % num_current
            if previous_index in used_previous or current_index in used_current:
                continue
            if distances[previous_index, current_index].item() > self.config.max_temporal_match_distance:
                continue
            used_previous.add(previous_index)
            used_current.add(current_index)
            pairs.append((previous_index, current_index))

        return pairs


@dataclass
class DynamicTrainStepResult:
    loss: float
    metrics: Dict[str, float]


@dataclass
class VideoFrameResult:
    frame_index: int
    detections: List[dict]
    tracks: List[dict]
    reconstructed_features: Optional[Tensor] = None
    decoder_masks: Optional[Tensor] = None
    background_mask: Optional[Tensor] = None


@dataclass
class VideoInferenceResult:
    frames: List[VideoFrameResult]


@dataclass
class DynamicEvaluationResult:
    num_batches: int
    loss: float
    metrics: Dict[str, float]


class TKDISADynamicTrainingPipeline:
    def __init__(
        self,
        model: TKDISAModel,
        dynamic_config: Optional[TKDISADynamicConfig] = None,
        config: Optional[TrainPipelineConfig] = None,
        loss_module: Optional[TKDISADynamicLoss] = None,
        optimizer: Optional[Optimizer] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.dynamic_config = dynamic_config or TKDISADynamicConfig()
        self.config = config or self.dynamic_config.train or TrainPipelineConfig()
        self.loss_module = loss_module or TKDISADynamicLoss(
            frame_loss=TKDISALoss(model.config.loss),
            config=self.dynamic_config.loss,
        )
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

    def train_video_step(self, batch: Dict[str, Tensor]) -> DynamicTrainStepResult:
        frames = self._get_frames(batch)
        position_targets = self._maybe_to_device(batch.get("position_targets"))
        target_features = self._maybe_to_device(batch.get("target_features"))
        track_ids = self._maybe_to_device(batch.get("track_ids"))

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with amp_context(self.use_amp, self.amp_dtype):
            outputs = [self.model(frames[:, frame_index]) for frame_index in range(frames.shape[1])]
            loss_output = self.loss_module.compute_video_loss(
                outputs,
                position_targets=position_targets,
                target_features=target_features,
                track_ids=track_ids,
            )
        self._optimize(loss_output.total_loss)
        return DynamicTrainStepResult(
            loss=float(loss_output.total_loss.detach().cpu().item()),
            metrics=metrics_to_float(loss_output.metrics),
        )

    def _get_frames(self, batch: Dict[str, Tensor]) -> Tensor:
        frames = batch.get("frames")
        if frames is None:
            raise KeyError("Batch must include 'frames'.")
        if not frames.ndim == 5:
            raise ValueError("Dynamic training expects video batches with shape [B, T, C, H, W].")
        return frames.to(self.device, non_blocking=self.device.type == "cuda")

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


class TKDISADynamicInferencePipeline:
    def __init__(
        self,
        model: TKDISAModel,
        dynamic_config: Optional[TKDISADynamicConfig] = None,
        config: Optional[InferencePipelineConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.dynamic_config = dynamic_config or TKDISADynamicConfig()
        self.config = config or self.dynamic_config.inference or InferencePipelineConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.use_amp = self.config.amp and self.device.type == "cuda"
        self.amp_dtype = resolve_amp_dtype(self.config.amp_dtype)

    @torch.no_grad()
    def infer_video(
        self,
        frames: Tensor,
        confidence_threshold: Optional[float] = None,
        return_reconstruction: Optional[bool] = None,
        slot_manager: Optional[SlotManager] = None,
        slot_manager_config: Optional[SlotManagerConfig] = None,
    ) -> VideoInferenceResult:
        if frames.ndim == 5:
            if not frames.shape[0] == 1:
                raise ValueError("infer_video currently supports a single video clip at a time.")
            frames = frames[0]
        if not frames.ndim == 4:
            raise ValueError("infer_video expects [T, C, H, W] or [1, T, C, H, W] tensors.")

        threshold = self.config.confidence_threshold if confidence_threshold is None else confidence_threshold
        reconstruct = self.config.return_reconstruction if return_reconstruction is None else return_reconstruction
        manager = slot_manager or SlotManager(slot_manager_config or self.dynamic_config.slot_manager)

        self.model.eval()
        results: List[VideoFrameResult] = []
        for frame_index in range(frames.shape[0]):
            with amp_context(self.use_amp, self.amp_dtype):
                outputs = self.model(
                    frames[frame_index].unsqueeze(0).to(self.device, non_blocking=self.device.type == "cuda")
                )
            detections = self.model.extract_detections(outputs, confidence_threshold=threshold)[0]
            tracks = manager.step(detections)
            results.append(
                VideoFrameResult(
                    frame_index=frame_index,
                    detections=detections,
                    tracks=tracks,
                    reconstructed_features=outputs.reconstructed_features[0].detach().cpu() if reconstruct else None,
                    decoder_masks=outputs.decoder_masks[0].detach().cpu() if reconstruct else None,
                    background_mask=(
                        outputs.background_mask[0].detach().cpu()
                        if reconstruct and outputs.background_mask is not None
                        else None
                    ),
                )
            )

        return VideoInferenceResult(frames=results)


class TKDISADynamicEvaluationPipeline:
    def __init__(
        self,
        model: TKDISAModel,
        dynamic_config: Optional[TKDISADynamicConfig] = None,
        config: Optional[InferencePipelineConfig] = None,
        loss_module: Optional[TKDISADynamicLoss] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.dynamic_config = dynamic_config or TKDISADynamicConfig()
        self.config = config or self.dynamic_config.inference or InferencePipelineConfig()
        self.loss_module = loss_module or TKDISADynamicLoss(
            frame_loss=TKDISALoss(model.config.loss),
            config=self.dynamic_config.loss,
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.loss_module.to(self.device)
        self.use_amp = self.config.amp and self.device.type == "cuda"
        self.amp_dtype = resolve_amp_dtype(self.config.amp_dtype)

    @torch.no_grad()
    def evaluate_video_batches(self, batches: Iterable[Dict[str, Tensor]]) -> DynamicEvaluationResult:
        self.model.eval()
        metric_totals: Dict[str, float] = {}
        total_loss = 0.0
        batch_count = 0

        for batch in batches:
            frames = self._get_frames(batch)
            position_targets = self._maybe_to_device(batch.get("position_targets"))
            target_features = self._maybe_to_device(batch.get("target_features"))
            track_ids = self._maybe_to_device(batch.get("track_ids"))
            with amp_context(self.use_amp, self.amp_dtype):
                outputs = [self.model(frames[:, frame_index]) for frame_index in range(frames.shape[1])]
                loss_output = self.loss_module.compute_video_loss(
                    outputs,
                    position_targets=position_targets,
                    target_features=target_features,
                    track_ids=track_ids,
                )

            scalar_metrics = metrics_to_float(loss_output.metrics)
            total_loss += float(loss_output.total_loss.detach().cpu().item())
            for name, value in scalar_metrics.items():
                metric_totals[name] = metric_totals.get(name, 0.0) + value
            batch_count += 1

        if batch_count == 0:
            raise ValueError("Dynamic evaluation requires at least one batch.")

        averaged_metrics = {name: value / batch_count for name, value in metric_totals.items()}
        return DynamicEvaluationResult(
            num_batches=batch_count,
            loss=total_loss / batch_count,
            metrics=averaged_metrics,
        )

    def _get_frames(self, batch: Dict[str, Tensor]) -> Tensor:
        frames = batch.get("frames")
        if frames is None:
            raise KeyError("Batch must include 'frames'.")
        if not frames.ndim == 5:
            raise ValueError("Dynamic evaluation expects video batches with shape [B, T, C, H, W].")
        return frames.to(self.device, non_blocking=self.device.type == "cuda")

    def _maybe_to_device(self, tensor: Optional[Tensor]) -> Optional[Tensor]:
        if tensor is None:
            return None
        return tensor.to(self.device, non_blocking=self.device.type == "cuda")
