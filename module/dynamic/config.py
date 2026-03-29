from __future__ import annotations

from dataclasses import dataclass, field

from module.slot_encoder.config import InferencePipelineConfig, TrainPipelineConfig


@dataclass
class TemporalLossConfig:
    temporal_visual_weight: float = 1.0
    temporal_velocity_weight: float = 0.1
    max_temporal_match_distance: float = 0.25


@dataclass
class SlotManagerConfig:
    max_occlusion: int = 30
    min_confidence: float = 0.3
    position_threshold: float = 0.2
    cost_threshold: float = 2.0
    velocity_decay: float = 0.95
    measurement_alpha: float = 0.7


@dataclass
class TKDISADynamicConfig:
    loss: TemporalLossConfig = field(default_factory=TemporalLossConfig)
    slot_manager: SlotManagerConfig = field(default_factory=SlotManagerConfig)
    train: TrainPipelineConfig = field(default_factory=TrainPipelineConfig)
    inference: InferencePipelineConfig = field(default_factory=InferencePipelineConfig)
