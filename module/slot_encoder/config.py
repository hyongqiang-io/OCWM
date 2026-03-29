from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


_DINOV2_EMBED_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


def infer_dinov2_output_dim(model_name: str, num_layers: int) -> int:
    try:
        embed_dim = _DINOV2_EMBED_DIMS[model_name]
    except KeyError as exc:
        known = ", ".join(sorted(_DINOV2_EMBED_DIMS))
        raise ValueError(f"Unsupported DINOv2 model_name '{model_name}'. Known models: {known}.") from exc
    return embed_dim * max(num_layers, 1)


@dataclass
class DINOv2Config:
    model_name: str = "dinov2_vits14"
    repo: str = "facebookresearch/dinov2"
    layers: Tuple[int, ...] = (4, 11)
    freeze: bool = True
    pretrained: bool = True
    patch_size: int = 14
    output_dim: Optional[int] = None
    source: str = "github"
    trust_repo: bool = True
    skip_validation: bool = True
    force_reload: bool = False
    verbose: bool = False
    disable_xformers_without_cuda: bool = True
    enable_cuda_autocast: bool = True
    cuda_autocast_dtype: str = "float16"

    def resolved_output_dim(self) -> int:
        if self.output_dim is not None:
            return self.output_dim
        return infer_dinov2_output_dim(self.model_name, len(self.layers))


@dataclass
class PositionISAConfig:
    k_max: int = 16
    d_feature: int = 768
    d_model: int = 256
    d_pos_enc: int = 16
    num_iterations: int = 3
    max_speed: float = 0.1


@dataclass
class VisualISAConfig:
    k_max: int = 16
    d_feature: int = 768
    d_vis: int = 256
    d_hidden: int = 512
    num_iterations: int = 3
    codebook_size: int = 512
    vq_decay: float = 0.99
    commitment_weight: float = 1.0


@dataclass
class DecoderConfig:
    d_feature: int = 768
    d_vis: int = 256
    d_hidden: int = 512
    min_scale: float = 0.05
    max_scale: float = 0.5
    mask_temperature: float = 1.0
    use_background_slot: bool = True


@dataclass
class LossConfig:
    reconstruction_weight: float = 1.0
    vq_weight: float = 1.0
    position_weight: float = 1.0
    use_pseudo_position_targets: bool = True


@dataclass
class TrainPipelineConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: Optional[float] = 1.0
    amp: bool = True
    amp_dtype: str = "float16"


@dataclass
class InferencePipelineConfig:
    confidence_threshold: float = 0.0
    return_reconstruction: bool = False
    amp: bool = True
    amp_dtype: str = "float16"


@dataclass
class TKDISAConfig:
    encoder: DINOv2Config = field(default_factory=DINOv2Config)
    position: PositionISAConfig = field(default_factory=PositionISAConfig)
    visual: VisualISAConfig = field(default_factory=VisualISAConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainPipelineConfig = field(default_factory=TrainPipelineConfig)
    inference: InferencePipelineConfig = field(default_factory=InferencePipelineConfig)

    def __post_init__(self) -> None:
        feature_dim = self.encoder.resolved_output_dim()
        self.position.d_feature = feature_dim
        self.visual.d_feature = feature_dim
        self.visual.k_max = self.position.k_max
        self.decoder.d_feature = feature_dim
        self.decoder.d_vis = self.visual.d_vis
