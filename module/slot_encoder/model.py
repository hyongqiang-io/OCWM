from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor, nn

from .config import TKDISAConfig
from .decoder import SlotFeatureDecoder
from .dino import DINOv2Encoder
from .position_isa import PositionISA
from .visual_isa import VisualISA


@dataclass
class SlotEncoderOutput:
    features: Tensor
    position_slots: Tensor
    position_attention: Tensor
    visual_slots: Tensor
    quantized_visual_slots: Tensor
    visual_code_indices: Tensor
    commitment_loss: Tensor


@dataclass
class TKDISAModelOutput:
    features: Tensor
    position_slots: Tensor
    position_attention: Tensor
    visual_slots: Tensor
    quantized_visual_slots: Tensor
    visual_code_indices: Tensor
    commitment_loss: Tensor
    reconstructed_features: Tensor
    decoder_masks: Tensor
    background_mask: Optional[Tensor]


class TKDISAEncoder(nn.Module):
    def __init__(self, config: Optional[TKDISAConfig] = None, encoder: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.config = config or TKDISAConfig()
        self.encoder = encoder or DINOv2Encoder(self.config.encoder)
        self.position_isa = PositionISA(self.config.position)
        self.visual_isa = VisualISA(self.config.visual)

    def forward(
        self,
        x: Tensor,
        position_slots: Optional[Tensor] = None,
        visual_slots: Optional[Tensor] = None,
    ) -> SlotEncoderOutput:
        features = self.encoder(x)
        position_slots_out, position_attention = self.position_isa(features, slots=position_slots)
        quantized_visual_slots, visual_code_indices, commitment_loss, visual_slots_out = self.visual_isa(
            features,
            position_slots_out.detach(),
            position_attention,
            slots=visual_slots,
        )
        return SlotEncoderOutput(
            features=features,
            position_slots=position_slots_out,
            position_attention=position_attention,
            visual_slots=visual_slots_out,
            quantized_visual_slots=quantized_visual_slots,
            visual_code_indices=visual_code_indices,
            commitment_loss=commitment_loss,
        )


class TKDISAModel(nn.Module):
    def __init__(self, config: Optional[TKDISAConfig] = None, encoder: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.config = config or TKDISAConfig()
        self.slot_encoder = TKDISAEncoder(self.config, encoder=encoder)
        self.decoder = SlotFeatureDecoder(self.config.decoder)

    def forward(
        self,
        x: Tensor,
        position_slots: Optional[Tensor] = None,
        visual_slots: Optional[Tensor] = None,
    ) -> TKDISAModelOutput:
        encoded = self.slot_encoder(x, position_slots=position_slots, visual_slots=visual_slots)
        reconstructed_features, decoder_masks, background_mask = self.decoder(
            position_slots=encoded.position_slots,
            visual_slots=encoded.quantized_visual_slots,
            output_size=encoded.features.shape[-2:],
        )
        return TKDISAModelOutput(
            features=encoded.features,
            position_slots=encoded.position_slots,
            position_attention=encoded.position_attention,
            visual_slots=encoded.visual_slots,
            quantized_visual_slots=encoded.quantized_visual_slots,
            visual_code_indices=encoded.visual_code_indices,
            commitment_loss=encoded.commitment_loss,
            reconstructed_features=reconstructed_features,
            decoder_masks=decoder_masks,
            background_mask=background_mask,
        )

    @torch.no_grad()
    def detect_slots(
        self,
        x: Tensor,
        confidence_threshold: float = 0.0,
    ) -> List[List[dict]]:
        outputs = self.forward(x)
        return self.extract_detections(outputs, confidence_threshold=confidence_threshold)

    @staticmethod
    def extract_detections(outputs: TKDISAModelOutput, confidence_threshold: float = 0.0) -> List[List[dict]]:
        peak_attention = outputs.position_attention.flatten(2).amax(dim=-1)
        detections: List[List[dict]] = []

        for batch_index in range(outputs.position_slots.shape[0]):
            batch_detections = []
            for slot_index in range(outputs.position_slots.shape[1]):
                confidence = peak_attention[batch_index, slot_index].item()
                if confidence < confidence_threshold:
                    continue
                batch_detections.append(
                    {
                        "position": outputs.position_slots[batch_index, slot_index].detach().cpu().numpy(),
                        "visual_code": int(outputs.visual_code_indices[batch_index, slot_index].item()),
                        "confidence": confidence,
                    }
                )
            detections.append(batch_detections)

        return detections
