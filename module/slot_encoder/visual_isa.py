from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .config import VisualISAConfig
from .quantize import VectorQuantizeEMA


class VisualISA(nn.Module):
    def __init__(self, config: Optional[VisualISAConfig] = None) -> None:
        super().__init__()
        self.config = config or VisualISAConfig()
        self.k_max = self.config.k_max
        self.d_vis = self.config.d_vis

        self.slots_init = nn.Parameter(torch.randn(self.k_max, self.d_vis) * 0.02)
        self.feature_extractor = nn.Sequential(
            nn.LayerNorm(self.config.d_feature),
            nn.Linear(self.config.d_feature, self.config.d_hidden),
            nn.ReLU(),
            nn.Linear(self.config.d_hidden, self.d_vis),
        )
        self.gru = nn.GRUCell(self.d_vis, self.d_vis)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.d_vis),
            nn.Linear(self.d_vis, self.d_vis),
            nn.ReLU(),
            nn.Linear(self.d_vis, self.d_vis),
        )
        self.vq = VectorQuantizeEMA(
            dim=self.d_vis,
            codebook_size=self.config.codebook_size,
            decay=self.config.vq_decay,
            commitment_weight=self.config.commitment_weight,
        )

    def forward(
        self,
        features: Tensor,
        pos_slots: Tensor,
        pos_attention_masks: Tensor,
        slots: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        del pos_slots

        batch_size, _, _, _ = features.shape
        features_flat = features.flatten(2).transpose(1, 2)
        masks = pos_attention_masks.flatten(2).detach()
        masks = masks / (masks.sum(dim=-1, keepdim=True) + 1e-6)

        if slots is None:
            slots = self.slots_init.unsqueeze(0).expand(batch_size, -1, -1).clone()

        pooled_features = torch.einsum("bkn,bnd->bkd", masks, features_flat)
        updates = self.feature_extractor(pooled_features)

        for _ in range(self.config.num_iterations):
            slots = self.gru(updates.reshape(-1, self.d_vis), slots.reshape(-1, self.d_vis)).reshape(
                batch_size, self.k_max, self.d_vis
            )
            slots = slots + self.mlp(slots)

        quantized_slots, indices, commitment_loss = self.vq(slots)
        return quantized_slots, indices, commitment_loss, slots
