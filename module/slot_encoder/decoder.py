from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import DecoderConfig


class SlotFeatureDecoder(nn.Module):
    def __init__(self, config: Optional[DecoderConfig] = None) -> None:
        super().__init__()
        self.config = config or DecoderConfig()

        combined_dim = self.config.d_vis + 4
        self.slot_feature_head = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, self.config.d_hidden),
            nn.ReLU(),
            nn.Linear(self.config.d_hidden, self.config.d_feature),
        )
        self.scale_head = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, self.config.d_hidden),
            nn.ReLU(),
            nn.Linear(self.config.d_hidden, 2),
        )
        self.presence_head = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, self.config.d_hidden),
            nn.ReLU(),
            nn.Linear(self.config.d_hidden, 1),
        )

        if self.config.use_background_slot:
            self.background_feature = nn.Parameter(torch.zeros(self.config.d_feature))
            self.background_logit = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("background_feature", None)
            self.register_parameter("background_logit", None)

    def forward(
        self,
        position_slots: Tensor,
        visual_slots: Tensor,
        output_size: Tuple[int, int],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        batch_size, num_slots, _ = position_slots.shape
        height, width = output_size

        combined_slots = torch.cat([position_slots, visual_slots], dim=-1)
        slot_features = self.slot_feature_head(combined_slots)
        scales = F.softplus(self.scale_head(combined_slots)) + self.config.min_scale
        scales = scales.clamp(max=self.config.max_scale)
        presence_logits = self.presence_head(combined_slots)

        grid = self._build_grid(height, width, position_slots.device, position_slots.dtype)
        centers = position_slots[:, :, :2].unsqueeze(2)
        rel = (grid - centers) / scales.unsqueeze(2)
        slot_logits = -0.5 * rel.pow(2).sum(dim=-1)
        slot_logits = slot_logits / max(self.config.mask_temperature, 1e-6)
        slot_logits = slot_logits + presence_logits

        if self.config.use_background_slot:
            background_logits = self.background_logit.view(1, 1, 1).expand(batch_size, 1, height * width)
            all_logits = torch.cat([slot_logits, background_logits], dim=1)
            all_masks = F.softmax(all_logits, dim=1)
            slot_masks = all_masks[:, :num_slots]
            background_mask = all_masks[:, num_slots]
        else:
            slot_masks = F.softmax(slot_logits, dim=1)
            background_mask = None

        recon_flat = torch.einsum("bkn,bkd->bnd", slot_masks, slot_features)

        if background_mask is not None and self.background_feature is not None:
            recon_flat = recon_flat + background_mask.unsqueeze(-1) * self.background_feature.view(1, 1, -1)

        reconstructed = recon_flat.transpose(1, 2).reshape(batch_size, self.config.d_feature, height, width)
        slot_masks = slot_masks.reshape(batch_size, num_slots, height, width)

        if background_mask is not None:
            background_mask = background_mask.reshape(batch_size, height, width)

        return reconstructed, slot_masks, background_mask

    def _build_grid(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        y_coords = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        x_coords = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(1, 1, height * width, 2)
