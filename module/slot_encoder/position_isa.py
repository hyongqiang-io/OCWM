from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import PositionISAConfig


class PositionISA(nn.Module):
    def __init__(self, config: Optional[PositionISAConfig] = None) -> None:
        super().__init__()
        self.config = config or PositionISAConfig()
        self.k_max = self.config.k_max

        self.slots_init = nn.Parameter(torch.zeros(self.k_max, 4))
        self._init_slots()

        self.feature_norm = nn.LayerNorm(self.config.d_feature)
        self.slot_norm = nn.LayerNorm(4)
        self.feature_proj = nn.Linear(self.config.d_feature, self.config.d_model)
        self.pos_enc_mlp = nn.Sequential(
            nn.Linear(2, self.config.d_pos_enc),
            nn.ReLU(),
            nn.Linear(self.config.d_pos_enc, self.config.d_pos_enc),
        )
        self.key_proj = nn.Linear(self.config.d_model + self.config.d_pos_enc, self.config.d_model)
        self.value_proj = nn.Linear(self.config.d_model + self.config.d_pos_enc, self.config.d_model)
        self.query_proj = nn.Linear(4, self.config.d_model)
        self.update_proj = nn.Linear(self.config.d_model, self.config.d_model)
        self.gru = nn.GRUCell(self.config.d_model, 4)
        self.mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, features: Tensor, slots: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_size, _, height, width = features.shape
        token_count = height * width
        features_flat = features.flatten(2).transpose(1, 2)
        features_flat = self.feature_norm(features_flat)
        feature_tokens = self.feature_proj(features_flat)

        if slots is None:
            slots = self.slots_init.unsqueeze(0).expand(batch_size, -1, -1).clone()

        attention = None
        for _ in range(self.config.num_iterations):
            relative_grid = self.compute_relative_grid(slots[:, :, :2], height=height, width=width)
            pos_enc = self.pos_enc_mlp(relative_grid)

            token_input = torch.cat(
                [feature_tokens.unsqueeze(1).expand(-1, self.k_max, -1, -1), pos_enc],
                dim=-1,
            )
            keys = self.key_proj(token_input)
            values = self.value_proj(token_input)
            queries = self.query_proj(self.slot_norm(slots))

            logits = torch.einsum("bkd,bknd->bkn", queries, keys) / math.sqrt(self.config.d_model)
            attention = F.softmax(logits, dim=-1)
            attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-6)

            updates = torch.einsum("bkn,bknd->bkd", attention, values)
            updates = self.update_proj(updates)
            slots = self.gru(updates.reshape(-1, self.config.d_model), slots.reshape(-1, 4)).reshape(
                batch_size, self.k_max, 4
            )
            slots = slots + self.mlp(slots)
            slots = self._clamp_slots(slots)

        if attention is None:
            attention = torch.zeros(batch_size, self.k_max, token_count, device=features.device, dtype=features.dtype)

        return slots, attention.reshape(batch_size, self.k_max, height, width)

    def compute_relative_grid(self, centers: Tensor, height: int, width: int) -> Tensor:
        device = centers.device
        dtype = centers.dtype
        y_coords = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        x_coords = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1).reshape(1, 1, height * width, 2)
        return grid - centers.unsqueeze(2)

    def _clamp_slots(self, slots: Tensor) -> Tensor:
        positions = slots[:, :, :2].clamp(0.0, 1.0)
        velocities = slots[:, :, 2:].clamp(-self.config.max_speed, self.config.max_speed)
        return torch.cat([positions, velocities], dim=-1)

    def _init_slots(self) -> None:
        grid_size = math.ceil(math.sqrt(self.k_max))
        coords = torch.linspace(0.0, 1.0, grid_size)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        centers = torch.stack([xx, yy], dim=-1).reshape(-1, 2)[: self.k_max]
        with torch.no_grad():
            self.slots_init[:, :2].copy_(centers)
            self.slots_init[:, 2:].zero_()
