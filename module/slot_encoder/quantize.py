from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class VectorQuantizeEMA(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        decay: float = 0.99,
        commitment_weight: float = 1.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.commitment_weight = commitment_weight
        self.eps = eps

        codebook = torch.randn(codebook_size, dim)
        codebook = F.normalize(codebook, dim=-1)

        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_codebook", codebook.clone())

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        original_shape = inputs.shape
        flat_inputs = inputs.reshape(-1, self.dim)

        distances = (
            flat_inputs.pow(2).sum(dim=-1, keepdim=True)
            - 2 * flat_inputs @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=-1).unsqueeze(0)
        )
        indices = distances.argmin(dim=-1)
        quantized = torch.nn.functional.embedding(indices, self.codebook).reshape(original_shape)

        if self.training:
            self._ema_update(flat_inputs, indices)

        commitment_loss = self.commitment_weight * F.mse_loss(inputs, quantized.detach())
        quantized = inputs + (quantized - inputs).detach()
        return quantized, indices.reshape(original_shape[:-1]), commitment_loss

    def get_codes_from_indices(self, indices: Tensor) -> Tensor:
        return torch.nn.functional.embedding(indices, self.codebook)

    def _ema_update(self, flat_inputs: Tensor, indices: Tensor) -> None:
        encodings = F.one_hot(indices, num_classes=self.codebook_size).type(flat_inputs.dtype)
        cluster_size = encodings.sum(dim=0)
        codebook_sum = encodings.t() @ flat_inputs

        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)
        self.ema_codebook.mul_(self.decay).add_(codebook_sum, alpha=1.0 - self.decay)

        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
        normalized_codebook = self.ema_codebook / smoothed.unsqueeze(-1).clamp_min(self.eps)
        self.codebook.copy_(normalized_codebook)
