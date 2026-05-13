"""EMA (Exponential Moving Average) target encoder for JEPA training."""

import copy

import torch
import torch.nn as nn


class EMAEncoder(nn.Module):
    """Wraps an encoder with EMA weight updates for JEPA target.

    The target encoder's parameters are an exponential moving average of the
    online encoder. Gradients do not flow through the target encoder.

    Usage:
        online_encoder = DLPEncoder(...)
        target_encoder = EMAEncoder(online_encoder, decay=0.996)

        # In training loop:
        z_online = online_encoder(x_t)
        z_target = target_encoder(x_{t+1})  # no grad
        loss = jepa_loss(z_predicted, z_target)

        # After optimizer step:
        target_encoder.update()
    """

    def __init__(self, online_encoder: nn.Module, decay: float = 0.996):
        super().__init__()
        self.decay = decay
        self.encoder = copy.deepcopy(online_encoder)
        # Freeze target encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, online_encoder: nn.Module):
        """Update target encoder parameters via EMA from online encoder."""
        for target_param, online_param in zip(
            self.encoder.parameters(), online_encoder.parameters()
        ):
            target_param.data.mul_(self.decay).add_(
                online_param.data, alpha=1.0 - self.decay
            )

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        """Forward pass through target encoder (no gradients)."""
        return self.encoder(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """Include decay in state dict."""
        state = super().state_dict(*args, **kwargs)
        state['_ema_decay'] = self.decay
        return state

    def load_state_dict(self, state_dict, strict=True):
        """Restore decay from state dict."""
        if '_ema_decay' in state_dict:
            self.decay = state_dict.pop('_ema_decay')
        super().load_state_dict(state_dict, strict=strict)
