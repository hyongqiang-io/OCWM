"""Mamba3 temporal block with PyTorch fallback.

Uses mamba_ssm.Mamba3 if available, otherwise falls back to a pure PyTorch
selective SSM implementation (correct but slower).
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from mamba_ssm import Mamba3
    MAMBA3_AVAILABLE = True
except ImportError:
    try:
        from mamba_ssm import Mamba2 as _Mamba2
        MAMBA3_AVAILABLE = False
        MAMBA2_AVAILABLE = True
        logger.info("Mamba3 not available, Mamba2 found as fallback option")
    except ImportError:
        MAMBA3_AVAILABLE = False
        MAMBA2_AVAILABLE = False
        logger.info("mamba-ssm not installed; using pure PyTorch SSM fallback")


class SimplifiedSSMBlock(nn.Module):
    """Pure PyTorch selective SSM block (sequential scan).

    Correct but slow — suitable for development and CPU/non-CUDA environments.
    Implements the core selective scan mechanism of Mamba.
    """

    def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = d_model * expand
        self.d_inner = d_inner

        self.in_proj = nn.Linear(d_model, d_inner, bias=False)
        self.gate_proj = nn.Linear(d_model, d_inner, bias=False)

        # 1D causal convolution
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True
        )

        # Input-dependent SSM parameters
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)

        # Learnable A parameter (log-space for stability)
        A_log_init = -torch.rand(d_inner, d_state) * 0.5
        self.A_log = nn.Parameter(A_log_init)

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        residual = x
        x = self.norm(x)
        B, T, _ = x.shape

        # Project input
        z = self.in_proj(x)  # (B, T, d_inner)
        gate = F.silu(self.gate_proj(x))  # (B, T, d_inner)

        # Causal conv1d
        z_conv = z.transpose(1, 2)  # (B, d_inner, T)
        z_conv = self.conv1d(z_conv)[:, :, :T]  # causal: trim future
        z = z_conv.transpose(1, 2)  # (B, T, d_inner)
        z = F.silu(z)

        # Compute input-dependent SSM parameters
        dt = F.softplus(self.dt_proj(z)).clamp(min=1e-3, max=1.0)  # (B, T, d_inner)
        B_param = self.B_proj(z)  # (B, T, d_state)
        C_param = self.C_proj(z)  # (B, T, d_state)

        # Sequential scan (fp32 for stability)
        A = -self.A_log.exp().clamp(min=1e-6, max=1.0)

        with torch.amp.autocast("cuda", enabled=False):
            z_f = z.float()
            dt_f = dt.float()
            B_f = B_param.float()
            C_f = C_param.float()
            A_f = A.float()

            h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=torch.float32)
            outputs = []
            for t in range(T):
                dA = torch.exp(torch.clamp(dt_f[:, t, :, None] * A_f[None, :, :], min=-20.0, max=0.0))
                dB = dt_f[:, t, :, None] * B_f[:, t, None, :]
                h = dA * h + dB * z_f[:, t, :, None]
                y_t = (h * C_f[:, t, None, :]).sum(dim=-1)
                outputs.append(y_t)
            y = torch.stack(outputs, dim=1).to(x.dtype)

        # Gate and project
        y = y * gate
        y = self.out_proj(y)

        return residual + y


class Mamba3Block(nn.Module):
    """Wrapper around mamba_ssm.Mamba3 with pre-norm and residual."""

    def __init__(self, d_model, d_state=64, headdim=64, is_mimo=True,
                 mimo_rank=4, chunk_size=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba3(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        return x + self.mamba(self.norm(x))


class Mamba3TemporalBlock(nn.Module):
    """Temporal modeling block using Mamba3 (or fallback).

    Processes per-particle temporal sequences independently.
    """

    def __init__(self, d_model, d_state=64, headdim=64, d_conv=4, expand=2,
                 is_mimo=True, mimo_rank=4, chunk_size=16, n_blocks=1):
        super().__init__()

        blocks = []
        for _ in range(n_blocks):
            if MAMBA3_AVAILABLE:
                blocks.append(Mamba3Block(
                    d_model=d_model,
                    d_state=d_state,
                    headdim=headdim,
                    is_mimo=is_mimo,
                    mimo_rank=mimo_rank,
                    chunk_size=chunk_size,
                ))
            else:
                blocks.append(SimplifiedSSMBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                ))
        self.blocks = nn.ModuleList(blocks)

        backend = "Mamba3" if MAMBA3_AVAILABLE else "PyTorch-fallback"
        logger.info(f"Mamba3TemporalBlock: using {backend}, d_model={d_model}, "
                    f"d_state={d_state}, n_blocks={n_blocks}")

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model) — single particle's temporal sequence
        Returns:
            (B, T, d_model)
        """
        for block in self.blocks:
            x = block(x)
        return x
