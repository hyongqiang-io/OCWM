"""Interleaved GNN-Mamba3 block for particle dynamics.

Architecture per block:
  1. GNN: compute interaction context at each timestep
  2. Fuse: project concat(particle, context) back to D
  3. Mamba3: temporal evolution per particle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.checkpoint as cp

from .egnn import EGNNInteraction
from .mamba3_temporal import Mamba3TemporalBlock


class ContextFusion(nn.Module):
    """Fuses GNN context with particle tokens via projection."""

    def __init__(self, node_dim, context_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(node_dim + context_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x, c):
        """
        Args:
            x: [B, K, D] particle tokens
            c: [B, K, d_c] interaction context
        Returns:
            [B, K, D] fused tokens
        """
        fused = self.proj(torch.cat([x, c], dim=-1))
        return self.norm(x + fused)


class AdaLNModulation(nn.Module):
    """Adaptive Layer Norm modulation for context conditioning."""

    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(cond_dim, 4 * d_model)
        nn.init.constant_(self.proj.weight, 0.0)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x, c):
        """
        Args:
            x: [..., D]
            c: [..., cond_dim] (broadcastable)
        Returns:
            [..., D] modulated x
        """
        if c is None:
            return self.norm(x)
        params = self.proj(c)
        scale1, shift1, scale2, shift2 = params.chunk(4, dim=-1)
        return self.norm(x) * (1 + scale1) + shift1


class GNNMamba3Block(nn.Module):
    """Single interleaved GNN-Mamba3 block.

    Data flow:
        x [B, N, T, D]
        → GNN per timestep → context c [B*T, N_fg, d_c]
        → fuse(x, c) → [B, N, T, D]
        → Mamba3 per particle → [B, N, T, D]
    """

    def __init__(self, d_model, gnn_hidden_dim=128, gnn_context_dim=128,
                 gnn_layers=3, egnn_n_rbf=16,
                 mamba_d_state=64, mamba_headdim=64, mamba_d_conv=4,
                 mamba_expand=2, mamba_is_mimo=True, mamba_mimo_rank=4,
                 mamba_chunk_size=16,
                 context_cond=False, cond_dim=None):
        super().__init__()

        self.d_model = d_model
        self.context_cond = context_cond

        self.gnn = EGNNInteraction(
            node_dim=d_model,
            hidden_dim=gnn_hidden_dim,
            context_dim=gnn_context_dim,
            n_layers=gnn_layers,
            n_rbf=egnn_n_rbf,
        )

        # Fuse GNN context with particle tokens
        self.fusion = ContextFusion(d_model, gnn_context_dim)

        # Mamba3 for temporal evolution
        self.mamba = Mamba3TemporalBlock(
            d_model=d_model,
            d_state=mamba_d_state,
            headdim=mamba_headdim,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            is_mimo=mamba_is_mimo,
            mimo_rank=mamba_mimo_rank,
            chunk_size=mamba_chunk_size,
            n_blocks=1,
        )

        # Optional context conditioning (AdaLN)
        if context_cond and cond_dim:
            self.adaln = AdaLNModulation(d_model, cond_dim)
        else:
            self.adaln = None

    def forward(self, x, pos=None, c=None):
        """
        Args:
            x: [B, N, T, D] particle tokens (N includes bg token at last position)
            pos: [B, N, T, pos_dims] raw spatial positions (optional, for GNN distance)
            c: [B, N, T, D] or [B, 1, T, D] context conditioning (optional)
        Returns:
            [B, N, T, D] updated particle tokens
        """
        B, N, T, D = x.shape

        x_fg = x[:, :-1]  # [B, N-1, T, D]
        x_bg = x[:, -1:]  # [B, 1, T, D]
        N_fg = N - 1

        # --- Spatial: GNN at each timestep (loop over T to avoid O(B*T*K^2) memory) ---
        fused_list = []
        for t_idx in range(T):
            x_fg_t = x_fg[:, :, t_idx, :]  # [B, N_fg, D]
            if pos is not None:
                pos_fg_t = pos[:, :-1, t_idx, :]  # [B, N_fg, pos_dims]
            else:
                pos_fg_t = torch.zeros(B, N_fg, 3, device=x.device)
            ctx_t = self.gnn(x_fg_t, pos_fg_t)  # [B, N_fg, d_c]
            fused_t = self.fusion(x_fg_t, ctx_t)  # [B, N_fg, D]
            fused_list.append(fused_t)

        # Stack: [B, N_fg, T, D]
        x_fg_fused = torch.stack(fused_list, dim=2)

        # --- Temporal: Mamba3 per particle ---
        # Process fg particles: [B, N_fg, T, D] -> [B*N_fg, T, D]
        x_fg_temporal = x_fg_fused.reshape(B * N_fg, T, D)
        x_fg_temporal = self.mamba(x_fg_temporal)  # [B*N_fg, T, D]
        x_fg_out = x_fg_temporal.reshape(B, N_fg, T, D)

        # Process bg token: [B, 1, T, D] -> [B, T, D]
        x_bg_temporal = x_bg.squeeze(1)  # [B, T, D]
        x_bg_temporal = self.mamba(x_bg_temporal)  # [B, T, D]
        x_bg_out = x_bg_temporal.unsqueeze(1)  # [B, 1, T, D]

        # Recombine
        x_out = torch.cat([x_fg_out, x_bg_out], dim=1)  # [B, N, T, D]

        # Optional AdaLN modulation
        if self.adaln is not None and c is not None:
            x_out = self.adaln(x_out, c)

        return x_out


class GNNMamba3Transformer(nn.Module):
    """Stack of interleaved GNN-Mamba3 blocks.

    Drop-in replacement for ParticleSpatioTemporalTransformer.
    Same interface: input [B, N, T, D], output [B, N, T, output_dim].
    """

    def __init__(self, n_embed, n_layer, block_size, output_dim,
                 gnn_hidden_dim=128, gnn_context_dim=128, gnn_layers=3,
                 egnn_n_rbf=16,
                 mamba_d_state=64, mamba_headdim=64, mamba_d_conv=4,
                 mamba_expand=2, mamba_is_mimo=True, mamba_mimo_rank=4,
                 mamba_chunk_size=16,
                 context_cond=False, init_std=0.02,
                 pos_embed_t_adaln=False):
        super().__init__()
        self.n_embed = n_embed
        self.block_size = block_size
        self.init_std = init_std
        self.context_cond = context_cond
        self.pos_embed_t_adaln = pos_embed_t_adaln

        # Positional embedding for timesteps
        if pos_embed_t_adaln:
            self.pos_embed_t_embedding = nn.Parameter(
                init_std * torch.randn(1, 1, block_size, n_embed))
        else:
            self.pos_emb = nn.Parameter(init_std * torch.randn(1, block_size, n_embed))

        # Stacked GNN-Mamba3 blocks
        self.blocks = nn.ModuleList([
            GNNMamba3Block(
                d_model=n_embed,
                gnn_hidden_dim=gnn_hidden_dim,
                gnn_context_dim=gnn_context_dim,
                gnn_layers=gnn_layers,
                egnn_n_rbf=egnn_n_rbf,
                mamba_d_state=mamba_d_state,
                mamba_headdim=mamba_headdim,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                mamba_is_mimo=mamba_is_mimo,
                mamba_mimo_rank=mamba_mimo_rank,
                mamba_chunk_size=mamba_chunk_size,
                context_cond=context_cond,
                cond_dim=n_embed if context_cond else None,
            )
            for _ in range(n_layer)
        ])

        # Output head (matches ParticleSpatioTemporalTransformer)
        self.head_norm = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, output_dim)

    def get_block_size(self):
        return self.block_size

    def forward(self, x, pos=None, c=None, l=None):
        """
        Args:
            x: [B, N, T, D] particle tokens (particles_first format)
            pos: [B, N, T, pos_dims] raw spatial positions (for GNN distance)
            c: [B, N, T, D] or [B, 1, T, D] context (AdaLN conditioning)
            l: language conditioning (unused, kept for interface compat)
        Returns:
            [B, N, T, output_dim] predictions
        """
        b, n, t, f = x.shape
        assert t <= self.block_size
        assert f == self.n_embed

        # Add positional embeddings
        if self.pos_embed_t_adaln:
            pos_t = self.pos_embed_t_embedding[:, :, :t].expand(b, n, -1, -1)
            if c is None:
                c = pos_t
            else:
                # Ensure c has the right time dimension before adding
                if c.shape[2] == t:
                    c = c + pos_t
                else:
                    c = pos_t
        else:
            pos_emb = self.pos_emb[:, :t, :].unsqueeze(1)  # [1, 1, T, D]
            x = x + pos_emb

        # Forward through stacked blocks (with gradient checkpointing to save memory)
        for block in self.blocks:
            if self.training:
                x = cp.checkpoint(block, x, pos, c, use_reentrant=False)
            else:
                x = block(x, pos=pos, c=c)

        # Output projection
        logits = self.head(self.head_norm(x))
        return logits
