"""Equivariant Graph Neural Network (EGNN) for particle interaction context.

Based on Satorras et al. 2021 "E(n) Equivariant Graph Neural Networks".
- Equivariant coordinate updates avoid over-smoothing across layers
- Edge messages use RBF-encoded distances (no K×K dense feature expansion)
- Drop-in replacement for LightweightMPNN: same forward(x, pos) → context interface
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RadialBasisEncoding(nn.Module):
    """Gaussian RBF encoding of scalar distances."""

    def __init__(self, n_rbf=16, d_min=0.0, d_max=2.0):
        super().__init__()
        self.n_rbf = n_rbf
        centers = torch.linspace(d_min, d_max, n_rbf)
        self.register_buffer('centers', centers)
        self.width = (d_max - d_min) / n_rbf

    def forward(self, dist):
        """
        Args:
            dist: [...] scalar distances
        Returns:
            [..., n_rbf] RBF features
        """
        dist = dist.unsqueeze(-1)  # [..., 1]
        return torch.exp(-((dist - self.centers) ** 2) / (self.width ** 2))


class EGNNLayer(nn.Module):
    """Single EGNN layer with equivariant coordinate updates."""

    def __init__(self, node_dim, hidden_dim, pos_dim=3, n_rbf=16):
        super().__init__()
        self.pos_dim = pos_dim

        self.rbf = RadialBasisEncoding(n_rbf=n_rbf, d_min=0.0, d_max=2.0)

        # Edge message: phi_e(h_i, h_j, rbf(dist))
        edge_input_dim = node_dim * 2 + n_rbf
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Coordinate update weight: phi_x(m_ij) → scalar
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Node update: phi_h(h_i, agg_i)
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )

        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h, pos, edge_index):
        """
        Args:
            h: [B, K, D] node features
            pos: [B, K, pos_dim] coordinates
            edge_index: [2, num_edges] source/target indices
        Returns:
            h_out: [B, K, D] updated features
            pos_out: [B, K, pos_dim] updated coordinates
        """
        B, K, D = h.shape
        src, dst = edge_index  # [num_edges] each

        # Gather node features for edges
        h_src = h[:, src]  # [B, E, D]
        h_dst = h[:, dst]  # [B, E, D]

        # Compute distances
        pos_src = pos[:, src]  # [B, E, pos_dim]
        pos_dst = pos[:, dst]  # [B, E, pos_dim]
        rel_pos = pos_src - pos_dst  # [B, E, pos_dim]
        dist = rel_pos.norm(dim=-1)  # [B, E]

        # RBF encoding of distances
        rbf_feat = self.rbf(dist)  # [B, E, n_rbf]

        # Edge messages
        edge_input = torch.cat([h_src, h_dst, rbf_feat], dim=-1)  # [B, E, 2D + n_rbf]
        messages = self.edge_mlp(edge_input)  # [B, E, hidden]

        # Coordinate update (equivariant)
        coord_weights = self.coord_mlp(messages)  # [B, E, 1]
        coord_delta = rel_pos * coord_weights  # [B, E, pos_dim]
        # Aggregate coord updates per node
        pos_update = torch.zeros_like(pos)  # [B, K, pos_dim]
        pos_update.scatter_add_(1, src.unsqueeze(0).unsqueeze(-1).expand(B, -1, self.pos_dim), coord_delta)
        pos_out = pos + pos_update

        # Aggregate messages per node
        agg = torch.zeros(B, K, messages.shape[-1], device=h.device)
        agg.scatter_add_(1, src.unsqueeze(0).unsqueeze(-1).expand(B, -1, messages.shape[-1]), messages)

        # Node update (residual)
        h_input = torch.cat([h, agg], dim=-1)  # [B, K, D + hidden]
        h_out = self.norm(h + self.node_mlp(h_input))

        return h_out, pos_out


class EGNNInteraction(nn.Module):
    """Multi-layer EGNN producing interaction context vectors.

    Drop-in replacement for LightweightMPNN.
    Same interface: forward(x, pos) → context [B, K, context_dim]
    """

    def __init__(self, node_dim, hidden_dim=128, context_dim=128, n_layers=3,
                 pos_dim=3, n_rbf=16):
        super().__init__()
        self.layers = nn.ModuleList([
            EGNNLayer(node_dim, hidden_dim, pos_dim=pos_dim, n_rbf=n_rbf)
            for _ in range(n_layers)
        ])
        self.context_proj = nn.Linear(node_dim, context_dim)
        self._edge_index_cache = {}

    def _get_edge_index(self, K, device):
        """Build fully-connected edge index (excluding self-loops)."""
        key = (K, device)
        if key not in self._edge_index_cache:
            src = torch.arange(K, device=device).repeat_interleave(K - 1)
            dst = torch.cat([
                torch.cat([torch.arange(i, device=device), torch.arange(i + 1, K, device=device)])
                for i in range(K)
            ])
            self._edge_index_cache[key] = (src, dst)
        return self._edge_index_cache[key]

    def forward(self, x, pos):
        """
        Args:
            x: [B, K, D] particle tokens
            pos: [B, K, pos_dim] raw spatial coordinates
        Returns:
            context: [B, K, context_dim] interaction context per particle
        """
        K = x.shape[1]
        edge_index = self._get_edge_index(K, x.device)

        h = x
        p = pos
        for layer in self.layers:
            h, p = layer(h, p, edge_index)

        return self.context_proj(h)
