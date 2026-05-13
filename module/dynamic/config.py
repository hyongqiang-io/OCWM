from dataclasses import dataclass, field


@dataclass
class GNNMamba3Config:
    # Architecture
    projection_dim: int = 512
    n_interleaved_blocks: int = 3

    # GNN (EGNN)
    gnn_layers: int = 3
    gnn_hidden_dim: int = 128
    gnn_context_dim: int = 128
    egnn_n_rbf: int = 16
    gnn_dropout: float = 0.0

    # Mamba3
    mamba_d_state: int = 64
    mamba_headdim: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_is_mimo: bool = True
    mamba_mimo_rank: int = 4
    mamba_chunk_size: int = 16
    mamba_dropout: float = 0.1

    # Context conditioning (matching LPWM adaln)
    context_cond: bool = True
    residual_modulation: bool = True
    context_gate: bool = True

    # JEPA
    ema_decay: float = 0.996
    lambda_rec_start: float = 1.0
    lambda_rec_end: float = 0.0
    anneal_epoch_start: int = 30
    anneal_epoch_end: int = 80
    lambda_jepa: float = 1.0
    lambda_cross_view: float = 1.0
    beta_kl: float = 0.08

    # Particles
    n_fg_particles: int = 30
    max_particles: int = 30

    # From LPWM config
    n_head: int = 8
    n_layer: int = 6
    block_size: int = 20
    dropout: float = 0.1
    norm_type: str = "rms"
