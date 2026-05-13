from .config import GNNMamba3Config
from .mpnn import LightweightMPNN
from .mamba3_temporal import Mamba3TemporalBlock
from .gnn_mamba3_block import GNNMamba3Block, GNNMamba3Transformer
from .ema_encoder import EMAEncoder
from .jepa_loss import JEPALoss, AnnealingScheduler

# GNNMamba3Dynamics requires LPWM dependencies (cv2, etc.)
# Import lazily to avoid blocking other module usage
def get_dynamics_class():
    """Lazy import of GNNMamba3Dynamics (requires LPWM runtime dependencies)."""
    from .gnn_mamba3_dynamics import GNNMamba3Dynamics
    return GNNMamba3Dynamics

__all__ = [
    "GNNMamba3Config",
    "LightweightMPNN",
    "Mamba3TemporalBlock",
    "GNNMamba3Block",
    "GNNMamba3Transformer",
    "get_dynamics_class",
    "EMAEncoder",
    "JEPALoss",
    "AnnealingScheduler",
]
