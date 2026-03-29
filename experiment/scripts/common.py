from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from module.slot_encoder import TKDISAConfig, TKDISAModel  # noqa: E402


DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiment" / "results"


class DummyEncoder(nn.Module):
    def __init__(self, feature_dim: int, patch_size: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        hidden_dim = min(max(feature_dim // 2, 32), 256)
        self.projection = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, feature_dim, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        height = x.shape[-2] // self.patch_size
        width = x.shape[-1] // self.patch_size
        pooled = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        return self.projection(pooled)


def resolve_device(device: str | None) -> str:
    if device:
        normalized = device.strip().lower()
        if normalized.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested but no GPU is available.")
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def timestamped_run_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def ensure_result_dir(run_name: str, output_dir: Path | None = None) -> Path:
    root = output_dir or DEFAULT_RESULTS_ROOT
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(payload: Any, path: Path) -> None:
    if is_dataclass(payload):
        payload = asdict(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def build_model(use_real_dino: bool, config: TKDISAConfig | None = None) -> TKDISAModel:
    config = config or TKDISAConfig()
    if use_real_dino:
        return TKDISAModel(config)

    encoder = DummyEncoder(
        feature_dim=config.encoder.resolved_output_dim(),
        patch_size=config.encoder.patch_size,
    )
    return TKDISAModel(config, encoder=encoder)


def collect_runtime_info(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "repo_root": str(REPO_ROOT),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_current_device"] = torch.cuda.current_device()
        info["cuda_device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return info
