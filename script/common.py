from __future__ import annotations

from contextlib import nullcontext
from typing import Dict

import torch
from torch import Tensor


def resolve_amp_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        known = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported amp dtype '{name}'. Known dtypes: {known}.") from exc


def amp_context(use_amp: bool, amp_dtype: torch.dtype):
    if not use_amp:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)


def metrics_to_float(metrics: Dict[str, Tensor]) -> Dict[str, float]:
    return {name: float(value.detach().cpu().item()) for name, value in metrics.items()}
