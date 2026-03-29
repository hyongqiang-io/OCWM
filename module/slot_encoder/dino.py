from __future__ import annotations

import math
import os
from contextlib import nullcontext
from typing import Optional

import torch
from torch import Tensor, nn

from .config import DINOv2Config


def _resolve_amp_dtype(name: str) -> torch.dtype:
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


class DINOv2Encoder(nn.Module):
    def __init__(self, config: Optional[DINOv2Config] = None, backbone: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.config = config or DINOv2Config()

        if backbone is None and self.config.disable_xformers_without_cuda and not torch.cuda.is_available():
            os.environ.setdefault("XFORMERS_DISABLED", "1")

        self.model = backbone or torch.hub.load(
            self.config.repo,
            self.config.model_name,
            source=self.config.source,
            pretrained=self.config.pretrained,
            trust_repo=self.config.trust_repo,
            skip_validation=self.config.skip_validation,
            force_reload=self.config.force_reload,
            verbose=self.config.verbose,
        )

        if self.config.freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def forward(self, x: Tensor) -> Tensor:
        patch_h = x.shape[-2] // self.config.patch_size
        patch_w = x.shape[-1] // self.config.patch_size

        amp_context = nullcontext()
        if x.device.type == "cuda" and self.config.enable_cuda_autocast:
            amp_context = torch.autocast(
                device_type="cuda",
                dtype=_resolve_amp_dtype(self.config.cuda_autocast_dtype),
                enabled=True,
            )

        with amp_context:
            if hasattr(self.model, "get_intermediate_layers"):
                outputs = self.model.get_intermediate_layers(x, n=list(self.config.layers))
                layer_tokens = [
                    self._normalize_token_output(output, patch_h=patch_h, patch_w=patch_w) for output in outputs
                ]
                tokens = torch.cat(layer_tokens, dim=-1)
            elif hasattr(self.model, "forward_features"):
                outputs = self.model.forward_features(x)
                tokens = self._normalize_token_output(outputs, patch_h=patch_h, patch_w=patch_w)
            else:
                outputs = self.model(x)
                tokens = self._normalize_token_output(outputs, patch_h=patch_h, patch_w=patch_w)

        return tokens.transpose(1, 2).reshape(x.shape[0], -1, patch_h, patch_w)

    def _normalize_token_output(self, output: object, patch_h: int, patch_w: int) -> Tensor:
        if isinstance(output, tuple):
            output = output[0]

        if isinstance(output, dict):
            for key in ("x_norm_patchtokens", "patch_tokens", "x_prenorm", "tokens"):
                if key in output:
                    output = output[key]
                    break
            else:
                raise KeyError("Unable to locate patch token tensor in DINO output.")

        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Unsupported DINO output type: {type(output)!r}")

        if output.ndim == 4:
            return output.flatten(2).transpose(1, 2)

        if output.ndim != 3:
            raise ValueError(f"Expected token tensor with 3 or 4 dims, got shape {tuple(output.shape)}")

        expected_tokens = patch_h * patch_w
        if output.shape[1] == expected_tokens + 1:
            output = output[:, 1:, :]
        elif output.shape[1] != expected_tokens:
            square_tokens = int(math.isqrt(output.shape[1]))
            if square_tokens * square_tokens + 1 == output.shape[1]:
                output = output[:, 1:, :]
            if output.shape[1] != expected_tokens:
                raise ValueError(
                    "Patch token count does not match image resolution. "
                    f"Expected {expected_tokens}, got {output.shape[1]}."
                )

        return output
