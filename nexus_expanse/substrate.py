"""Sparse full-rank compute substrate for Nexus Expanse (DirectMonarch + Causal Gating)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def rank_safe_blocks(dim: int, hidden: int, minimum: int = 8) -> int:
    """Return smallest block count G >= minimum dividing dim & hidden with G^2 >= min(dim, hidden)."""
    if min(dim, hidden, minimum) < 1:
        raise ValueError("positive dimensions and block count required")
    for g in range(minimum, math.gcd(dim, hidden) + 1):
        if dim % g == 0 and hidden % g == 0 and g * g >= min(dim, hidden):
            return g
    raise ValueError(f"no rank-safe common divisor for dim={dim}, hidden={hidden}")


class DirectMonarchLinear(nn.Module):
    """Full-rank 2-stage Monarch butterfly linear layer (zero dense matrix materialization)."""

    def __init__(self, in_dim: int, out_dim: int, n_blocks: int = 16, bias: bool = True) -> None:
        super().__init__()
        if in_dim % n_blocks != 0 or out_dim % n_blocks != 0:
            raise ValueError(f"in_dim={in_dim}, out_dim={out_dim} not divisible by n_blocks={n_blocks}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_blocks = n_blocks
        self.b_in = in_dim // n_blocks
        self.b_out = out_dim // n_blocks

        self.b1 = nn.Parameter(torch.randn(n_blocks, self.b_in, n_blocks) / math.sqrt(self.b_in))
        self.b2 = nn.Parameter(torch.randn(n_blocks, n_blocks, self.b_out) / math.sqrt(n_blocks))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        xg = x.reshape(-1, self.n_blocks, self.b_in)
        y1 = torch.einsum("ngi,gij->ngj", xg, self.b1)
        y1_t = y1.transpose(1, 2).contiguous()
        y2 = torch.einsum("ngj,gjo->ngo", y1_t, self.b2)
        out = y2.reshape(*orig_shape, self.out_dim)
        if self.bias is not None:
            out = out + self.bias
        return out


class CausalGatedMonarchFFN(nn.Module):
    """Full-rank DirectMonarch FFN with strictly causal pre-FFN uncertainty gating."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_blocks: int = 16,
        target_skip_rate: float = 0.50,
        enable_skip: bool = True,
        ffn_kind: str = "monarch",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.target_skip_rate = float(target_skip_rate)
        self.enable_skip = bool(enable_skip)

        self.gate_proj = nn.Linear(dim, 1)
        self.register_buffer("tau_eval", torch.tensor(float("-inf"), dtype=torch.float32))
        if ffn_kind == "monarch":
            self.fc1 = DirectMonarchLinear(dim, hidden_dim, n_blocks=n_blocks)
            self.fc2 = DirectMonarchLinear(hidden_dim, dim, n_blocks=n_blocks)
        elif ffn_kind == "dense":
            self.fc1 = nn.Linear(dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, dim)
        else:
            raise ValueError(f"unknown FFN kind: {ffn_kind}")

    def gate_scores(self, h: torch.Tensor) -> torch.Tensor:
        return self.gate_proj(h.detach()).squeeze(-1)

    def forward(self, h: torch.Tensor, valid_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self.gate_scores(h)
        if not self.enable_skip:
            deep = torch.ones_like(scores, dtype=torch.bool)
        else:
            deep = scores >= self.tau_eval

        if valid_length is not None:
            deep[:, valid_length:] = False
        out = torch.zeros_like(h)
        if bool(deep.any().item()):
            selected = h[deep]
            if self.enable_skip and not self.training:
                # A compacted sparse batch also changes shape when the suffix changes.
                # Use fixed-size kernels, padding only the final group of active rows.
                deltas = []
                for group in selected.split(32):
                    block = F.pad(group, (0, 0, 0, 32 - len(group)))
                    deltas.append(self.fc2(F.silu(self.fc1(block)))[:len(group)])
                out[deep] = torch.cat(deltas)
            else:
                out[deep] = self.fc2(F.silu(self.fc1(selected)))
        return out, scores, deep


def static_footprint_bytes(module: nn.Module) -> Tuple[int, int]:
    elems = 0
    nbytes = 0
    for t in list(module.parameters()) + list(module.buffers()):
        elems += t.numel()
        nbytes += t.numel() * t.element_size()
    return elems, nbytes
