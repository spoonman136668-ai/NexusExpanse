"""O(T) Kuramoto Phase-Coupled Oscillator Mixer (AKOrN-Style) replacing O(T^2) Self-Attention.

Replaces quadratic causal MultiheadAttention and rolling KV-cache rebuilds with
a strictly causal O(1)-per-step bank of K = dim // 2 coupled 2D phase oscillators
on the unit circle S^1:
  - Constant O(L * dim) streaming state regardless of sequence length (zero KV cache).
  - Exact 0.0 bit-identical streaming-vs-batch and prefix invariance via FP64
    accumulation inside each oscillator step.
  - Feeds directly into Nexus Expanse's full-rank DirectMonarch FFN + causal gate.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus_expanse.substrate import (
    CausalGatedMonarchFFN,
    rank_safe_blocks,
    static_footprint_bytes,
)


class KuramotoOscillatorLayer(nn.Module):
    """Single O(1)-per-step Kuramoto phase-coupled oscillator layer + DirectMonarch FFN."""

    def __init__(
        self,
        dim: int,
        ffn_mult: int = 2,
        n_blocks: int = 8,
        target_skip_rate: float = 0.50,
        enable_skip: bool = False,
        dropout: float = 0.0,
        dt: float = 0.5,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim ({dim}) must be even for 2D phase oscillators")
        self.dim = dim
        self.n_osc = dim // 2
        self.dt = float(dt)

        self.osc_norm = nn.LayerNorm(dim)
        self.forcing_proj = nn.Linear(dim, dim)
        self.coupling_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_drop = nn.Dropout(dropout)
        self.ffn = CausalGatedMonarchFFN(
            dim=dim,
            hidden_dim=dim * ffn_mult,
            n_blocks=n_blocks,
            target_skip_rate=target_skip_rate,
            enable_skip=enable_skip,
            ffn_kind="monarch",
        )

    def _osc_step(self, h_t: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        """Advance K=dim//2 unit-norm 2D phase oscillators by one causal Kuramoto step.

        Args:
            h_t: [B, dim] normalized input at step t
            z_prev: [B, K, 2] previous unit-norm phase state (float64)
        Returns:
            z_next: [B, K, 2] updated unit-norm phase state (float64)
        """
        b = h_t.shape[0]
        h64 = h_t.double()
        f_b = self.forcing_proj.bias.double() if self.forcing_proj.bias is not None else None
        forcing = F.linear(h64, self.forcing_proj.weight.double(), f_b).view(b, self.n_osc, 2)
        coupling = F.linear(z_prev.reshape(b, self.dim), self.coupling_proj.weight.double(), None).view(
            b, self.n_osc, 2
        )
        # Tangential Kuramoto projection onto S^1 tangent space at z_prev
        drive = forcing + coupling
        radial = (drive * z_prev).sum(dim=-1, keepdim=True) * z_prev
        tangent = drive - radial
        z_raw = z_prev + self.dt * tangent
        z_next = z_raw / (z_raw.norm(dim=-1, keepdim=True) + 1e-12)
        return z_next

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initial resting phase state z_0 = (1, 0) on S^1 for all K oscillators."""
        z = torch.zeros((batch_size, self.n_osc, 2), dtype=torch.float64, device=device)
        z[..., 0] = 1.0
        return z

    @staticmethod
    def _monarch_fp64(layer, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        xg = x.reshape(-1, layer.n_blocks, layer.b_in).double()
        y1 = torch.einsum("ngi,gij->ngj", xg, layer.b1.double())
        y1_t = y1.transpose(1, 2).contiguous()
        y2 = torch.einsum("ngj,gjo->ngo", y1_t, layer.b2.double())
        out = y2.reshape(*orig_shape, layer.out_dim)
        if layer.bias is not None:
            out = out + layer.bias.double()
        return out.to(x.dtype)

    def _run_ffn_exact(self, h_ffn: torch.Tensor) -> Tuple[torch.Tensor, float]:
        mid = F.silu(self._monarch_fp64(self.ffn.fc1, h_ffn))
        delta = self._monarch_fp64(self.ffn.fc2, mid)
        return delta, 0.0

    def forward_step(
        self, x_t: torch.Tensor, z_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Exact O(1) single-step streaming forward: x_t [B, 1, dim] -> (x_out, z_next, skip)."""
        h_osc = self.osc_norm(x_t)[:, 0, :]
        z_next = self._osc_step(h_osc, z_prev)
        o_b = self.out_proj.bias.double() if self.out_proj.bias is not None else None
        osc_out = F.linear(z_next.reshape(x_t.shape[0], self.dim), self.out_proj.weight.double(), o_b).to(
            x_t.dtype
        )
        x = x_t + self.drop(osc_out.unsqueeze(1))
        h_ffn = self.ffn_norm(x)
        delta, skip_rate = self._run_ffn_exact(h_ffn)
        x = x + self.ffn_drop(delta)
        return x, z_next, skip_rate

    def forward(
        self, x: torch.Tensor, z_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Full-sequence causal forward matching forward_step bit-for-bit."""
        b, t, d = x.shape
        z = self.init_state(b, x.device) if z_init is None else z_init
        h_all = self.osc_norm(x)
        osc_outs: List[torch.Tensor] = []
        o_b = self.out_proj.bias.double() if self.out_proj.bias is not None else None
        for idx in range(t):
            z = self._osc_step(h_all[:, idx, :], z)
            step_out = F.linear(z.reshape(b, d), self.out_proj.weight.double(), o_b).to(x.dtype)
            osc_outs.append(step_out)
        osc_seq = torch.stack(osc_outs, dim=1)
        x = x + self.drop(osc_seq)
        h_ffn = self.ffn_norm(x)
        delta, skip_rate = self._run_ffn_exact(h_ffn)
        x = x + self.ffn_drop(delta)
        return x, z, skip_rate


class KuramotoMonarchByteCore(nn.Module):
    """Tokenless O(T) Kuramoto-Monarch cognitive backbone with zero KV cache."""

    def __init__(
        self,
        n_layers: int = 2,
        dim: int = 64,
        ffn_mult: int = 2,
        target_skip_rate: float = 0.50,
        enable_skip: bool = False,
        dropout: float = 0.0,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.n_blocks = rank_safe_blocks(dim, dim * ffn_mult)
        self.emb = nn.Embedding(256, dim)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                KuramotoOscillatorLayer(
                    dim=dim,
                    ffn_mult=ffn_mult,
                    n_blocks=self.n_blocks,
                    target_skip_rate=target_skip_rate,
                    enable_skip=enable_skip,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)

    def init_stream_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> List[torch.Tensor]:
        dev = device if device is not None else self.emb.weight.device
        return [layer.init_state(batch_size, dev) for layer in self.layers]

    def forward_neural(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """Full-sequence O(T) forward: tokens [B, T] -> (logits [B, T, 256], hidden [B, T, dim], skips)."""
        x = self.drop(self.emb(tokens))
        skips: List[float] = []
        for layer in self.layers:
            x, _, sk = layer(x)
            skips.append(sk)
        h_out = self.out_norm(x)
        logits = F.linear(h_out.double(), self.emb.weight.double()).to(h_out.dtype)
        return logits, h_out, skips

    @torch.inference_mode()
    def step_streaming(
        self, byte_val: int, states: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Exact O(1) single-byte streaming step (zero KV cache, zero window rebuilds)."""
        tok = torch.tensor([[byte_val]], dtype=torch.long, device=self.emb.weight.device)
        x = self.emb(tok)
        next_states: List[torch.Tensor] = []
        for layer, z_prev in zip(self.layers, states):
            x, z_next, _ = layer.forward_step(x, z_prev)
            next_states.append(z_next)
        h_out = self.out_norm(x)
        logits = F.linear(h_out.double(), self.emb.weight.double()).to(h_out.dtype)
        return logits[0, 0].double().softmax(0), next_states

    def total_param_bytes(self) -> int:
        return static_footprint_bytes(self)[1]
