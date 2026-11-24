"""Decoupled non-parametric episodic memory & surprise-gated Hebbian fast weights."""

from __future__ import annotations

import math
import hashlib
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


class EpisodicByteMemory:
    """Multi-scale non-parametric episodic memory for raw byte streams.

    Decouples factual/pattern memorization from neural weights:
      - Indexes variable-length byte context suffixes (orders 1..max_order) in O(1).
      - Computes a smoothed longest-context-first interpolated byte posterior
        (PPM-inspired Dirichlet/Kneser-Ney-like backoff) strictly causally.
      - Supports instant one-shot online learning during streaming inference with
        zero backprop; conflicting observations can still degrade old predictions.
    """

    def __init__(self, max_order: int = 8, smoothing: float = 0.05) -> None:
        if max_order < 1:
            raise ValueError("max_order must be >= 1")
        self.max_order = max_order
        self.smoothing = float(smoothing)
        # Map: context bytes -> {next_byte: float count}
        self._table: Dict[bytes, Dict[int, float]] = defaultdict(dict)
        self._totals: Dict[bytes, float] = defaultdict(float)
        self._unigram = [1.0] * 256
        self._unigram_total = 256.0
        self.num_observations = 0

    def observe_transition(self, context: bytes, next_byte: int, weight: float = 1.0) -> None:
        """Record a single causal (context -> next_byte) transition across all suffix orders."""
        max_len = min(len(context), self.max_order)
        for order in range(1, max_len + 1):
            key = context[-order:]
            bucket = self._table[key]
            bucket[next_byte] = bucket.get(next_byte, 0.0) + weight
            self._totals[key] += weight
        self._unigram[next_byte] += weight
        self._unigram_total += weight
        self.num_observations += 1

    def ingest_stream(self, data: bytes, weight: float = 1.0) -> None:
        """Ingest a raw byte stream into episodic memory in O(N * max_order)."""
        n = len(data)
        for i in range(n - 1):
            start = max(0, i + 1 - self.max_order)
            self.observe_transition(data[start : i + 1], data[i + 1], weight=weight)

    def predict_probs(self, context: bytes) -> Tuple[torch.Tensor, float]:
        """Return (256-dim probability vector, confidence in [0, 1]) strictly from past context."""
        # Start from unigram prior
        probs = torch.tensor(self._unigram, dtype=torch.float64) / self._unigram_total
        best_confidence = 0.0

        max_len = min(len(context), self.max_order)
        # Hierarchical backoff interpolation from shortest to longest matching suffix
        for order in range(1, max_len + 1):
            key = context[-order:]
            total = self._totals.get(key, 0.0)
            if total <= 0.0:
                continue
            bucket = self._table[key]
            distinct = float(len(bucket))
            # Escape weight shrinks as evidence grows and distinct continuations stay focused
            lam = total / (total + distinct * 0.75 + self.smoothing * 4.0)
            # Longer exact matches earn higher trust weight
            order_boost = min(0.98, lam * (0.65 + 0.05 * order))
            probs = probs * (1.0 - order_boost)
            scale = order_boost / total
            for b_val, cnt in bucket.items():
                probs[b_val] += cnt * scale
            best_confidence = max(best_confidence, order_boost)

        probs = probs / probs.sum()
        return probs, float(best_confidence)

    def memory_entry_count(self) -> int:
        return len(self._table)


class HebbianFastWeightAdapter(nn.Module):
    """Surprise-gated local Hebbian fast-weight scratchpad (zero backprop at test time).

    Maintains a dense fast weight state W_fast [dim, 256] that adapts
    online via local error-modulated Hebbian updates whenever prediction surprise
    exceeds a threshold, with exponential decay so stale transient styles fade.
    """

    def __init__(self, dim: int, vocab_size: int = 256, lr: float = 0.05, decay: float = 0.995) -> None:
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.lr = float(lr)
        self.decay = float(decay)
        self.register_buffer("w_fast", torch.zeros(dim, vocab_size, dtype=torch.float32))
        self.updates_applied = 0

    def reset_fast_state(self) -> None:
        self.w_fast.zero_()
        self.updates_applied = 0

    @torch.inference_mode()
    def forward_delta(self, h: torch.Tensor, *, context: bytes = b"") -> torch.Tensor:
        """Compute fast-weight logit residual from hidden state h [..., dim]."""
        return h @ self.w_fast

    @torch.inference_mode()
    def hebbian_update(
        self,
        h: torch.Tensor,
        probs: torch.Tensor,
        target_byte: int,
        surprise_nll: float,
        surprise_threshold: float = 1.5,
        *, context: bytes = b"",
    ) -> bool:
        """Local delta-rule Hebbian update triggered ONLY on high-surprise events."""
        self.w_fast.mul_(self.decay)
        if surprise_nll < surprise_threshold:
            return False
        # Normalized pre-synaptic vector h [dim]
        h_vec = h.detach().reshape(-1).float()
        h_norm = h_vec / (h_vec.norm() + 1e-6)
        # Post-synaptic error signal (target_onehot - predicted_probs)
        err = -probs.detach().reshape(-1).float()
        err[target_byte] += 1.0
        # Bounded outer-product Hebbian write
        step = self.lr * min(2.0, surprise_nll / max(surprise_threshold, 1e-6))
        self.w_fast.add_(torch.outer(h_norm, err) * step)
        self.w_fast.clamp_(-2.0, 2.0)
        self.updates_applied += 1
        return True


class ContextBankFastWeightAdapter(HebbianFastWeightAdapter):
    """DEV-only fixed-budget banks, routed by observed byte suffix, not task labels.

    Each bank uses a disjoint hidden feature slice. Inactive banks neither decay
    nor receive updates. Hash collisions and loss of feature capacity are expected.
    """

    def __init__(self, dim: int, vocab_size: int = 256, lr: float = .05,
                 decay: float = .995, banks: int = 2, context_order: int = 16):
        if type(banks) is not int or banks < 2 or dim % banks:
            raise ValueError('banks must divide hidden dimension')
        if type(context_order) is not int or not 1 <= context_order <= 128:
            raise ValueError('invalid context order')
        super().__init__(dim, vocab_size, lr, decay)
        self.banks = banks
        self.context_order = context_order

    def _slice(self, context: bytes):
        key = hashlib.blake2s(context[-self.context_order:], digest_size=8).digest()
        bank = int.from_bytes(key, 'little') % self.banks
        width = self.dim // self.banks
        return slice(bank * width, (bank + 1) * width)

    @torch.inference_mode()
    def forward_delta(self, h: torch.Tensor, *, context: bytes = b''):
        sl = self._slice(context)
        return h[..., sl] @ self.w_fast[sl]

    @torch.inference_mode()
    def hebbian_update(self, h, probs, target_byte, surprise_nll,
                       surprise_threshold=1.5, *, context: bytes = b''):
        sl = self._slice(context)
        w = self.w_fast[sl]
        w.mul_(self.decay)
        if surprise_nll < surprise_threshold:
            return False
        h_vec = h.detach().reshape(-1).float()[sl]
        h_norm = h_vec / (h_vec.norm() + 1e-6)
        err = -probs.detach().reshape(-1).float()
        err[target_byte] += 1.0
        step = self.lr * min(2.0, surprise_nll / max(surprise_threshold, 1e-6))
        w.add_(torch.outer(h_norm, err) * step)
        w.clamp_(-2.0, 2.0)
        self.updates_applied += 1
        return True


class ProjectedFastWeightAdapter(HebbianFastWeightAdapter):
    """Opt-in no-decay plasticity with a bounded causal orthonormal protection basis.

    Admit directions after observed confident correct predictions; project new
    updates into the orthogonal complement. No task labels or stored answers.
    A full basis is never silently evicted; acquisition can consequently stall.
    """
    def __init__(self, dim, vocab_size=256, lr=.05, rank=8):
        if type(rank) is not int or not 1 <= rank < dim:
            raise ValueError('invalid protection rank')
        super().__init__(dim, vocab_size, lr, decay=1.)
        self.rank = rank
        self.register_buffer('basis', torch.zeros(rank, dim, dtype=torch.float32))

    def reset_fast_state(self):
        super().reset_fast_state()
        self.basis.zero_()

    @torch.inference_mode()
    def hebbian_update(self, h, probs, target_byte, surprise_nll,
                       surprise_threshold=1.5, *, context=b''):
        v = h.detach().reshape(-1).float()
        v = v / (v.norm() + 1e-6)
        q = self.basis
        # Reorthogonalize to limit floating point drift.
        u = v - q.T @ (q @ v)
        u = u - q.T @ (q @ u)
        if float(probs[target_byte]) >= .5 and self.updates_applied:
            slots = (q.square().sum(dim=1) == 0).nonzero().flatten()
            if slots.numel() and float(u.norm()) >= .1:
                q[slots[0]].copy_(u / u.norm())
            return False
        if surprise_nll < surprise_threshold or float(u.norm()) < 1e-6:
            return False
        err = -probs.detach().reshape(-1).float()
        err[target_byte] += 1.
        delta = torch.outer(u, err) * (self.lr * min(2., surprise_nll / max(surprise_threshold, 1e-6)))
        # A scalar line search preserves the projected direction, unlike clipping each weight.
        bound = torch.where(delta > 0, (2. - self.w_fast) / delta,
                            torch.where(delta < 0, (-2. - self.w_fast) / delta, torch.inf))
        scale = min(1., max(0., float(bound.min()) * (1. - 1e-6)))
        if scale == 0.: return False
        self.w_fast.add_(delta * scale)
        self.updates_applied += 1
        return True
