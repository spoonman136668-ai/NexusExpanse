"""Decoupled non-parametric episodic memory & surprise-gated Hebbian fast weights."""

from __future__ import annotations

import math
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
    def forward_delta(self, h: torch.Tensor) -> torch.Tensor:
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
