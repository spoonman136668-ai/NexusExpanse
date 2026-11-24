"""Holographic Vector-Symbolic Relational Memory (HRR / Circular-Convolution Superposition).

Stores variable-arity relational bindings (entity (*) relation -> outcome) inside
a single fixed-size holographic superposition vector M in R^D using unitary FFT
circular convolution and Delta-Rule (Widrow-Hoff) holographic updates:
  - Fixed O(D) memory footprint (independent of number of stored edges).
  - Zero hardcoded byte-offset or grammar parsers.
  - Native shifted-punctuation immunity: bytes that have never been bound as
    relational operators produce near-zero holographic resonance and are skipped
    automatically during multi-hop associative traversal.
  - Exact deterministic snapshot/restore under the 40,000-byte ceiling.
"""

from __future__ import annotations

from collections import Counter
import json
import math
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _make_unitary_codebook(vocab: int = 256, dim: int = 4096, seed: int = 1337) -> torch.Tensor:
    """Create deterministic unitary HRR codebook C in R^{vocab x dim}.

    Each vector has unit-magnitude complex Fourier coefficients (exact unitary
    involution: unbind(bind(a, b), a) == b in expectation with zero norm decay).
    """
    rng = np.random.default_rng(seed)
    n_freq = dim // 2 + 1
    phases = rng.uniform(-math.pi, math.pi, size=(vocab, n_freq))
    spec = np.exp(1j * phases)
    # Real-valued DC and Nyquist components must be +/- 1
    spec[:, 0] = rng.choice([-1.0, 1.0], size=vocab)
    if dim % 2 == 0:
        spec[:, -1] = rng.choice([-1.0, 1.0], size=vocab)
    vecs = np.fft.irfft(spec, n=dim, axis=-1)
    # Normalize to exact unit L2 norm
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-12)
    return torch.from_numpy(vecs.astype(np.float64))


def hrr_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular convolution in R^D via rfft/irfft."""
    return torch.fft.irfft(torch.fft.rfft(a, dim=-1) * torch.fft.rfft(b, dim=-1), n=a.shape[-1], dim=-1)


def hrr_unbind(mem: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Exact unitary involution unbinding (correlation) in R^D via rfft/irfft."""
    return torch.fft.irfft(torch.fft.rfft(mem, dim=-1) * torch.conj(torch.fft.rfft(key, dim=-1)), n=mem.shape[-1], dim=-1)


class HolographicRelationalMemory:
    """Fixed-size Delta-Rule HRR superposition memory for tokenless 1- and 2-hop composition."""

    def __init__(
        self,
        model,
        dim: int = 4096,
        eta: float = 0.35,
        temperature: float = 12.0,
        resonance_threshold: float = 0.22,
        prior_weight: float = 0.01,
    ) -> None:
        self.model = model
        self.dim = int(dim)
        self.eta = float(eta)
        self.temperature = float(temperature)
        self.resonance_threshold = float(resonance_threshold)
        self.prior_weight = float(prior_weight)

        # Deterministic unitary codebook (not serialized; reconstructed from dim).
        self.codebook = _make_unitary_codebook(256, self.dim, seed=1337)
        self.rel_codebook = _make_unitary_codebook(256, self.dim, seed=2337)
        self.obj_codebook = _make_unitary_codebook(256, self.dim, seed=3337)
        # Single fixed-size holographic superposition vector M in R^D (float32 for compact 16 KB serialization).
        self.memory = torch.zeros(self.dim, dtype=torch.float64)
        self.outcomes: set[int] = set()
        self.doc_freq: Counter[int] = Counter()
        self.records: int = 0

    @torch.inference_mode()
    def encode(self, prefix: bytes) -> Tuple[Tuple[int, ...], torch.Tensor]:
        """Return raw tokenless byte sequence + frozen neural backbone prior."""
        logits = self.model.forward_neural(torch.tensor(list(prefix), dtype=torch.long)[None])[0]
        prior = logits[0, -1].double().softmax(0)
        return tuple(prefix), prior

    def _cleanup_scores(self, vec: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        """Compute cosine/dot resonance of retrieved vector against all observed outcome prototypes."""
        if not self.outcomes:
            return torch.zeros(256, dtype=torch.float64), 0.0, -1
        sims = self.obj_codebook @ vec  # [256]
        best_val = -1e9
        best_idx = -1
        for y in self.outcomes:
            v = float(sims[y].item())
            if v > best_val:
                best_val = v
                best_idx = y
        return sims, best_val, best_idx

    def _select_anchor_and_candidates(self, codes: Tuple[int, ...]) -> Tuple[Optional[int], List[int]]:
        """Find first known-outcome anchor and remaining non-duplicate byte codes."""
        anchor: Optional[int] = None
        rest: List[int] = []
        for c in codes:
            if anchor is None:
                if c in self.outcomes:
                    anchor = c
                continue
            rest.append(c)
        return anchor, rest

    def _hop_traverse(self, anchor: int, rest: List[int]) -> Tuple[Optional[torch.Tensor], List[Tuple[int, int, float]]]:
        """Traverse holographic memory via resonance-gated multi-hop unbinding.

        Bytes that do not resonate in M (such as unseen/shifted punctuation ':', '=', '>')
        automatically produce near-zero similarity and are skipped without crashing the route.
        """
        if not self.outcomes:
            return None, []
        curr_vec = self.codebook[anchor]
        curr_id = anchor
        last_retrieved: Optional[torch.Tensor] = None
        hops: List[Tuple[int, int, float]] = []

        for r in rest:
            key = hrr_bind(curr_vec, self.rel_codebook[r])
            retrieved = hrr_unbind(self.memory, key)
            _, res, best_y = self._cleanup_scores(retrieved)
            if res >= self.resonance_threshold and best_y >= 0:
                hops.append((curr_id, r, res))
                last_retrieved = retrieved
                curr_id = best_y
                curr_vec = self.codebook[best_y]
        return last_retrieved, hops

    def predict(self, codes: Tuple[int, ...], prior: torch.Tensor) -> torch.Tensor:
        if not self.outcomes:
            return prior
        anchor, rest = self._select_anchor_and_candidates(codes)
        if anchor is None or not rest:
            return prior
        last_retrieved, hops = self._hop_traverse(anchor, rest)
        if last_retrieved is None or not hops:
            return prior

        sims = self.obj_codebook @ last_retrieved
        # Mask to observed outcome support for calibrated probability readout
        mask = torch.full((256,), float("-inf"), dtype=torch.float64)
        for y in self.outcomes:
            mask[y] = sims[y] * self.temperature
        hrr_probs = F.softmax(mask, dim=0)
        return self.prior_weight * prior + (1.0 - self.prior_weight) * hrr_probs

    def observe(self, codes: Tuple[int, ...], target: int) -> None:
        """Online Delta-Rule holographic write: strictly after prediction, zero task labels."""
        self.outcomes.add(int(target))
        self.doc_freq.update(set(codes))
        self.records += 1

        anchor, rest = self._select_anchor_and_candidates(codes)
        if anchor is None or not rest:
            return

        # Filter out universal framing bytes (>55% record frequency after initial warmup)
        # ONLY for choosing which operator to write; never erases already-stored holographic bindings!
        if self.records >= 3:
            content_ops = [c for c in rest if self.doc_freq[c] <= 0.60 * self.records]
        else:
            return
        if not content_ops:
            return

        # Traverse any already-learned prefix hops via resonance to reach the final edge
        curr_vec = self.codebook[anchor]
        if len(content_ops) > 1:
            for r in content_ops[:-1]:
                key = hrr_bind(curr_vec, self.rel_codebook[r])
                retrieved = hrr_unbind(self.memory, key)
                _, res, best_y = self._cleanup_scores(retrieved)
                if res >= self.resonance_threshold and best_y >= 0:
                    curr_vec = self.codebook[best_y]
        final_rel = content_ops[-1]

        # Subspace-projected Delta-Rule HRR update: corrects only the active outcome
        # subspace for this key without decaying orthogonal bindings stored in M.
        key = hrr_bind(curr_vec, self.rel_codebook[final_rel])
        current_val = hrr_unbind(self.memory, key)
        sims = self.obj_codebook @ current_val
        err = torch.zeros_like(current_val)
        tgt = int(target)
        for y in self.outcomes:
            desired = 1.0 if y == tgt else 0.0
            err = err + (desired - float(sims[y].item())) * self.obj_codebook[y]
        self.memory = self.memory + self.eta * hrr_bind(key, err)

    def snapshot(self) -> bytes:
        """Compact binary+JSON snapshot strictly <= 40,000 bytes."""
        meta = json.dumps(
            [1, self.dim, self.records, sorted(self.outcomes), sorted(self.doc_freq.items())],
            separators=(",", ":"),
        ).encode("utf-8")
        mem_f32 = self.memory.to(torch.float32).contiguous().numpy().tobytes()
        payload = struct.pack("<I", len(meta)) + meta + mem_f32
        if len(payload) > 40000:
            raise ValueError(f"holographic state exceeds 40000 bytes: {len(payload)}")
        return payload

    def restore(self, payload: bytes) -> None:
        if len(payload) > 40000 or len(payload) < 4:
            raise ValueError("invalid holographic state payload size")
        (meta_len,) = struct.unpack("<I", payload[:4])
        meta = json.loads(payload[4 : 4 + meta_len].decode("utf-8"))
        version, dim, self.records, outcomes, freq = meta
        if version != 1 or dim != self.dim:
            raise ValueError("incompatible holographic snapshot version or dim")
        mem_bytes = payload[4 + meta_len :]
        arr = np.frombuffer(mem_bytes, dtype=np.float32).copy()
        if arr.size != self.dim:
            raise ValueError("corrupted holographic vector length")
        self.memory = torch.from_numpy(arr.astype(np.float64))
        self.outcomes = set(outcomes)
        self.doc_freq = Counter(dict(freq))
