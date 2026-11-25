"""Compact one-time offline byte-marginalized KL cache for tokenless student assimilation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

from nexus_expanse.donor_assimilation import canonical_byte_bpe


@torch.inference_mode()
def build_topk_byte_cache(
    donor_model: torch.nn.Module,
    vocab: list[bytes],
    merges: list[tuple[int, int]],
    x_bytes: torch.Tensor,
    top_k: int = 16,
) -> dict:
    """Run donor ONCE offline on x_bytes to export compact top-k next-byte distributions.

    Returns a plain dict of tensors (uint8 indices, float16 probs) with zero donor references.
    """
    indices = {pair: i + 256 for i, pair in enumerate(merges)}
    first = torch.tensor([v[0] for v in vocab], dtype=torch.long)
    bsz, seq_len = x_bytes.shape

    queries = [
        canonical_byte_bpe(bytes(row[:t]), indices)
        for row in x_bytes.tolist()
        for t in range(1, seq_len + 1)
    ]
    outputs = []
    for start in range(0, len(queries), 64):
        group = queries[start : start + 64]
        lengths = torch.tensor([len(q) for q in group], dtype=torch.long)
        padded = torch.zeros((len(group), int(lengths.max())), dtype=torch.long)
        for i, q in enumerate(group):
            padded[i, : len(q)] = torch.tensor(q, dtype=torch.long)
        logits = donor_model(padded)[torch.arange(len(group)), lengths - 1].double()
        probs = logits.softmax(-1)
        compact = torch.zeros((len(group), 256), dtype=torch.float64)
        compact.scatter_add_(1, first.expand_as(probs), probs)
        outputs.append(compact)

    full_probs = torch.cat(outputs, dim=0).view(bsz, seq_len, 256)
    vals, idx = full_probs.topk(top_k, dim=-1)
    vals = (vals / vals.sum(dim=-1, keepdim=True).clamp_min(1e-12)).to(torch.float16)
    idx_u8 = idx.to(torch.uint8)
    raw_bytes = idx_u8.numpy().tobytes() + vals.numpy().tobytes()
    return {
        "top_k": top_k,
        "indices": idx_u8,
        "probs": vals,
        "shape": (bsz, seq_len, 256),
        "payload_bytes": len(raw_bytes),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def save_offline_cache(cache: dict, path: Path) -> None:
    torch.save(cache, path)


def load_offline_cache(path: Path) -> dict:
    return torch.load(path, weights_only=True, map_location="cpu")


def offline_kl_loss(
    student_log_probs: torch.Tensor,
    targets: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_probs: torch.Tensor,
    kl_weight: float = 0.5,
) -> torch.Tensor:
    """Combined ground-truth CE + cached top-k KL(P_cache || P_student) loss."""
    ce = F.cross_entropy(student_log_probs.float().flatten(0, 1), targets.flatten())
    log_q = F.log_softmax(student_log_probs.float(), dim=-1)
    q_gathered = log_q.gather(-1, cache_indices.long())
    p = cache_probs.float().clamp_min(1e-7)
    p = p / p.sum(dim=-1, keepdim=True)
    kl = (p * (p.log() - q_gathered)).sum(dim=-1).mean()
    return (1.0 - kl_weight) * ce + kl_weight * kl
