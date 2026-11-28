"""Frozen three-byte suffix memory for a newly born raw-byte receiver."""
from __future__ import annotations

import math

import torch
from torch import nn

from nexus_expanse.checkpoint_transport import ImportedSkill
from nexus_expanse.unified_receiver import UnifiedSnapshotSkill


SUFFIX_BYTES = 3
MEMORY_WEIGHT = 0.9


class EpisodicAssistedBornSkill(nn.Module):
    """Blend a neural forecast with a stored next byte on exact suffix hits.

    Fit a compact table from observed byte transitions. A suffix with conflicting
    successors falls back to the neural receiver, as does an unseen suffix.
    """

    def __init__(self, neural: ImportedSkill | UnifiedSnapshotSkill, keys: torch.Tensor, successors: torch.Tensor):
        super().__init__()
        if type(neural) not in (ImportedSkill, UnifiedSnapshotSkill):
            raise ValueError("episodic memory requires a recognized neural receiver")
        if (
            not isinstance(keys, torch.Tensor) or keys.dtype != torch.int32 or keys.ndim != 1
            or not isinstance(successors, torch.Tensor) or successors.dtype != torch.uint8
            or successors.shape != keys.shape
            or keys.device != successors.device
            or bool(((keys < 0) | (keys >= 1 << (8 * SUFFIX_BYTES))).any())
            or bool((keys[1:] <= keys[:-1]).any())
        ):
            raise ValueError("invalid episodic suffix table")
        self.neural = neural
        self.register_buffer("keys", keys)
        self.register_buffer("successors", successors)
        self.requires_grad_(False).eval()

    @property
    def core(self):
        return self.neural.core

    @classmethod
    def from_encounter(cls, neural, raw):
        successors = {}
        for line in raw.tolist():
            stream = bytes(line)
            for t in range(SUFFIX_BYTES, len(stream)):
                key = int.from_bytes(stream[t - SUFFIX_BYTES:t], "big")
                value = stream[t]
                if successors.setdefault(key, value) != value:
                    successors[key] = -1
        entries = sorted((key, value) for key, value in successors.items() if value >= 0)
        keys = torch.tensor([key for key, _ in entries], dtype=torch.int32, device=raw.device)
        values = torch.tensor([value for _, value in entries], dtype=torch.uint8, device=raw.device)
        return cls(neural, keys, values)

    def forward(self, raw):
        logits = self.neural(raw)
        if raw.size(1) < SUFFIX_BYTES or not self.keys.numel():
            return logits

        observed = raw.int()
        query = ((observed[:, :-2] << 16) | (observed[:, 1:-1] << 8) | observed[:, 2:]).contiguous()
        index = torch.searchsorted(self.keys, query).clamp_max(self.keys.numel() - 1)
        hit = self.keys[index] == query
        if not hit.any():
            return logits

        result = logits.clone()
        suffix_logits = result[:, SUFFIX_BYTES - 1:]
        mixed = suffix_logits[hit].float().log_softmax(-1) + math.log1p(-MEMORY_WEIGHT)
        next_byte = self.successors[index[hit]].long().unsqueeze(-1)
        neural_logp = mixed.gather(-1, next_byte)
        mixed.scatter_(-1, next_byte, torch.logaddexp(neural_logp, neural_logp.new_full((), math.log(MEMORY_WEIGHT))))
        suffix_logits[hit] = mixed.to(logits.dtype)
        return result
