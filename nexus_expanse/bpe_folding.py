"""Causal chunk-state folding for multi-byte BPE checkpoint-only transfer.

Zero BPE merges, zero tokenizer calls, and zero donor execution in the receiver.
"""

from __future__ import annotations

import collections
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus_expanse.checkpoint_transport import CONFIG, CorrectedLinear, ImportedSkill, load_skill
from nexus_expanse.donor_assimilation import forbid_donor_forward


def build_static_join_table(vocab_bytes: List[bytes], threshold: float = 0.35) -> torch.Tensor:
    """Build a static 256x256 causal continuation mask from vocab_bytes alone (no BPE merges)."""
    counts = torch.zeros((256, 256), dtype=torch.float32)
    for tok in vocab_bytes[256:]:
        for a, b in zip(tok[:-1], tok[1:]):
            counts[a, b] += 1.0
    if counts.max() > 0:
        row_sum = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        probs = counts / row_sum
        return (probs >= threshold) & (counts >= 1.0)
    return torch.zeros((256, 256), dtype=torch.bool)


def build_entropy_join_table() -> torch.Tensor:
    """Pure byte-class continuation table (zero vocab needed): group contiguous word/digit bytes."""
    is_word = torch.zeros(256, dtype=torch.bool)
    for b in range(256):
        ch = chr(b)
        if ch.isalnum() or ch == "_":
            is_word[b] = True
    return is_word.unsqueeze(1) & is_word.unsqueeze(0)


@torch.no_grad()
def compute_causal_chunk_schedule(
    raw: torch.Tensor,
    join_table: torch.Tensor,
    max_chunk_len: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute strictly causal chunk boundaries, chunk position indices, and chunk-end attention mask.

    At step t:
      - starts_new[b, t] depends ONLY on raw[b, t-1], raw[b, t], and chunk_len[b, t-1].
      - chunk_pos[b, t] = cumsum(starts_new[b, :t+1]) - 1.
      - Key j < t is visible to query t iff starts_new[b, j+1] is True (finalized chunk end).
        Key j == t is always visible (current chunk state).
        Since j + 1 <= t, visibility of j to t depends ONLY on raw[b, :t+1].
    """
    bsz, seq_len = raw.shape
    device = raw.device
    join_dev = join_table.to(device=device)

    starts_new = torch.ones((bsz, seq_len), dtype=torch.bool, device=device)
    chunk_pos = torch.zeros((bsz, seq_len), dtype=torch.long, device=device)
    chunk_lens = torch.ones((bsz,), dtype=torch.long, device=device)

    for t in range(1, seq_len):
        can_join = join_dev[raw[:, t - 1], raw[:, t]] & (chunk_lens < max_chunk_len)
        starts_new[:, t] = ~can_join
        chunk_lens = torch.where(can_join, chunk_lens + 1, torch.ones_like(chunk_lens))
        chunk_pos[:, t] = chunk_pos[:, t - 1] + (~can_join).long()

    # Build per-batch causal float attention mask: shape (B * n_heads, T, T) or (B, T, T).
    # Key j is a finalized chunk end wrt t > j iff starts_new[:, j+1] is True.
    is_final_end = torch.zeros((bsz, seq_len), dtype=torch.bool, device=device)
    if seq_len > 1:
        is_final_end[:, :-1] = starts_new[:, 1:]

    eye = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0)
    tril = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=-1).unsqueeze(0)
    allowed = eye | (tril & is_final_end.unsqueeze(1))

    attn_mask = torch.full((bsz, seq_len, seq_len), float("-inf"), dtype=torch.float32, device=device)
    attn_mask = attn_mask.masked_fill(allowed, 0.0)
    return starts_new, chunk_pos, attn_mask


class SuffixEmbedding(nn.Module):
    def __init__(self, emb: nn.Embedding, keys: torch.Tensor, delta: torch.Tensor) -> None:
        super().__init__()
        self.weight = emb.weight
        self.register_buffer("keys", keys)
        self.register_buffer("delta", delta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = F.embedding(x, self.weight)
        if x.size(1) <= 1 or len(self.keys) == 0:
            return value
        code = x[:, :-1] * 256 + x[:, 1:]
        index = torch.searchsorted(self.keys, code).clamp_max(len(self.keys) - 1)
        correction = self.delta[index] * (self.keys[index] == code).unsqueeze(-1)
        return value + F.pad(correction, (0, 0, 1, 0))


class FoldedBoundarySkill(nn.Module):
    """Raw-byte receiver that folds intra-chunk state and aligns attention to chunk-end slots."""

    def __init__(
        self,
        donor_path: Path,
        ffn_method: str,
        mode: str,
        *,
        max_chunk_len: int = 4,
        fold_alpha: float = 0.65,
    ) -> None:
        super().__init__()
        if mode not in ("entropy_fold", "static_trie_fold", "shuffled_entropy_fold", "shuffled_static_trie_fold"):
            raise ValueError(f"Unsupported fold mode: {mode}")

        self.mode = mode
        self.max_chunk_len = max_chunk_len
        self.fold_alpha = fold_alpha

        with forbid_donor_forward() as guard, torch.no_grad():
            self.skill = load_skill(donor_path, ffn_method)
            info = torch.load(donor_path, weights_only=True, map_location="cpu")
            vocab = [bytes.fromhex(s) for s in info["spellings"]]
            base_mode = mode
            if mode.startswith("shuffled_"):
                rest = vocab[256:]
                random.Random(9341).shuffle(rest)
                vocab = vocab[:256] + rest
                base_mode = mode.removeprefix("shuffled_")

            state = info["weights"]
            self.register_buffer("first", torch.tensor([v[0] for v in vocab], dtype=torch.long))
            self.skill.readout = nn.Linear(CONFIG["dim"], len(vocab), bias=False)
            self.skill.readout.weight.copy_(state["readout.weight"])

            residual = collections.defaultdict(list)
            for i, v in enumerate(vocab[256:], 256):
                key = 256 * v[-2] + v[-1]
                residual[key].append(state["emb.weight"][i] - state["emb.weight"][v[-1]])
            keys = sorted(residual)
            delta = 0.75 * torch.stack([torch.stack(residual[k]).mean(0) for k in keys])
            self.skill.core.emb = SuffixEmbedding(
                self.skill.core.emb,
                torch.tensor(keys, dtype=torch.long),
                delta,
            )

            if base_mode == "entropy_fold":
                join_table = build_entropy_join_table()
            else:
                join_table = build_static_join_table(vocab)
            self.register_buffer("join_table", join_table)

            if guard.donor_forward_calls:
                raise RuntimeError("donor execution during FoldedBoundarySkill init")

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = raw.shape
        starts_new, chunk_pos, per_batch_mask = compute_causal_chunk_schedule(
            raw, self.join_table, max_chunk_len=self.max_chunk_len
        )

        # Causal intra-chunk hidden folding before positional encoding
        emb = self.skill.core.emb(raw)
        folded = torch.empty_like(emb)
        folded[:, 0] = emb[:, 0]
        alpha = self.fold_alpha
        for t in range(1, seq_len):
            cont = (~starts_new[:, t]).unsqueeze(-1).to(emb.dtype)
            prev_fold = (1.0 - alpha) * folded[:, t - 1] + alpha * emb[:, t]
            folded[:, t] = cont * prev_fold + (1.0 - cont) * emb[:, t]

        pos = self.skill.core.pos_emb(chunk_pos.clamp_max(self.skill.core.max_seq_len - 1))
        x = self.skill.core.drop(folded + pos)

        n_heads = self.skill.core.layers[0].attn.num_heads
        attn_mask = per_batch_mask.repeat_interleave(n_heads, dim=0)

        for layer in self.skill.core.layers:
            x = layer.attention_residual(x, attn_mask)
            h_ffn = layer.ffn_norm(x)
            delta, _, _ = layer.ffn(h_ffn, valid_length=seq_len)
            x = x + layer.ffn_drop(delta)

        hidden = self.skill.core.out_norm(x)
        logits = self.skill.readout(hidden)
        p = logits.double().softmax(-1)
        out = p.new_zeros(*p.shape[:-1], 256)
        return out.scatter_add(-1, self.first.expand_as(p), p).clamp_min(1e-300).log()


def rebuild_folded_skill_from_state(state: dict, ffn_method: str, mode: str) -> FoldedBoundarySkill:
    """Reconstruct a FoldedBoundarySkill purely from saved state_dict for cold-restore verification."""
    skill = ImportedSkill()
    for layer in skill.core.layers:
        for name in ("fc1", "fc2"):
            base = getattr(layer.ffn, name)
            if ffn_method != "monarch":
                setattr(layer.ffn, name, CorrectedLinear(base, torch.zeros(base.out_dim, base.in_dim), ffn_method))
    skill.core.emb = SuffixEmbedding(
        skill.core.emb,
        state["skill.core.emb.keys"].clone(),
        state["skill.core.emb.delta"].clone(),
    )
    m = FoldedBoundarySkill.__new__(FoldedBoundarySkill)
    nn.Module.__init__(m)
    m.mode = mode
    m.max_chunk_len = 4
    m.fold_alpha = 0.65
    m.skill = skill
    m.skill.readout = nn.Linear(CONFIG["dim"], len(state["first"]), bias=False)
    m.register_buffer("first", state["first"].clone())
    m.register_buffer("join_table", state["join_table"].clone())
    m.load_state_dict(state)
    return m
