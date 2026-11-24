"""NexusExpanseByteCore: Tokenless Monarch + Causal Gate + Episodic Memory + Fast Weights."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus_expanse.arbitration import ReliabilityArbitrator
from nexus_expanse.memory import EpisodicByteMemory, HebbianFastWeightAdapter
from nexus_expanse.substrate import (
    CausalGatedMonarchFFN,
    rank_safe_blocks,
    static_footprint_bytes,
)


class NexusLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_mult: int,
        n_blocks: int,
        target_skip_rate: float,
        enable_skip: bool,
        dropout: float,
        ffn_kind: str = "monarch",
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_drop = nn.Dropout(dropout)
        self.ffn = CausalGatedMonarchFFN(
            dim=dim,
            hidden_dim=dim * ffn_mult,
            n_blocks=n_blocks,
            target_skip_rate=target_skip_rate,
            enable_skip=enable_skip,
            ffn_kind=ffn_kind,
        )

    def attention_residual(self, x, causal_mask):
        h = self.attn_norm(x)
        attn, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        return x + self.drop(attn)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        out_norm: nn.LayerNorm,
        emb_weight: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        valid_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        x = self.attention_residual(x, causal_mask)
        h_ffn = self.ffn_norm(x)
        delta, scores, deep = self.ffn(h_ffn, valid_length=valid_length)
        skip_rate = float((~deep[:, :valid_length]).float().mean().item())

        aux = torch.zeros((), device=x.device)
        if self.ffn.enable_skip and targets is not None:
            with torch.no_grad():
                pre_logits = F.linear(out_norm(x[:, :valid_length]), emb_weight)
                pre_nll = F.cross_entropy(
                    pre_logits.reshape(-1, 256), targets.reshape(-1), reduction="none"
                ).view_as(targets)
            aux = F.mse_loss(scores[:, :valid_length], pre_nll)

        x = x + self.ffn_drop(delta)
        return x, aux, skip_rate


class NexusExpanseByteCore(nn.Module):
    """Tokenless 4-pillar experimental cognitive kernel operating directly on raw bytes (0..255).

    Synergistic architecture:
      1. DirectMonarch sparse full-rank FFN backbone (~6.2x smaller than Dense at dim=128, 50x at dim=2048).
      2. Strictly causal pre-FFN uncertainty gating (~50%+ deep FFN skip).
      3. Decoupled EpisodicByteMemory (non-parametric multi-scale recall; interference possible).
      4. Surprise-gated HebbianFastWeightAdapter (zero-backprop test-time plasticity).
    """

    def __init__(
        self,
        n_layers: int = 4,
        dim: int = 128,
        n_heads: int = 4,
        ffn_mult: int = 4,
        target_skip_rate: float = 0.50,
        enable_skip: bool = True,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        memory_max_order: int = 8,
        fast_lr: float = 0.08,
        ffn_kind: str = "monarch",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.n_blocks = rank_safe_blocks(dim, dim * ffn_mult)

        self.emb = nn.Embedding(256, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                NexusLayer(
                    dim=dim,
                    n_heads=n_heads,
                    ffn_mult=ffn_mult,
                    n_blocks=self.n_blocks,
                    target_skip_rate=target_skip_rate,
                    enable_skip=enable_skip,
                    dropout=dropout,
                    ffn_kind=ffn_kind,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)

        # Decoupled memory & test-time Hebbian scratchpad
        self.episodic = EpisodicByteMemory(max_order=memory_max_order)
        self.fast_weights = HebbianFastWeightAdapter(dim=dim, vocab_size=256, lr=fast_lr)

    def new_kv_cache(self):
        """Create an opt-in independent CPU inference stream for frozen weights."""
        from nexus_expanse.kv_cache import CausalKVCache
        return CausalKVCache(self)

    def _backbone_inputs(self, tokens):
        # Fixed eval geometry prevents a length-dependent kernel choice at a hard gate.
        if not self.training and any(layer.ffn.enable_skip for layer in self.layers):
            if tokens.shape[1] > self.max_seq_len:
                raise ValueError("input exceeds position embedding length")
            tokens = F.pad(tokens, (0, self.max_seq_len - tokens.shape[1]))
        t = tokens.shape[1]
        pos = torch.arange(t, device=tokens.device).unsqueeze(0)
        mask = torch.triu(torch.ones((t, t), dtype=torch.bool, device=tokens.device), diagonal=1)
        x = self.drop(self.emb(tokens) + self.pos_emb(pos))

        return x, mask

    def forward_neural(
        self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[float]]:
        """Run the tokenless Monarch neural backbone; returns (logits, hidden, aux_loss, skip_rates)."""
        length = tokens.shape[1]
        x, mask = self._backbone_inputs(tokens)

        total_aux = torch.zeros((), device=tokens.device)
        skips: List[float] = []
        for layer in self.layers:
            x, aux, sk = layer(x, mask, self.out_norm, self.emb.weight, targets=targets, valid_length=length)
            total_aux = total_aux + aux
            skips.append(sk)

        h_out = self.out_norm(x)
        logits = F.linear(h_out, self.emb.weight)
        return logits[:, :length], h_out[:, :length], total_aux / float(len(self.layers)), skips

    @torch.inference_mode()
    def calibrate_gates(
        self, train_batches: List[Tuple[torch.Tensor, torch.Tensor]], max_batches: int = 16
    ) -> None:
        """Between-epoch train-only calibration of causal skip thresholds (zero val leakage)."""
        was_training = self.training
        self.eval()
        try:
            for layer_idx, layer in enumerate(self.layers):
                if not layer.ffn.enable_skip:
                    continue
                scores_list: List[torch.Tensor] = []
                for xb, _ in train_batches[:max_batches]:
                    length = xb.shape[1]
                    x, mask = self._backbone_inputs(xb)
                    for previous in self.layers[:layer_idx]:
                        x, _, _ = previous(x, mask, self.out_norm, self.emb.weight, valid_length=length)
                    x_pre = layer.attention_residual(x, mask)
                    scores = layer.ffn.gate_scores(layer.ffn_norm(x_pre))
                    scores_list.append(scores[:, :length].reshape(-1))
                all_s = torch.cat(scores_list, dim=0)
                q = torch.quantile(all_s, layer.ffn.target_skip_rate)
                layer.ffn.tau_eval.copy_(q)
        finally:
            self.train(was_training)

    def total_ffn_footprint(self) -> Tuple[int, int]:
        total_p, total_b = 0, 0
        for layer in self.layers:
            p, b = static_footprint_bytes(layer.ffn)
            total_p += p
            total_b += b
        return total_p, total_b

    @torch.inference_mode()
    def evaluate_streaming(
        self,
        val_bytes: bytes,
        seq_len: int = 64,
        batch_size: int = 32,
        use_episodic: bool = True,
        online_episodic_update: bool = False,
        online_hebbian_update: bool = False,
        mem_blend_cap: float = 0.72,
        use_fast_weights: Optional[bool] = None,
        reset_fast_state: bool = True,
        arbitrator: Optional[ReliabilityArbitrator] = None,
        online_arbitration_update: bool = False,
    ) -> Dict[str, object]:
        """Strictly causal FP64 evaluation over EVERY transition in val_bytes (len-1 bytes).

        When online_episodic_update / online_hebbian_update are enabled, byte y_t
        is ALWAYS predicted strictly BEFORE observing y_t, and only then written
        into episodic memory / Hebbian fast weights.
        """
        if len(val_bytes) < 2 or not 1 <= seq_len <= self.max_seq_len:
            raise ValueError("stream needs two bytes and valid sequence length")
        if not 0 <= mem_blend_cap <= 1:
            raise ValueError("invalid memory blend cap")
        if online_episodic_update and not use_episodic:
            raise ValueError("episodic update requires episodic prediction")
        if arbitrator is not None and not use_episodic:
            raise ValueError("arbitration requires episodic prediction")
        if online_arbitration_update and arbitrator is None:
            raise ValueError("arbitration update requires an arbitrator")
        fast_enabled = online_hebbian_update if use_fast_weights is None else use_fast_weights
        if online_hebbian_update and not fast_enabled:
            raise ValueError("Hebbian update requires fast-weight prediction")
        was_training = self.training
        self.eval()
        if reset_fast_state:
            self.fast_weights.reset_fast_state()

        val_tokens = torch.tensor(list(val_bytes), dtype=torch.long)
        n_total = len(val_tokens) - 1
        # Build non-overlapping sequential chunks covering all len(val_bytes) - 1 transitions
        chunks: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        pos = 0
        while pos < n_total:
            end = min(pos + seq_len, n_total)
            chunks.append((pos, val_tokens[pos:end].unsqueeze(0), val_tokens[pos + 1 : end + 1].unsqueeze(0)))
            pos = end

        total_nll_nats = 0.0
        count = 0
        memory_weight_sum = 0.0
        arbitration_start = arbitrator.updates if arbitrator is not None else 0
        skips = [0.0 for _ in self.layers]

        try:
            for start_idx, xb, yb in chunks:
                logits, h_out, _, chunk_skips = self.forward_neural(xb)
                t_len = yb.shape[1]
                for idx, sk in enumerate(chunk_skips):
                    skips[idx] += sk * t_len

                for j in range(t_len):
                    global_i = start_idx + j
                    target_b = int(yb[0, j].item())

                    # 1. Neural Monarch logits (+ optional test-time Hebbian fast-weight residual)
                    step_logits = logits[0, j].double()
                    if fast_enabled:
                        fast_context = val_bytes[max(0, global_i + 1 - getattr(self.fast_weights, 'context_order', 0)):global_i + 1]
                        step_logits = step_logits + self.fast_weights.forward_delta(h_out[0, j], context=fast_context).double()
                    p_neural = F.softmax(step_logits, dim=-1)

                    # 2. Confidence-weighted fusion with decoupled EpisodicByteMemory
                    if use_episodic:
                        ctx_start = max(0, global_i + 1 - self.episodic.max_order)
                        ctx = val_bytes[ctx_start : global_i + 1]
                        p_mem, conf = self.episodic.predict_probs(ctx)
                        w_mem = (arbitrator.weight(conf) if arbitrator is not None
                                 else min(mem_blend_cap, conf * mem_blend_cap))
                        memory_weight_sum += w_mem
                        p_final = (1.0 - w_mem) * p_neural + w_mem * p_mem
                    else:
                        p_final = p_neural

                    p_target = max(float(p_final[target_b].item()), 1e-12)
                    nll = -math.log(p_target)
                    total_nll_nats += nll
                    count += 1

                    # 3. Strictly AFTER scoring target_b, update online episodic & Hebbian state
                    if online_arbitration_update:
                        arbitrator.observe(conf, float(p_neural[target_b]), float(p_mem[target_b]))
                    if use_episodic and online_episodic_update:
                        ctx_start = max(0, global_i + 1 - self.episodic.max_order)
                        self.episodic.observe_transition(val_bytes[ctx_start : global_i + 1], target_b)
                    if online_hebbian_update:
                        self.fast_weights.hebbian_update(
                            h_out[0, j], p_neural, target_b, surprise_nll=nll, surprise_threshold=2.0, context=fast_context
                        )
        finally:
            self.train(was_training)

        return {
            "bpb": total_nll_nats / (count * math.log(2.0)),
            "bytes": count,
            "mean_skip": sum(skips) / (count * len(self.layers)),
            "per_layer_skip": [s / count for s in skips],
            "episodic_entries": self.episodic.memory_entry_count(),
            "hebbian_updates": self.fast_weights.updates_applied,
            "mean_memory_weight": memory_weight_sum / count,
            "arbitration_updates": arbitrator.updates - arbitration_start if arbitrator is not None else 0,
        }
