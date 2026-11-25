"""Opt-in raw-byte hybrid with a recurrent state and KV only where required."""
from collections import deque

import torch
from torch import nn
from torch.nn import functional as F

from nexus_expanse.core import NexusLayer
from nexus_expanse.deltanet import GatedDeltaNetLayer
from nexus_expanse.substrate import rank_safe_blocks


def _causal_mha(layer, hidden, keys, values, position):
    """One-query attention over the written prefix, using fixed per-step geometry."""
    attn = layer.attn
    width = hidden.shape[-1] // attn.num_heads
    packed = F.linear(layer.attn_norm(hidden), attn.in_proj_weight, attn.in_proj_bias)
    q, k, v = packed.view(len(hidden), 3, attn.num_heads, width).unbind(1)
    keys[:, :, position] = k
    values[:, :, position] = v
    weights = ((q.unsqueeze(-2) * width ** -.5) @ keys[:, :, :position + 1].transpose(-2, -1)).softmax(-1)
    read = (weights @ values[:, :, :position + 1]).reshape_as(hidden)
    hidden = hidden + attn.out_proj(read)
    return hidden + layer.ffn.fc2(F.silu(layer.ffn.fc1(layer.ffn_norm(hidden))))


class HybridDeltaAttentionCore(nn.Module):
    """Always-on DirectMonarch with exact shared batch/stream eval steps."""

    def __init__(self, dim=64, n_heads=4, ffn_mult=2, max_seq_len=256,
                 schedule=("delta", "attention")):
        super().__init__()
        schedule = tuple(schedule)
        if set(schedule) != {"delta", "attention"}:
            raise ValueError("schedule must contain delta and attention layers only")
        if dim < 1 or n_heads < 1 or dim % n_heads or max_seq_len < 1:
            raise ValueError("invalid head or window geometry")
        self.dim = dim
        self.n_heads = n_heads
        self.ffn_mult = ffn_mult
        self.max_seq_len = max_seq_len
        self.schedule = schedule
        blocks = rank_safe_blocks(dim, dim * ffn_mult)
        self.emb = nn.Embedding(256, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.layers = nn.ModuleList(
            GatedDeltaNetLayer(dim, n_heads, ffn_mult) if tag == "delta"
            else NexusLayer(dim, n_heads, ffn_mult, blocks, .5, False, 0.0)
            for tag in schedule
        )
        self.out_norm = nn.LayerNorm(dim)

    def _state(self, batch):
        shape = (batch, self.n_heads, self.max_seq_len, self.dim // self.n_heads)
        return [layer.init_state(batch, self.emb.weight.device) if tag == "delta"
                else (self.emb.weight.new_zeros(shape), self.emb.weight.new_zeros(shape))
                for tag, layer in zip(self.schedule, self.layers, strict=True)]

    def _inference_step(self, tokens, states, position):
        x = self.emb(tokens) + self.pos_emb.weight[position]
        for i, (tag, layer) in enumerate(zip(self.schedule, self.layers, strict=True)):
            if tag == "delta":
                x, states[i] = layer.forward_step(x, states[i])
            else:
                x = _causal_mha(layer, x, *states[i], position)
        hidden = self.out_norm(x)
        return F.linear(hidden, self.emb.weight), hidden

    @torch.inference_mode()
    def _evaluate(self, tokens):
        # Each independently streamed row has the same single-query geometry.
        rows = []
        for row in tokens:
            states = self._state(1)
            rows.append([self._inference_step(byte[None], states, i)
                         for i, byte in enumerate(row)])
        logits = torch.stack([torch.cat([part[0] for part in row]) for row in rows])
        hidden = torch.stack([torch.cat([part[1] for part in row]) for row in rows])
        return logits, hidden

    def forward_neural(self, tokens):
        if tokens.ndim != 2 or not len(tokens) or not 1 <= tokens.shape[1] <= self.max_seq_len:
            raise ValueError("expected nonempty [batch, time] raw bytes within the window")
        if not self.training:
            logits, hidden = self._evaluate(tokens)
        else:
            t = tokens.shape[1]
            positions = torch.arange(t, device=tokens.device)
            x = self.emb(tokens) + self.pos_emb(positions)
            mask = torch.ones((t, t), dtype=torch.bool, device=tokens.device).triu(1)
            for tag, layer in zip(self.schedule, self.layers, strict=True):
                if tag == "delta":
                    x = layer(x)
                else:
                    x = layer(x, mask, self.out_norm, self.emb.weight, valid_length=t)[0]
            hidden = self.out_norm(x)
            logits = F.linear(hidden, self.emb.weight)
        return logits, hidden, [0.0] * len(self.layers)

    def new_stream(self):
        if self.training:
            raise ValueError("streaming requires an eval model")
        return HybridByteStream(self)


class HybridByteStream:
    def __init__(self, backbone):
        self.backbone = backbone
        self.history = deque(maxlen=backbone.max_seq_len)
        self.reset()

    def reset(self):
        self.history.clear()
        self.states = self.backbone._state(1)

    @property
    def tensor_bytes(self):
        return sum(t.nbytes for entry in self.states for t in (entry if isinstance(entry, tuple) else (entry,)))

    def _step(self, byte):
        position = len(self.history)
        tokens = torch.tensor([byte], device=self.backbone.emb.weight.device)
        logits, hidden = self.backbone._inference_step(tokens, self.states, position)
        self.history.append(byte)
        return logits[0], hidden[0]

    @torch.inference_mode()
    def step(self, byte):
        if type(byte) is not int or not 0 <= byte <= 255:
            raise ValueError("expected a raw byte")
        if len(self.history) == self.history.maxlen:
            retained = list(self.history)[1:]
            self.reset()
            for previous in retained:
                self._step(previous)
        return self._step(byte)

    def append(self, byte):
        return self.step(byte)[0]
