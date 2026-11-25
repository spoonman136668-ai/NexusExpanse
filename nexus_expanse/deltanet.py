"""Opt-in raw-byte DeltaNet with fixed-size per-head associative matrices."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from nexus_expanse.substrate import CausalGatedMonarchFFN, rank_safe_blocks


class GatedDeltaNetLayer(nn.Module):
    """Store each head as [value, key], so a query is a right multiplication."""

    def __init__(self, dim, n_heads=4, ffn_mult=2, dropout=0.0):
        super().__init__()
        if dim < 1 or n_heads < 1 or dim % n_heads:
            raise ValueError("dim must be positive and divisible by n_heads")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.input_norm = nn.LayerNorm(dim)
        self.input_proj = nn.Linear(dim, 3 * dim + 2 * n_heads)
        self.output_proj = nn.Linear(dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = CausalGatedMonarchFFN(
            dim, dim * ffn_mult, rank_safe_blocks(dim, dim * ffn_mult),
            enable_skip=False,
        )
        self.dropout = nn.Dropout(dropout)

    def init_state(self, batch_size, device):
        return torch.zeros(
            batch_size, self.n_heads, self.head_dim, self.head_dim,
            device=device, dtype=torch.float64,
        )

    def _project(self, x):
        packed = self.input_proj(self.input_norm(x)).double()
        q, k, v, alpha, beta = packed.split(
            (self.dim, self.dim, self.dim, self.n_heads, self.n_heads), dim=-1,
        )
        shape = (*x.shape[:-1], self.n_heads, self.head_dim)
        return (
            q.reshape(shape), F.normalize(k.reshape(shape), dim=-1),
            v.reshape(shape), alpha.sigmoid(), beta.sigmoid(),
        )

    @staticmethod
    def update(state, key, value, alpha, beta):
        error = value - (state @ key.unsqueeze(-1)).squeeze(-1)
        write = (beta.unsqueeze(-1) * error).unsqueeze(-1) * key.unsqueeze(-2)
        return alpha[..., None, None] * state + write

    def _finish(self, x, read):
        x = x + self.dropout(self.output_proj(read.reshape(x.shape).to(x.dtype)))
        hidden = self.ffn_norm(x)
        delta = self.ffn.fc2(F.silu(self.ffn.fc1(hidden)))
        return x + self.dropout(delta)

    def forward_step(self, x, state):
        q, k, v, alpha, beta = self._project(x)
        state = self.update(state, k, v, alpha, beta)
        read = (state @ q.unsqueeze(-1)).squeeze(-1)
        return self._finish(x, read), state

    @staticmethod
    def _training_reads(q, k, v, alpha, beta):
        # Solve for past delta writes in parallel. Entry [t, i] decays
        # write i through the state just before t (strictly lower triangle).
        log_alpha = alpha.clamp_min(torch.finfo(alpha.dtype).tiny).log()
        cumulative = log_alpha.cumsum(dim=-1)
        before = F.pad(cumulative[..., :-1], (1, 0))
        before_decay = (before.unsqueeze(-1) - cumulative.unsqueeze(-2)).clamp_max(0).exp()
        key_similarity = k @ k.transpose(-2, -1)
        lower = (beta.unsqueeze(-1) * before_decay * key_similarity).tril(-1)
        writes = torch.linalg.solve_triangular(
            lower, beta.unsqueeze(-1) * v, upper=False, unitriangular=True,
        )
        decay = (cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)).clamp_max(0).exp()
        return ((q @ k.transpose(-2, -1)) * decay).tril() @ writes

    def forward(self, x):
        """Differentiable parallel formulation for whole training sequences."""
        q, k, v, alpha, beta = self._project(x)
        read = self._training_reads(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            alpha.transpose(1, 2), beta.transpose(1, 2),
        )
        return self._finish(x, read.transpose(1, 2))


class DeltaNetMonarchByteCore(nn.Module):
    """Always-on DirectMonarch with single-byte and exact batch evaluation APIs."""

    def __init__(self, n_layers=2, dim=64, n_heads=4, ffn_mult=2, dropout=0.0):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        self.dim = dim
        self.emb = nn.Embedding(256, dim)
        self.layers = nn.ModuleList(
            GatedDeltaNetLayer(dim, n_heads, ffn_mult, dropout) for _ in range(n_layers)
        )
        self.out_norm = nn.LayerNorm(dim)

    def init_stream_state(self, batch_size=1, device=None):
        device = self.emb.weight.device if device is None else device
        return tuple(layer.init_state(batch_size, device) for layer in self.layers)

    def _readout(self, x):
        h = self.out_norm(x)
        return F.linear(h, self.emb.weight), h

    def _step(self, tokens, states):
        x = self.emb(tokens)
        next_states = []
        for layer, state in zip(self.layers, states, strict=True):
            x, state = layer.forward_step(x, state)
            next_states.append(state)
        logits, hidden = self._readout(x)
        return logits, hidden, tuple(next_states)

    def forward_neural(self, tokens):
        if tokens.ndim != 2 or not tokens.shape[1]:
            raise ValueError("expected nonempty [batch, time] raw-byte tokens")
        if self.training:
            x = self.emb(tokens)
            for layer in self.layers:
                x = layer(x)
            logits, hidden = self._readout(x)
        else:
            states = self.init_stream_state(len(tokens), tokens.device)
            outputs, features = [], []
            for column in tokens.unbind(dim=1):
                logits, hidden, states = self._step(column, states)
                outputs.append(logits)
                features.append(hidden)
            logits, hidden = torch.stack(outputs, 1), torch.stack(features, 1)
        return logits, hidden, [0.0] * len(self.layers)

    @torch.inference_mode()
    def step_logits(self, byte, states):
        if self.training:
            raise ValueError("streaming requires an eval model")
        if type(byte) is not int or not 0 <= byte <= 255:
            raise ValueError("expected a raw byte")
        tokens = torch.tensor([byte], device=self.emb.weight.device)
        logits, hidden, states = self._step(tokens, states)
        return logits[0], hidden[0], states

    @torch.inference_mode()
    def step_streaming(self, byte, states):
        logits, _, states = self.step_logits(byte, states)
        return logits.double().softmax(0), states
