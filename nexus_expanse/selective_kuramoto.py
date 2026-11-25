"""Gate 11: opt-in selective S^1 oscillator and contrastive JEPA."""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F

from nexus_expanse.kuramoto import KuramotoOscillatorLayer, KuramotoMonarchByteCore
from nexus_expanse.jepa import latent
from nexus_expanse.jepa_domain import momentum
from nexus_expanse.substrate import DirectMonarchLinear


class SelectiveKuramotoLayer(KuramotoOscillatorLayer):
    """A per-oscillator input-dependent phase update; no additional recurrent state."""

    def __init__(self, dim, **kwargs):
        super().__init__(dim, **kwargs)
        self.gate_proj = nn.Linear(dim, self.n_osc)
        # Neutral half-open initialization, independent of the original weights.
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def _osc_step(self, h_t, z_prev):
        b = h_t.shape[0]
        h64 = h_t.double()
        forcing = F.linear(h64, self.forcing_proj.weight.double(),
                           self.forcing_proj.bias.double()).reshape(b, self.n_osc, 2)
        coupling = F.linear(z_prev.reshape(b, self.dim),
                            self.coupling_proj.weight.double()).reshape(b, self.n_osc, 2)
        gate = F.linear(h64, self.gate_proj.weight.double(),
                        self.gate_proj.bias.double()).sigmoid().unsqueeze(-1)
        drive = forcing + coupling
        tangent = drive - (drive * z_prev).sum(-1, keepdim=True) * z_prev
        z_raw = z_prev + (self.dt * gate) * tangent
        return z_raw / (z_raw.norm(dim=-1, keepdim=True) + 1e-12)


class SelectiveKuramotoByteCore(KuramotoMonarchByteCore):
    """Raw-byte core preserving Gate 10 shared initialization and streaming API."""

    def __init__(self, n_layers=2, dim=64, ffn_mult=2, target_skip_rate=.50,
                 enable_skip=False, dropout=0., max_seq_len=256):
        if enable_skip:
            raise ValueError("Gate 11 requires always-on FFNs")
        super().__init__(n_layers=n_layers, dim=dim, ffn_mult=ffn_mult,
                         target_skip_rate=target_skip_rate, enable_skip=False,
                         dropout=dropout, max_seq_len=max_seq_len)
        for i, original in enumerate(self.layers):
            layer = SelectiveKuramotoLayer(dim=dim, ffn_mult=ffn_mult,
                                           n_blocks=self.n_blocks,
                                           target_skip_rate=target_skip_rate,
                                           enable_skip=False, dropout=dropout)
            missing, unexpected = layer.load_state_dict(original.state_dict(), strict=False)
            if set(missing) != {"gate_proj.weight", "gate_proj.bias"} or unexpected:
                raise RuntimeError("shared initialization mismatch")
            self.layers[i] = layer

    def forward_neural(self, tokens):
        # Training retains Gate 10's batched FFN path. Eval calls the exact
        # single-step geometry, including embedding, layer norm and tied readout.
        if self.training:
            return super().forward_neural(tokens)
        states = self.init_stream_state(tokens.shape[0], tokens.device)
        logits, hidden = [], []
        for t in range(tokens.shape[1]):
            x = self.emb(tokens[:, t:t+1])
            new = []
            for layer, old in zip(self.layers, states):
                x, z, _ = layer.forward_step(x, old)
                new.append(z)
            states = new
            h = self.out_norm(x)
            hidden.append(h)
            logits.append(F.linear(h.double(), self.emb.weight.double()).to(h.dtype))
        return torch.cat(logits, 1), torch.cat(hidden, 1), [0.] * len(self.layers)


def contrastive_loss(pred, target, temperature=.1):
    """Each matched future is positive; every other in-batch future is negative."""
    if pred.ndim != 2 or pred.shape != target.shape or len(pred) < 2:
        raise ValueError("expected matched latent batches with at least two rows")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scores = F.normalize(pred, dim=-1) @ F.normalize(target.detach(), dim=-1).T
    return F.cross_entropy(scores / temperature, torch.arange(len(pred), device=pred.device))


class InfoNCEKuramotoJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = SelectiveKuramotoByteCore()
        self.target = copy.deepcopy(self.context).requires_grad_(False).eval()
        self.predictor = nn.Sequential(DirectMonarchLinear(64,128,8), nn.SiLU(),
                                       DirectMonarchLinear(128,64,8))

    def train(self, mode=True):
        super().train(mode)
        self.target.eval()
        return self

    def loss(self, ctx, clen, fut, flen):
        pred = self.predictor(latent(self.context, ctx, clen))
        with torch.no_grad():
            target = latent(self.target, fut, flen)
        return contrastive_loss(pred, target)

    @torch.no_grad()
    def update_target(self, step):
        m = momentum(step)
        for p, q in zip(self.context.parameters(), self.target.parameters()):
            q.mul_(m).add_(p, alpha=1-m)
