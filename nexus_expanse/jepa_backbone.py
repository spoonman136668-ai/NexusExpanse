"""InfoNCE and per-domain VICReg for byte encoders with causal hidden states."""
import copy

import torch
from torch import nn

from nexus_expanse.jepa_domain import domain_vicreg, momentum
from nexus_expanse.selective_kuramoto import contrastive_loss
from nexus_expanse.substrate import DirectMonarchLinear, rank_safe_blocks


def last_hidden(encoder, tokens, lengths):
    hidden = encoder.forward_neural(tokens)[1]
    return hidden[torch.arange(len(tokens), device=tokens.device), lengths - 1]


class BackboneJEPA(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.context = encoder
        self.target = copy.deepcopy(encoder).requires_grad_(False).eval()
        dim = encoder.dim
        blocks = rank_safe_blocks(dim, 2 * dim)
        self.predictor = nn.Sequential(
            DirectMonarchLinear(dim, 2 * dim, blocks), nn.SiLU(),
            DirectMonarchLinear(2 * dim, dim, blocks),
        )

    def train(self, mode=True):
        super().train(mode)
        self.target.eval()
        return self

    def loss(self, ctx, clen, fut, flen, domains):
        context = last_hidden(self.context, ctx, clen)
        prediction = self.predictor(context)
        with torch.no_grad():
            target = last_hidden(self.target, fut, flen)
        cv, cc = domain_vicreg(context, domains)
        pv, pc = domain_vicreg(prediction, domains)
        return contrastive_loss(prediction, target) + 25 * (cv + pv) + cc + pc

    @torch.no_grad()
    def update_target(self, step, total):
        weight = momentum(step, total)
        for source, ema in zip(self.context.parameters(), self.target.parameters(), strict=True):
            ema.mul_(weight).add_(source, alpha=1.0 - weight)
