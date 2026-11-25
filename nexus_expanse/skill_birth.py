"""Causal novelty screening and isolated raw-byte skill adaptation."""
from __future__ import annotations

import copy
import math
from threading import RLock

import torch
from torch import nn
from torch.nn import functional as F

from nexus_expanse.episodic_born_skill import EpisodicAssistedBornSkill
from nexus_expanse.jepa import state_hash
from nexus_expanse.multi_skill_router import MultiSkillBank, load_bank, save_bank


NOVELTY_WINDOW = 64
NOVELTY_THRESHOLD_BPB = 8.0
ADAPT_STEPS = 32
PHASE4_BIRTH = "phase4_baseline_birth"
EPISODIC_BIRTH = "episodic_assisted_birth"
CONSOLIDATED_BIRTH = "consolidated_neural_birth"
BIRTH_STEPS = {PHASE4_BIRTH: ADAPT_STEPS, EPISODIC_BIRTH: ADAPT_STEPS, CONSOLIDATED_BIRTH: 192}


def adapt_skill(skill, raw, steps):
    """Fit one receiver on consecutive observed bytes with bounded context."""
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if raw.size(1) < 2:
        raise ValueError("skill adaptation needs at least two observed bytes")
    with torch.inference_mode(False), torch.enable_grad():
        observed = raw.clone()
        context = skill.core.max_seq_len
        starts = range(0, observed.size(1) - 1, context)
        skill.requires_grad_(True).train()
        optimizer = torch.optim.AdamW(skill.parameters(), lr=0.004, weight_decay=0.01)
        for step in range(steps):
            start = starts[step % len(starts)]
            block = observed[:, start:start + context + 1]
            optimizer.zero_grad(set_to_none=True)
            logits = skill(block[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, 256), block[:, 1:].reshape(-1))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite skill adaptation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(skill.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        skill.zero_grad(set_to_none=True)
        skill.requires_grad_(False).eval()


class OpenWorldSkillBank(nn.Module):
    """Wrap a fixed bank with an explicit, causal observation and birth path."""

    def __init__(self, bank: MultiSkillBank):
        super().__init__()
        self.bank = bank
        self._birth_lock = RLock()
        self.eval()

    @property
    def resident_names(self):
        return self.bank.resident_names

    def forward(self, raw, *, include_next=False):
        """Forecast through the currently registered bank without updating it."""
        return self.bank.forward_hmm_bma(raw, include_next=include_next)

    @torch.inference_mode()
    def detect_novelty(self, raw, min_bpb_threshold=NOVELTY_THRESHOLD_BPB, *, window=NOVELTY_WINDOW):
        """Return one flag per stream using only the last completed observation window."""
        if not math.isfinite(min_bpb_threshold) or min_bpb_threshold <= 0:
            raise ValueError("novelty threshold must be finite and positive")
        if type(window) is not int or window < 1:
            raise ValueError("novelty window must be a positive integer")
        raw, _ = self.bank._prepare(raw, include_next=True)
        if raw.size(1) <= window:
            return torch.zeros(raw.size(0), dtype=torch.bool, device=raw.device)
        context = raw[:, -(self.bank.context + window):]
        components = self.bank.component_log_probs(context)[:, -window:]
        batch, _, k, _ = components.shape
        targets = context[:, -window:].view(batch, window, 1, 1).expand(-1, -1, k, -1)
        expert_bpb = -components.gather(-1, targets).squeeze(-1).double().mean(1) / math.log(2)
        return expert_bpb.min(-1).values > min_bpb_threshold

    def observe(self, name, raw_stream, *, min_bpb_threshold=NOVELTY_THRESHOLD_BPB,
                window=NOVELTY_WINDOW, steps=None, mode=PHASE4_BIRTH):
        """Register a skill only when one ordered, already observed stream is novel."""
        if raw_stream.ndim != 2 or raw_stream.size(0) != 1:
            raise ValueError("online birth requires one observed byte stream")
        with self._birth_lock:
            if name in self.bank.skills or not bool(self.detect_novelty(raw_stream, min_bpb_threshold, window=window)):
                return None
            return self.spawn_and_adapt_skill(name, raw_stream, steps=steps, mode=mode)

    def spawn_and_adapt_skill(self, name, raw_stream, steps=None, *, mode=PHASE4_BIRTH):
        """Consolidate a clone and register it after validating every resident state."""
        if mode not in BIRTH_STEPS:
            raise ValueError("unknown birth consolidation mode")
        steps = BIRTH_STEPS[mode] if steps is None else steps
        with self._birth_lock, torch.inference_mode(False):
            if not isinstance(name, str) or not name or "." in name or hasattr(self.bank.skills, name):
                raise ValueError("expected a unique, valid skill name")
            raw, _ = self.bank._prepare(raw_stream, include_next=True)
            before = {key: state_hash(skill) for key, skill in self.bank.skills.items()}
            candidate = copy.deepcopy(self.bank.base)
            adapt_skill(candidate, raw, steps)
            if mode == EPISODIC_BIRTH:
                candidate = EpisodicAssistedBornSkill.from_encounter(candidate, raw)
            after = {key: state_hash(skill) for key, skill in self.bank.skills.items()}
            if before != after:
                raise RuntimeError("resident skill state changed during birth")
            self.bank.skills[name] = candidate
            return candidate

    def save_bank(self, path):
        with self._birth_lock:
            save_bank(self.bank, path)

    @classmethod
    def load_bank(cls, path):
        return cls(load_bank(path))
