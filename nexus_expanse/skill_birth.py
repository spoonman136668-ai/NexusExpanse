"""Causal novelty screening and isolated raw-byte skill adaptation."""
from __future__ import annotations

import copy
from dataclasses import dataclass
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


@dataclass(frozen=True)
class TailSurpriseCalibration:
    """Per-skill validation limits for the upper observed-surprise tail."""

    skill_names: tuple[str, ...]
    limits_bpb: tuple[float, ...]
    quantile: float
    window: int


def _routed_window(bank, raw, window):
    raw, _ = bank._prepare(raw, include_next=True)
    if raw.size(1) <= window:
        return raw, None, None
    logp, weights = bank.forward_hmm_bma_hierarchical(raw)
    nll = -logp[:, -window:].gather(-1, raw[:, -window:].unsqueeze(-1)).squeeze(-1).double()
    active = weights[:, -window:].argmax(-1)
    dominant = F.one_hot(active, num_classes=len(bank.skills)).sum(1).argmax(-1)
    return raw, nll.mean(1) / math.log(2), dominant


@dataclass(frozen=True)
class PerSkillNoveltyDetector:
    """Frozen validation-window BPB limits keyed to the dominant routed skill."""

    skill_names: tuple[str, ...]
    max_val_bpb: tuple[float, ...]
    mean_val_bpb: tuple[float, ...]
    std_val_bpb: tuple[float, ...]
    window: int
    min_margin_bpb: float
    std_multiplier: float

    @classmethod
    @torch.inference_mode()
    def calibrate(cls, bank, validation, *, window=NOVELTY_WINDOW, min_margin_bpb=0.35, std_multiplier=3.0):
        if type(window) is not int or window < 1:
            raise ValueError("novelty window must be a positive integer")
        if any(not math.isfinite(value) or value < 0 for value in (min_margin_bpb, std_multiplier)):
            raise ValueError("novelty margins must be finite and nonnegative")
        names = bank.skill_names
        if set(validation) != set(names):
            raise ValueError("validation must cover each skill exactly once")
        maxima, means, spreads = [], [], []
        for name in names:
            _, bpb, _ = _routed_window(bank, validation[name], window)
            if bpb is None or not bool(bpb.isfinite().all()):
                raise ValueError(f"validation for {name} requires finite BPB from a complete window")
            maxima.append(bpb.max().item())
            means.append(bpb.mean().item())
            spreads.append(bpb.std(correction=0).item())
        return cls(names, tuple(maxima), tuple(means), tuple(spreads), window, min_margin_bpb, std_multiplier)

    @property
    def ceilings_bpb(self):
        return tuple(limit + max(self.min_margin_bpb, self.std_multiplier * spread)
                     for limit, spread in zip(self.max_val_bpb, self.std_val_bpb, strict=True))

    @torch.inference_mode()
    def assess(self, bank, raw):
        """Return flags and the observed-window BPB, dominant skill index and ceiling."""
        if bank.skill_names != self.skill_names:
            raise ValueError("per-skill calibration belongs to a different skill roster")
        raw, bpb, dominant = _routed_window(bank, raw, self.window)
        if bpb is None:
            return raw.new_zeros(raw.size(0), dtype=torch.bool), None, None, None
        ceiling = bpb.new_tensor(self.ceilings_bpb)[dominant]
        return bpb > ceiling, bpb, dominant, ceiling


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

    def _observed_nll(self, raw, window, selected=None):
        if type(window) is not int or window < 1:
            raise ValueError("novelty window must be a positive integer")
        raw, _ = self.bank._prepare(raw, include_next=True)
        if raw.size(1) <= window:
            return raw, None
        context = raw[:, -(self.bank.context + window):]
        if selected is not None:
            selected = context.new_full(context.shape, selected)
        components = self.bank.component_log_probs(context, selected=selected)[:, -window:]
        if selected is not None:
            components = components.unsqueeze(-2)
        batch, _, k, _ = components.shape
        targets = context[:, -window:].view(batch, window, 1, 1).expand(-1, -1, k, -1)
        return raw, -components.gather(-1, targets).squeeze(-1).double()

    @torch.inference_mode()
    def calibrate_tail_surprise(self, validation, *, quantile=0.90, margin_bpb=0.5, window=NOVELTY_WINDOW):
        """Freeze an upper-tail limit per skill from its own ID validation streams."""
        if not math.isfinite(quantile) or not 0 < quantile <= 1:
            raise ValueError("tail quantile must be in (0, 1]")
        if not math.isfinite(margin_bpb) or margin_bpb < 0:
            raise ValueError("tail margin must be finite and nonnegative")
        names = self.bank.skill_names
        if set(validation) != set(names):
            raise ValueError("validation must cover each skill exactly once")
        limits = []
        for index, name in enumerate(names):
            _, nll = self._observed_nll(validation[name], window, selected=index)
            if nll is None:
                raise ValueError(f"validation for {name} needs more than one window of bytes")
            tail = nll.kthvalue(math.ceil(quantile * window), dim=1).values
            limit = tail.max().item() / math.log(2) + margin_bpb
            if not math.isfinite(limit):
                raise ValueError("tail limits must be finite")
            limits.append(limit)
        return TailSurpriseCalibration(names, tuple(limits), quantile, window)

    @torch.inference_mode()
    def detect_novelty(self, raw, min_bpb_threshold=NOVELTY_THRESHOLD_BPB, *, window=NOVELTY_WINDOW, tail=None, per_skill=None):
        """Return one flag per stream using only the last completed observation window."""
        if per_skill is not None:
            if tail is not None:
                raise ValueError("choose one novelty calibration")
            return per_skill.assess(self.bank, raw)[0]
        if tail is not None:
            if tail.skill_names != self.bank.skill_names:
                raise ValueError("tail calibration belongs to a different skill roster")
            raw, nll = self._observed_nll(raw, tail.window)
            if nll is None:
                return torch.zeros(raw.size(0), dtype=torch.bool, device=raw.device)
            measured = nll.kthvalue(math.ceil(tail.quantile * tail.window), dim=1).values / math.log(2)
            return (measured > measured.new_tensor(tail.limits_bpb)).all(-1)
        if not math.isfinite(min_bpb_threshold) or min_bpb_threshold <= 0:
            raise ValueError("novelty threshold must be finite and positive")
        raw, nll = self._observed_nll(raw, window)
        if nll is None:
            return torch.zeros(raw.size(0), dtype=torch.bool, device=raw.device)
        return (nll.mean(1) / math.log(2)).min(-1).values > min_bpb_threshold

    def observe(self, name, raw_stream, *, min_bpb_threshold=NOVELTY_THRESHOLD_BPB,
                window=NOVELTY_WINDOW, steps=None, mode=PHASE4_BIRTH, tail=None, per_skill=None):
        """Register a skill only when one ordered, already observed stream is novel."""
        if raw_stream.ndim != 2 or raw_stream.size(0) != 1:
            raise ValueError("online birth requires one observed byte stream")
        with self._birth_lock:
            if name in self.bank.skills or not bool(
                self.detect_novelty(raw_stream, min_bpb_threshold, window=window, tail=tail, per_skill=per_skill)
            ):
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
