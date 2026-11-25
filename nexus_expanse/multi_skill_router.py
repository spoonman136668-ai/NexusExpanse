"""Frozen raw-byte skill routing using only evidence available before each forecast."""
from __future__ import annotations

from collections.abc import Mapping
import math
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

from nexus_expanse.checkpoint_transport import CONFIG, CorrectedLinear, ImportedSkill


class MultiSkillBank(nn.Module):
    """Route between frozen imported skills in insertion order, with the base first.

    For raw [batch, time], row t forecasts raw[:, t] from raw[:, :t].
    With include_next=True there is an additional forecast after the last byte.
    A uniform byte prior requires no expert calls at position zero.
    """

    def __init__(
        self, base: ImportedSkill, skills: Mapping[str, ImportedSkill], *,
        base_name: str = "base_python", gamma: float = 0.85,
        surprise_threshold_bits: float = 3.5, probe_interval: int = 16,
    ):
        super().__init__()
        if base_name in skills:
            raise ValueError("the base skill name must be unique")
        modules = {base_name: base, **skills}
        if any(type(skill) is not ImportedSkill for skill in modules.values()):
            raise ValueError("a bank requires checkpoint-transport ImportedSkill receivers")
        if len({id(skill) for skill in modules.values()}) != len(modules):
            raise ValueError("each bank entry requires an independent receiver")
        self.skills = nn.ModuleDict(modules)
        self.base_name = base_name
        self.context = min(skill.core.max_seq_len for skill in self.skills.values())
        if self.context < 2:
            raise ValueError("a bank requires a context of at least two bytes")
        self.gamma = self._discount(gamma)
        self.surprise_threshold_bits, self.probe_interval = self._probe_options(
            surprise_threshold_bits, probe_interval
        )
        self.requires_grad_(False).eval()

    @property
    def base(self):
        return self.skills[self.base_name]

    @property
    def skill_names(self):
        return tuple(self.skills)

    @staticmethod
    def _discount(gamma):
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        return float(gamma)

    @staticmethod
    def _probe_options(threshold, interval):
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("surprise threshold must be finite and nonnegative")
        if type(interval) is not int or interval < 1:
            raise ValueError("probe interval must be a positive integer")
        return float(threshold), interval

    def _prepare(self, raw, include_next):
        if self.training or any(skill.training for skill in self.skills.values()):
            raise ValueError("routing requires a frozen eval bank")
        if (raw.ndim != 2 or raw.shape[0] == 0
                or raw.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
            raise ValueError("expected a [batch, time] tensor of integer raw bytes")
        if raw.numel() and (raw.min().item() < 0 or raw.max().item() > 255):
            raise ValueError("expected raw bytes in [0, 255]")
        steps = raw.shape[1] + int(include_next)
        if steps == 0:
            raise ValueError("provide at least one byte or request the initial forecast")
        return raw.long().contiguous(), steps

    def _prior(self, raw):
        return torch.full((raw.size(0), 256), -math.log(256), device=raw.device, dtype=torch.float32)

    def _prefix(self, raw, t):
        return raw[:, max(0, t - self.context):t].contiguous()

    @torch.inference_mode()
    def component_log_probs(self, raw, *, selected=None, include_next=False):
        """Return forecasts [B,T,K,256], or [B,T,256] for supplied expert indices.

        A supplied selection is for controlled comparisons with fixed or oracle
        routing. Each expert runs only on its selected streams in that case.
        """
        raw, steps = self._prepare(raw, include_next)
        prior = self._prior(raw)
        experts = tuple(self.skills.values())
        if selected is None:
            forecasts = [prior.unsqueeze(1).expand(-1, len(experts), -1)]
            for t in range(1, steps):
                prefix = self._prefix(raw, t)
                forecasts.append(torch.stack([
                    model(prefix)[:, -1].float().log_softmax(-1) for model in experts
                ], dim=1))
            return torch.stack(forecasts, dim=1)

        if (selected.shape != (raw.size(0), steps) or selected.dtype != torch.long
                or selected.device != raw.device or selected.min().item() < 0
                or selected.max().item() >= len(experts)):
            raise ValueError("expected valid expert indices for every forecast")
        forecasts = [prior]
        for t in range(1, steps):
            prefix = self._prefix(raw, t)
            forecast = torch.empty_like(prior)
            for k, model in enumerate(experts):
                rows = selected[:, t] == k
                if rows.any():
                    forecast[rows] = model(prefix[rows])[:, -1].float().log_softmax(-1)
            forecasts.append(forecast)
        return torch.stack(forecasts, dim=1)

    def _bayes(self, raw, gamma, include_next):
        components = self.component_log_probs(raw, include_next=include_next)
        scores = components.new_zeros(components.size(0), components.size(2))
        history = [scores]
        for t in range(1, components.size(1)):
            observed = raw[:, t - 1].long().view(-1, 1, 1).expand(-1, scores.size(1), 1)
            scores = gamma * scores + components[:, t - 1].gather(-1, observed).squeeze(-1)
            history.append(scores)
        return components, torch.stack(history, dim=1)

    @torch.inference_mode()
    def forward_bayes_bma(self, raw, gamma=None, *, include_next=False):
        gamma = self.gamma if gamma is None else self._discount(gamma)
        components, scores = self._bayes(raw, gamma, include_next)
        log_weights = scores.log_softmax(-1)
        return (components + log_weights.unsqueeze(-1)).logsumexp(-2), log_weights.exp()

    @torch.inference_mode()
    def forward_bayes_argmax(self, raw, gamma=None, *, include_next=False):
        gamma = self.gamma if gamma is None else self._discount(gamma)
        components, scores = self._bayes(raw, gamma, include_next)
        active = scores.argmax(-1)
        index = active.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 256)
        return components.gather(-2, index).squeeze(-2), active

    @torch.inference_mode()
    def forward_surprise_sparse(
        self, raw, gamma=None, surprise_threshold_bits=None, probe_interval=None, *,
        include_next=False,
    ):
        """Return forecasts, top-1 indices and actual expert calls per forecast.

        A probe uses the previous observed byte to compare all experts on
        one common target. The same forward pass also provides the next-byte
        forecasts. Non-probe steps score just the active expert's surprise.
        """
        gamma = self.gamma if gamma is None else self._discount(gamma)
        threshold, interval = self._probe_options(
            self.surprise_threshold_bits if surprise_threshold_bits is None else surprise_threshold_bits,
            self.probe_interval if probe_interval is None else probe_interval,
        )
        raw, steps = self._prepare(raw, include_next)
        batch = raw.size(0)
        experts = tuple(self.skills.values())
        active = raw.new_zeros(batch)
        scores = torch.zeros(batch, len(experts), device=raw.device)
        forecasts = [self._prior(raw)]
        selections = [active]
        calls = 0

        for t in range(1, steps):
            observed = raw[:, t - 1:t]
            surprise = -forecasts[-1].gather(-1, observed).squeeze(-1) / math.log(2)
            probe = (surprise > threshold) | (t % interval == 0)
            prefix = self._prefix(raw, t)
            scores = scores * gamma
            options = forecasts[-1].new_empty(batch, len(experts), 256)
            for k, model in enumerate(experts):
                rows = probe | (active == k)
                count = int(rows.sum().item())
                if count:
                    logits = model(prefix[rows])[:, -2:].float().log_softmax(-1)
                    options[rows, k] = logits[:, -1]
                    if t >= 2:
                        likelihood = logits[:, -2].gather(-1, observed[rows]).squeeze(-1)
                        scores[rows, k] += torch.where(probe[rows], likelihood, 0.0)
                    calls += count
            active = torch.where(probe, scores.argmax(-1), active)
            forecasts.append(options[torch.arange(batch, device=raw.device), active])
            selections.append(active)

        return torch.stack(forecasts, dim=1), torch.stack(selections, dim=1), calls / (batch * steps)


def save_bank(bank: MultiSkillBank, path):
    """Atomically save tensor-only receiver data and every routing default."""
    entries = {}
    for name, skill in bank.skills.items():
        methods = []
        for layer in skill.core.layers:
            for linear in (layer.ffn.fc1, layer.ffn.fc2):
                methods.append(linear.method if isinstance(linear, CorrectedLinear) else "monarch")
        entries[name] = dict(width=skill.core.dim, methods=methods, state=skill.state_dict())
    package = dict(
        schema="nexus-multi-skill-bank-v1", base_name=bank.base_name, entries=entries,
        gamma=bank.gamma, surprise_threshold_bits=bank.surprise_threshold_bits,
        probe_interval=bank.probe_interval,
    )
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
        partial = Path(out.name)
        try:
            torch.save(package, out)
            out.flush()
            os.fsync(out.fileno())
            out.close()
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)


def load_bank(path):
    """Rebuild only recognized receiver classes from a weights-only snapshot."""
    package = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(package, dict) or package.get("schema") != "nexus-multi-skill-bank-v1":
        raise ValueError("unsupported multi-skill bank snapshot")
    entries = package.get("entries")
    base_name = package.get("base_name")
    if not isinstance(entries, dict) or not isinstance(base_name, str) or base_name not in entries:
        raise ValueError("snapshot requires a named base skill")
    skills = {}
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        for name, entry in entries.items():
            if (not isinstance(entry, dict) or type(entry.get("width")) is not int
                    or not isinstance(entry.get("methods"), list)
                    or len(entry["methods"]) != CONFIG["n_layers"] * 2
                    or not isinstance(entry.get("state"), Mapping)):
                raise ValueError("invalid skill configuration in bank snapshot")
            emb = entry["state"].get("core.emb.weight")
            if not isinstance(emb, torch.Tensor) or emb.shape != (256, entry["width"]):
                raise ValueError("snapshot width disagrees with the receiver weights")
            model = ImportedSkill(entry["width"])
            methods = iter(entry["methods"])
            for layer in model.core.layers:
                for slot in ("fc1", "fc2"):
                    base = getattr(layer.ffn, slot)
                    method = next(methods)
                    if method != "monarch":
                        weight = torch.zeros(base.out_dim, base.in_dim)
                        setattr(layer.ffn, slot, CorrectedLinear(base, weight, method))
            model.load_state_dict(entry["state"], strict=True, assign=True)
            skills[name] = model
    return MultiSkillBank(
        skills.pop(base_name), skills, base_name=base_name,
        gamma=package.get("gamma", float("nan")),
        surprise_threshold_bits=package.get("surprise_threshold_bits", float("nan")),
        probe_interval=package.get("probe_interval"),
    )
