"""Frozen raw-byte skill routing using only evidence available before each forecast."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import math
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

from nexus_expanse.checkpoint_transport import CONFIG, CorrectedLinear, ImportedSkill
from nexus_expanse.episodic_born_skill import EpisodicAssistedBornSkill


BANK_V1 = "nexus-multi-skill-bank-v1"
BANK_V2 = "nexus-multi-skill-bank-v2"


def _validate_probability(value, name):
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


def _sticky_transition(reference, k, switch_prob):
    _validate_probability(switch_prob, "switch_prob")
    if k == 1:
        return reference.new_zeros(1, 1)
    return reference.new_full((k, k), switch_prob / (k - 1)).fill_diagonal_(1 - switch_prob).log()


def _hmm_predict(log_posterior, transition):
    return (log_posterior.unsqueeze(-1) + transition).logsumexp(-2).log_softmax(-1)


def _hmm_update(log_prior, log_emission, transition):
    posterior = (log_prior + log_emission).log_softmax(-1)
    return _hmm_predict(posterior, transition)


class MultiSkillBank(nn.Module):
    """Route between frozen imported skills in insertion order, with the base first.

    For raw [batch, time], row t forecasts raw[:, t] from raw[:, :t].
    With include_next=True there is an additional forecast after the last byte.
    A uniform byte prior requires no expert calls at position zero.
    """

    def __init__(
        self, base: ImportedSkill, skills: Mapping[str, ImportedSkill | EpisodicAssistedBornSkill], *,
        base_name: str = "base_python", gamma: float = 0.85,
        surprise_threshold_bits: float = 3.5, probe_interval: int = 16,
        resident_names: tuple[str, ...] | list[str] | None = None,
    ):
        super().__init__()
        if base_name in skills:
            raise ValueError("the base skill name must be unique")
        modules = {base_name: base, **skills}
        if type(base) is not ImportedSkill or any(
            type(skill) not in (ImportedSkill, EpisodicAssistedBornSkill) for skill in skills.values()
        ):
            raise ValueError("a bank requires an ImportedSkill base and recognized receivers")
        neural = [skill.neural if isinstance(skill, EpisodicAssistedBornSkill) else skill
                  for skill in modules.values()]
        if len({id(skill) for skill in neural}) != len(modules):
            raise ValueError("each bank entry requires an independent receiver")
        if resident_names is None:
            resident_names = tuple(modules)
        if (not isinstance(resident_names, (tuple, list))
                or any(not isinstance(name, str) or name not in modules for name in resident_names)
                or len(set(resident_names)) != len(resident_names) or base_name not in resident_names):
            raise ValueError("resident names must uniquely identify existing skills including the base")
        self._resident_names = tuple(resident_names)
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

    @property
    def resident_names(self):
        """The original group retained across births."""
        return self._resident_names

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


    @torch.inference_mode()
    def forward_hmm_bma(self, raw, switch_prob=0.05, *, include_next=False):
        """Mix expert probabilities using the predictive Sticky-HMM prior."""
        components = self.component_log_probs(raw, include_next=include_next)
        batch, _, k, _ = components.shape
        transition = _sticky_transition(components, k, switch_prob)
        prior = components.new_full((batch, k), -math.log(k))
        return self._mix_hmm(raw, components, prior, transition)

    @torch.inference_mode()
    def forward_hmm_bma_hierarchical(
        self, raw, switch_prob=0.05, born_prior_mass=0.02, born_switch_share=0.10, *, include_next=False,
    ):
        """Give the born group a fixed initial and resident-switch mass."""
        names = self.skill_names
        residents = len(self.resident_names)
        born = len(names) - residents
        if born == 0:
            return self.forward_hmm_bma(raw, switch_prob, include_next=include_next)

        _validate_probability(switch_prob, "switch_prob")
        _validate_probability(born_prior_mass, "born_prior_mass")
        _validate_probability(born_switch_share, "born_switch_share")
        components = self.component_log_probs(raw, include_next=include_next)
        batch, _, k, _ = components.shape
        is_resident = torch.tensor([name in self.resident_names for name in names], device=components.device)

        prior_probs = components.new_full((k,), born_prior_mass / born)
        prior_probs[is_resident] = (1 - born_prior_mass) / residents
        prior = prior_probs.log().expand(batch, k)

        # A lone resident has no peer to receive its resident-switch share.
        share = born_switch_share if residents > 1 else 1.0
        rates = components.new_full((k,), switch_prob * share / born)
        if residents > 1:
            rates[is_resident] = switch_prob * (1 - share) / (residents - 1)
        transition = components.new_full((k, k), switch_prob / (k - 1))
        transition[is_resident] = rates
        transition.fill_diagonal_(1 - switch_prob).log_()
        return self._mix_hmm(raw, components, prior, transition)

    @staticmethod
    def _mix_hmm(raw, components, prior, transition):
        _, steps, k, _ = components.shape
        history = [prior]
        for t in range(1, steps):
            observed = raw[:, t - 1].long().view(-1, 1, 1).expand(-1, k, 1)
            emission = components[:, t - 1].gather(-1, observed).squeeze(-1)
            prior = _hmm_update(prior, emission, transition)
            history.append(prior)
        log_weights = torch.stack(history, dim=1)
        return (components + log_weights.unsqueeze(-1)).logsumexp(-2), log_weights.exp()

    @torch.inference_mode()
    def forward_hmm_sparse(
        self, raw, switch_prob=0.05, surprise_threshold_bits=3.0, probe_interval=16, *,
        include_next=False,
    ):
        """Forecast with one expert and return predictive priors and actual call rate.

        A probe filters up to four observed bytes from the stored prior at that
        window's start. Only a probe scores cross-skill emissions. Other steps
        apply the transition alone. Replayed state affects future forecasts only.
        """
        threshold, interval = self._probe_options(surprise_threshold_bits, probe_interval)
        raw, steps = self._prepare(raw, include_next)
        batch = raw.size(0)
        experts = tuple(self.skills.values())
        k = len(experts)
        forecasts = [self._prior(raw)]
        prior = forecasts[0].new_full((batch, k), -math.log(k))
        transition = _sticky_transition(prior, k, switch_prob)
        priors = deque([prior], maxlen=min(4, self.context - 1))
        weights = [prior.exp()]
        rows_all = torch.arange(batch, device=raw.device)
        calls = 0

        for t in range(1, steps):
            observed = raw[:, t - 1:t]
            surprise = -forecasts[-1].gather(-1, observed).squeeze(-1) / math.log(2)
            probe = (surprise > threshold) | (t % interval == 0)
            n_probe = int(probe.sum().item())
            prefix = self._prefix(raw, t)
            prior = _hmm_predict(priors[-1], transition)
            active = prior.argmax(-1)
            options = forecasts[-1].new_empty(batch, k, 256)
            window = len(priors)
            available = min(window, prefix.size(1) - 1) if n_probe else 0
            emissions = prior.new_full((n_probe, window, k), -math.log(256))
            recent = raw[probe, t - available:t].unsqueeze(-1)

            for index, model in enumerate(experts):
                rows = probe | (active == index)
                count = int(rows.sum().item())
                if count == 0:
                    continue
                predictions = model(prefix[rows])[:, -(available + 1):].float().log_softmax(-1)
                options[rows, index] = predictions[:, -1]
                if available:
                    earlier = predictions[probe[rows], :-1]
                    emissions[:, -available:, index] = earlier.gather(-1, recent).squeeze(-1)
                calls += count

            if n_probe:
                replay = priors[0][probe]
                for offset in range(window):
                    replay = _hmm_update(replay, emissions[:, offset], transition)
                    if offset + 1 < window:
                        priors[offset + 1][probe] = replay
                prior[probe] = replay
                active = prior.argmax(-1)

            forecasts.append(options[rows_all, active])
            priors.append(prior)
            weights.append(prior.exp())

        return torch.stack(forecasts, dim=1), torch.stack(weights, dim=1), calls / (batch * steps)


def save_bank(bank: MultiSkillBank, path):
    """Atomically save tensor-only receiver data and every routing default."""
    entries = {}
    schema = BANK_V1
    for name, skill in bank.skills.items():
        episodic = isinstance(skill, EpisodicAssistedBornSkill)
        model = skill.neural if episodic else skill
        methods = []
        for layer in model.core.layers:
            for linear in (layer.ffn.fc1, layer.ffn.fc2):
                methods.append(linear.method if isinstance(linear, CorrectedLinear) else "monarch")
        entry = dict(width=model.core.dim, methods=methods, state=model.state_dict())
        if episodic:
            entry["episodic"] = dict(keys=skill.keys, successors=skill.successors)
            schema = BANK_V2
        entries[name] = entry
    package = dict(
        schema=schema, base_name=bank.base_name, entries=entries, resident_names=list(bank.resident_names),
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
    if not isinstance(package, dict) or package.get("schema") not in (BANK_V1, BANK_V2):
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
            if "episodic" in entry:
                episode = entry["episodic"]
                if (package["schema"] != BANK_V2 or name == base_name
                        or not isinstance(episode, dict) or set(episode) != {"keys", "successors"}):
                    raise ValueError("invalid episodic receiver in bank snapshot")
                model = EpisodicAssistedBornSkill(model, episode["keys"], episode["successors"])
            skills[name] = model
    return MultiSkillBank(
        skills.pop(base_name), skills, base_name=base_name,
        gamma=package.get("gamma", float("nan")),
        surprise_threshold_bits=package.get("surprise_threshold_bits", float("nan")),
        probe_interval=package.get("probe_interval"), resident_names=package.get("resident_names"),
    )
