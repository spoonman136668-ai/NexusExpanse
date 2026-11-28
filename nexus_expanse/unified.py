"""Opt-in Nexus Expanse v1.0-dev raw-byte stream and relational memory."""
from __future__ import annotations

import io
import math
from collections import deque
from copy import deepcopy
from dataclasses import dataclass

import torch

from nexus_expanse.arbitration import ReliabilityArbitrator
from nexus_expanse.consolidation import ConsolidatingMemory
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.deltanet import DeltaNetMonarchByteCore
from nexus_expanse.hybrid_backbone import HybridDeltaAttentionCore
from nexus_expanse.hopfield_vsa import HopfieldVSAMemory
from nexus_expanse.patching import BytePatchReader, CompactBigram
from nexus_expanse.snapshot_quant import SNAPSHOT_QUANTS, pack_weights, unpack_weights
from nexus_expanse.substrate import DirectMonarchLinear
from nexus_expanse.sleep_consolidation import EPISODIC_BYTES, SleepHopfieldMemory, sleep


class _DeltaStream:
    def __init__(self, backbone):
        self.backbone = backbone
        self.reset()

    def reset(self):
        self.states = self.backbone.init_stream_state()

    def append(self, byte):
        logits, _, self.states = self.backbone.step_logits(byte, self.states)
        return logits

    @property
    def tensor_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.states)


def _stream_for(backbone, *, fast_decode=False):
    if isinstance(backbone, NexusExpanseByteCore):
        return backbone.new_kv_cache(batched=fast_decode)
    if isinstance(backbone, HybridDeltaAttentionCore):
        return backbone.new_stream()
    return _DeltaStream(backbone)


class _PrefillPatchReader(BytePatchReader):
    def prefill(self, raw):
        # Advance the same surprise and pending-byte state as forecast(),
        # without scoring intermediate bytes or replaying expired events.
        for observed in raw:
            wake = self.previous is None or (
                (self.counts[self.previous, observed] + 1.) / self.totals[self.previous]
                < self.minimum_probability)
            self.previous = observed
            if wake:
                if self.pending is not None:
                    self.events.append(self.pending)
                    self.pending = None
                self.events.append(observed)
                self.deep_calls += 1
            else:
                self.pending = observed
        logits = self.cache.rebuild(self.events)
        if wake:
            return logits.double().softmax(0)
        return torch.from_numpy((self.counts[observed].astype("float64") + 1.) / self.totals[observed])


@dataclass(frozen=True)
class ByteForecast:
    mixed: torch.Tensor
    patch: torch.Tensor
    episodic: torch.Tensor
    confidence: float


class NexusExpanseUnified:
    """Own an eval backbone and bounded memories for a single raw-byte stream.

    Byte forecasts update episodic counts and reliability only after the target
    byte arrives. Relation observations and corpus blocks have explicit entry
    points. Resetting the stream clears context while preserving learned memory.
    """

    MEMORY_BUDGET = 40_000
    # Eight dense FP32 layers at width 512 and 4096 learned positions fit.
    SNAPSHOT_LIMIT = 80_000_000
    LEGACY_SNAPSHOT_LIMIT = 32_000_000
    SNAPSHOT_VERSION = "nexus-unified-dev-v1"

    def __init__(self, backbone, counts, *, memory=None, sleep_budget=False, fast_decode=False):
        if not isinstance(counts, CompactBigram):
            raise ValueError("a compact fitted byte bigram is required")
        if not isinstance(backbone, (DeltaNetMonarchByteCore, HybridDeltaAttentionCore, NexusExpanseByteCore)):
            raise ValueError("unsupported reasoning backbone")
        if backbone.emb.weight.device.type != "cpu" or backbone.emb.weight.dtype != torch.float32:
            raise ValueError("unified DEV requires FP32 CPU weights")
        self.backbone = backbone.requires_grad_(False).eval()
        self.snapshot_quant = "fp32"
        self.max_seq_len = getattr(backbone, "max_seq_len", 256)
        self.sleep_budget = sleep_budget
        cap = EPISODIC_BYTES if sleep_budget else self.MEMORY_BUDGET
        self.episodic = ConsolidatingMemory(cap, "sparse_support", 8) if memory is None else memory
        if not isinstance(self.episodic, ConsolidatingMemory) or self.episodic.budget > cap:
            raise ValueError("unified DEV requires bounded consolidating memory")
        self.reasoning = (SleepHopfieldMemory if sleep_budget else HopfieldVSAMemory)(self.backbone)
        self.arbitrator = ReliabilityArbitrator()
        if isinstance(backbone, NexusExpanseByteCore):
            for layer in backbone.layers:
                if not isinstance(layer.ffn.fc1, (DirectMonarchLinear, torch.nn.Linear)):
                    raise ValueError("unsupported unified attention feed-forward layout")
                layer.ffn.enable_skip = True
                layer.ffn.tau_eval.fill_(-math.inf)
        self.stream = _stream_for(backbone, fast_decode=fast_decode)
        self.reader = _PrefillPatchReader(self, counts, kv_cache=self.stream)
        self._history = deque(maxlen=self.episodic.max_order)
        self._previous = None

    def reset_context(self):
        self.reader.reset_context()
        self._history.clear()
        self._previous = None

    @torch.inference_mode()
    def step(self, observed, *, adapt=True):
        if type(observed) is not int or not 0 <= observed <= 255:
            raise ValueError("expected a raw byte")
        if adapt and self._previous is not None:
            patch, episodic, confidence = self._previous
            self.arbitrator.observe(confidence, float(patch[observed]), float(episodic[observed]))
            self.episodic.observe_transition(bytes(self._history), observed)
        self._history.append(observed)
        patch = self.reader.forecast(observed)
        return self._forecast(patch)

    def _forecast(self, patch):
        patch = patch.cpu()
        episodic, confidence = self.episodic.predict_probs(bytes(self._history))
        weight = self.arbitrator.weight(confidence)
        self._previous = (patch, episodic, confidence)
        mixed = (1 - weight) * patch + weight * episodic
        return ByteForecast(mixed, patch.clone(), episodic.clone(), confidence)

    @torch.inference_mode()
    def prefill(self, raw, *, adapt=False):
        if not raw:
            raise ValueError("expected a nonempty raw-byte prompt")
        if not adapt and getattr(self.stream, "batched", False):
            self._history.extend(raw)
            return self._forecast(self.reader.prefill(raw))
        for byte in raw:
            forecast = self.step(byte, adapt=adapt)
        return forecast

    @torch.inference_mode()
    def forward_neural(self, tokens):
        return self.backbone.forward_neural(tokens)

    def predict_relation(self, prefix):
        return self.reasoning.predict(*self.reasoning.encode(prefix))

    def observe_relation(self, prefix, target):
        self.reasoning.observe(prefix, target)

    def consolidate_sleep(self, boundary_replay=()):
        """Apply opt-in, bounded offline representation consolidation."""
        if not self.sleep_budget:
            raise ValueError("initialize the unified runtime with sleep_budget=True")
        return sleep(self, boundary_replay)

    def _config(self):
        model = self.backbone
        layer = model.layers[0]
        fc1 = layer.ffn.fc1
        hidden = fc1.out_features if isinstance(fc1, torch.nn.Linear) else fc1.out_dim
        shape = dict(n_layers=len(model.layers), dim=model.dim, ffn_mult=hidden // model.dim)
        if isinstance(model, NexusExpanseByteCore):
            if isinstance(fc1, torch.nn.Linear):
                shape["ffn_kind"] = "dense"
            if model.readout is not None:
                shape["untied_readout"] = True
            return "attention", dict(shape, n_heads=layer.attn.num_heads, max_seq_len=model.max_seq_len)
        if isinstance(model, HybridDeltaAttentionCore):
            return "hybrid", dict(dim=model.dim, n_heads=model.n_heads, ffn_mult=model.ffn_mult,
                                  max_seq_len=model.max_seq_len, schedule=model.schedule)
        return "delta", dict(shape, n_heads=layer.n_heads)

    @classmethod
    def _snapshot_limit(cls, kind, config):
        if (kind == "attention" and 32 <= config["dim"] <= 512 and 2 <= config["n_layers"] <= 8
                and 8 <= config["max_seq_len"] <= 4096 and config["ffn_mult"] == 2):
            return cls.SNAPSHOT_LIMIT
        return cls.LEGACY_SNAPSHOT_LIMIT

    def snapshot(self, *, snapshot_quant="fp32"):
        """Serialize model, adaptive memories, and the in-flight stream cursor."""
        if snapshot_quant not in SNAPSHOT_QUANTS:
            raise ValueError("unsupported snapshot quantization")
        kind, config = self._config()
        payload = dict(
            version=self.SNAPSHOT_VERSION, kind=kind, config=config,
            backbone=self.backbone.state_dict(),
            compact_counts=self.reader.counts.snapshot(),
            episodic=self.episodic.snapshot(max_bytes=self.MEMORY_BUDGET),
            relation=self.reasoning.snapshot(),
            arbitration=(self.arbitrator.share, self.arbitrator.weight(0.0), self.arbitrator.updates),
            reader=(bytes(self.reader.events).hex(), self.reader.previous, self.reader.pending,
                    self.reader.deep_calls),
            history=bytes(self._history).hex(),
            preceding_forecast=self._previous,
            delta_state=self.stream.states if kind == "delta" else None,
        )
        if self.sleep_budget:
            payload["sleep_budget"] = True
        if snapshot_quant != "fp32":
            payload["snapshot_quant"] = snapshot_quant
            scales = pack_weights(self.backbone, payload["backbone"], snapshot_quant)
            if snapshot_quant == "int8":
                payload["int8_scales"] = scales
        output = io.BytesIO()
        torch.save(payload, output)
        if output.tell() > self._snapshot_limit(kind, config):
            raise ValueError("whole-system checkpoint exceeds DEV limit")
        return output.getvalue()

    @classmethod
    def restore(cls, payload, *, fast_decode=False, decode_device="cpu"):
        if len(payload) > cls.SNAPSHOT_LIMIT:
            raise ValueError("whole-system checkpoint exceeds DEV limit")
        saved = torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")
        if saved["version"] != cls.SNAPSHOT_VERSION:
            raise ValueError("unsupported unified checkpoint version")
        snapshot_quant = saved.get("snapshot_quant", "fp32")
        if snapshot_quant not in SNAPSHOT_QUANTS:
            raise ValueError("unsupported snapshot quantization")
        if len(payload) > cls.LEGACY_SNAPSHOT_LIMIT and len(payload) > cls._snapshot_limit(saved["kind"], saved["config"]):
            raise ValueError("whole-system checkpoint exceeds DEV limit for its geometry")
        with torch.random.fork_rng(devices=[]):
            if saved["kind"] == "attention":
                backbone = NexusExpanseByteCore(**saved["config"], dropout=0.0, enable_skip=True)
            elif saved["kind"] == "hybrid":
                backbone = HybridDeltaAttentionCore(**saved["config"])
            elif saved["kind"] == "delta":
                backbone = DeltaNetMonarchByteCore(**saved["config"])
            else:
                raise ValueError("unsupported checkpoint backbone")
        if snapshot_quant != "fp32":
            unpack_weights(backbone, saved["backbone"], snapshot_quant, saved.get("int8_scales"))
        backbone.load_state_dict(saved["backbone"])
        counts = CompactBigram.restore(saved["compact_counts"])
        memory = ConsolidatingMemory.restore(saved["episodic"], max_bytes=cls.MEMORY_BUDGET)
        restored = cls(backbone, counts, memory=memory, sleep_budget=saved.get("sleep_budget", False),
                       fast_decode=fast_decode)
        restored.snapshot_quant = snapshot_quant
        restored.reasoning.restore(saved["relation"])

        share, weight, updates = saved["arbitration"]
        if not math.isfinite(weight) or not 0 <= weight <= 1 or type(updates) is not int or updates < 0:
            raise ValueError("invalid arbitration state")
        restored.arbitrator = ReliabilityArbitrator(share=share)
        restored.arbitrator._weights[0] = weight
        restored.arbitrator.updates = updates

        cursor, previous, pending, calls = saved["reader"]
        events = bytes.fromhex(cursor)
        history = bytes.fromhex(saved["history"])
        forecast = saved["preceding_forecast"]
        if (len(events) > restored.max_seq_len
                or len(history) > restored.episodic.max_order
                or previous != (history[-1] if history else None)
                or (pending is not None and pending != previous)
                or type(calls) is not int or calls < 0
                or bool(history) != (forecast is not None)):
            raise ValueError("invalid stream cursor")
        if forecast is not None:
            patch, memory, confidence = forecast
            if (not 0 <= confidence <= 1
                    or any(t.shape != (256,) or t.dtype != torch.float64
                           or not torch.isfinite(t).all() or (t < 0).any()
                           or not torch.isclose(t.sum(), t.new_tensor(1.0)) for t in (patch, memory))):
                raise ValueError("invalid preceding forecast")
        restored.reader.events.extend(events)
        restored.reader.previous, restored.reader.pending = previous, pending
        restored.reader.deep_calls = calls
        restored._history.extend(history)
        restored._previous = forecast

        device = torch.device(decode_device)
        if device.type != "cpu":
            # Keep the snapshot and relational runtime on their portable CPU representation.
            restored.stream = _stream_for(deepcopy(backbone).to(device), fast_decode=fast_decode)
            restored.reader.cache = restored.stream
        if saved["kind"] in ("attention", "hybrid"):
            if events and getattr(restored.stream, "batched", False):
                restored.stream.rebuild(events)
            else:
                for byte in events:
                    restored.stream.append(byte)
        else:
            states = saved["delta_state"]
            if any(new.shape != old.shape or new.dtype != old.dtype or not torch.isfinite(new).all()
                   for new, old in zip(states, restored.stream.states, strict=True)):
                raise ValueError("invalid recurrent state")
            restored.stream.states = tuple(state.to(device) for state in states)
        return restored
