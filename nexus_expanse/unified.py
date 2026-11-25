"""Opt-in Nexus Expanse v1.0-dev raw-byte stream and relational memory."""
from __future__ import annotations

import io
import math
from collections import deque
from dataclasses import dataclass

import torch

from nexus_expanse.arbitration import ReliabilityArbitrator
from nexus_expanse.consolidation import ConsolidatingMemory
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.deltanet import DeltaNetMonarchByteCore
from nexus_expanse.hybrid_backbone import HybridDeltaAttentionCore
from nexus_expanse.hopfield_vsa import HopfieldVSAMemory
from nexus_expanse.patching import BytePatchReader, CompactBigram
from nexus_expanse.substrate import DirectMonarchLinear


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
    SNAPSHOT_LIMIT = 32_000_000
    SNAPSHOT_VERSION = "nexus-unified-dev-v1"

    def __init__(self, backbone, counts, *, memory=None):
        if not isinstance(counts, CompactBigram):
            raise ValueError("a compact fitted byte bigram is required")
        if not isinstance(backbone, (DeltaNetMonarchByteCore, HybridDeltaAttentionCore, NexusExpanseByteCore)):
            raise ValueError("unsupported reasoning backbone")
        if backbone.emb.weight.device.type != "cpu" or backbone.emb.weight.dtype != torch.float32:
            raise ValueError("unified DEV requires FP32 CPU weights")
        self.backbone = backbone.requires_grad_(False).eval()
        self.max_seq_len = getattr(backbone, "max_seq_len", 256)
        self.episodic = ConsolidatingMemory(self.MEMORY_BUDGET, "sparse_support", 8) if memory is None else memory
        if not isinstance(self.episodic, ConsolidatingMemory) or self.episodic.budget > self.MEMORY_BUDGET:
            raise ValueError("unified DEV requires bounded consolidating memory")
        self.reasoning = HopfieldVSAMemory(self.backbone)
        self.arbitrator = ReliabilityArbitrator()
        if isinstance(backbone, NexusExpanseByteCore):
            for layer in backbone.layers:
                if not isinstance(layer.ffn.fc1, DirectMonarchLinear):
                    raise ValueError("unified attention requires DirectMonarch")
                layer.ffn.enable_skip = True
                layer.ffn.tau_eval.fill_(-math.inf)
            self.stream = backbone.new_kv_cache()
        elif isinstance(backbone, HybridDeltaAttentionCore):
            self.stream = backbone.new_stream()
        else:
            self.stream = _DeltaStream(backbone)
        self.reader = BytePatchReader(self, counts, kv_cache=self.stream)
        self._history = deque(maxlen=self.episodic.max_order)
        self._previous = None

    def reset_context(self):
        self.reader.reset_context()
        self._history.clear()
        self._previous = None

    @torch.inference_mode()
    def step(self, observed):
        if type(observed) is not int or not 0 <= observed <= 255:
            raise ValueError("expected a raw byte")
        if self._previous is not None:
            patch, episodic, confidence = self._previous
            self.arbitrator.observe(confidence, float(patch[observed]), float(episodic[observed]))
            self.episodic.observe_transition(bytes(self._history), observed)
        self._history.append(observed)
        patch = self.reader.forecast(observed)
        episodic, confidence = self.episodic.predict_probs(bytes(self._history))
        weight = self.arbitrator.weight(confidence)
        self._previous = (patch, episodic, confidence)
        mixed = (1 - weight) * patch + weight * episodic
        return ByteForecast(mixed, patch.clone(), episodic.clone(), confidence)

    @torch.inference_mode()
    def forward_neural(self, tokens):
        return self.backbone.forward_neural(tokens)

    def predict_relation(self, prefix):
        return self.reasoning.predict(*self.reasoning.encode(prefix))

    def observe_relation(self, prefix, target):
        self.reasoning.observe(prefix, target)

    def _config(self):
        model = self.backbone
        layer = model.layers[0]
        shape = dict(n_layers=len(model.layers), dim=model.dim,
                     ffn_mult=layer.ffn.fc1.out_dim // model.dim)
        if isinstance(model, NexusExpanseByteCore):
            return "attention", dict(shape, n_heads=layer.attn.num_heads, max_seq_len=model.max_seq_len)
        if isinstance(model, HybridDeltaAttentionCore):
            return "hybrid", dict(dim=model.dim, n_heads=model.n_heads, ffn_mult=model.ffn_mult,
                                  max_seq_len=model.max_seq_len, schedule=model.schedule)
        return "delta", dict(shape, n_heads=layer.n_heads)

    def snapshot(self):
        """Serialize model, adaptive memories, and the in-flight stream cursor."""
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
        output = io.BytesIO()
        torch.save(payload, output)
        if output.tell() > self.SNAPSHOT_LIMIT:
            raise ValueError("whole-system checkpoint exceeds DEV limit")
        return output.getvalue()

    @classmethod
    def restore(cls, payload):
        if len(payload) > cls.SNAPSHOT_LIMIT:
            raise ValueError("whole-system checkpoint exceeds DEV limit")
        saved = torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")
        if saved["version"] != cls.SNAPSHOT_VERSION:
            raise ValueError("unsupported unified checkpoint version")
        with torch.random.fork_rng(devices=[]):
            if saved["kind"] == "attention":
                backbone = NexusExpanseByteCore(**saved["config"], dropout=0.0, enable_skip=True)
            elif saved["kind"] == "hybrid":
                backbone = HybridDeltaAttentionCore(**saved["config"])
            elif saved["kind"] == "delta":
                backbone = DeltaNetMonarchByteCore(**saved["config"])
            else:
                raise ValueError("unsupported checkpoint backbone")
        backbone.load_state_dict(saved["backbone"])
        counts = CompactBigram.restore(saved["compact_counts"])
        memory = ConsolidatingMemory.restore(saved["episodic"], max_bytes=cls.MEMORY_BUDGET)
        restored = cls(backbone, counts, memory=memory)
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

        if saved["kind"] in ("attention", "hybrid"):
            for byte in events:
                restored.stream.append(byte)
        else:
            states = saved["delta_state"]
            if any(new.shape != old.shape or new.dtype != old.dtype or not torch.isfinite(new).all()
                   for new, old in zip(states, restored.stream.states, strict=True)):
                raise ValueError("invalid recurrent state")
            restored.stream.states = states
        return restored
