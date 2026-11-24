"""Opt-in bounded, versioned episodic memory; no neural checkpoint or pickle loading."""
from __future__ import annotations

import json
import math
from pathlib import Path

from nexus_expanse.memory import EpisodicByteMemory


class BoundedEpisodicByteMemory(EpisodicByteMemory):
    """Bound distinct suffix contexts, evicting least recently observed contexts.

    Reads never change recency. Unigram statistics survive eviction. The limit
    bounds context count, not Python allocator bytes; one observe call can
    temporarily allocate up to max_order additional contexts before eviction.
    This is an explicit tradeoff, not a zero-forgetting memory.
    """

    def __init__(self, max_contexts: int, max_order: int = 8, smoothing: float = 0.05):
        if type(max_contexts) is not int or max_contexts < 1:
            raise ValueError("positive integer context budget required")
        if type(max_order) is not int or not 1 <= max_order <= 64:
            raise ValueError("max_order must be in 1..64")
        if not math.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("positive finite smoothing required")
        super().__init__(max_order, smoothing)
        self.max_contexts = max_contexts

    def observe_transition(self, context: bytes, next_byte: int, weight: float = 1.0):
        if not isinstance(context, bytes) or type(next_byte) is not int or not 0 <= next_byte <= 255:
            raise ValueError("invalid byte transition")
        if not math.isfinite(weight) or weight <= 0 or not math.isfinite(self._unigram_total + weight):
            raise ValueError("invalid transition weight")
        super().observe_transition(context, next_byte, weight)
        for order in range(1, min(len(context), self.max_order) + 1):
            key = context[-order:]
            self._table[key] = self._table.pop(key)
        while len(self._table) > self.max_contexts:
            oldest = next(iter(self._table))
            del self._table[oldest]
            del self._totals[oldest]

    def snapshot(self, *, max_bytes: int) -> bytes:
        """Canonical JSON, preserving recency and all floating counts exactly."""
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("positive serialized-byte budget required")
        state = {"schema": "nexus-bounded-episodic-v1", "max_contexts": self.max_contexts,
                 "max_order": self.max_order, "smoothing": self.smoothing,
                 "observations": self.num_observations, "unigram": self._unigram,
                 "unigram_total": self._unigram_total,
                 "contexts": [[key.hex(), self._totals[key], sorted(bucket.items())]
                              for key, bucket in self._table.items()]}
        payload = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
        if len(payload) > max_bytes:
            raise ValueError("serialized memory exceeds budget")
        return payload

    @classmethod
    def restore(cls, payload: bytes, *, max_bytes: int, max_contexts: int):
        """Validate untrusted state and enforce externally supplied limits before use."""
        if type(max_bytes) is not int or max_bytes < 1 or len(payload) > max_bytes:
            raise ValueError("serialized memory exceeds budget")
        if type(max_contexts) is not int or max_contexts < 1:
            raise ValueError("invalid external context budget")
        def positive(n):
            return type(n) in (int, float) and math.isfinite(n) and n > 0
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate field")
                result[key] = value
            return result
        try:
            s = json.loads(payload, object_pairs_hook=unique_pairs)
            fields = {"schema", "max_contexts", "max_order", "smoothing", "observations",
                      "unigram", "unigram_total", "contexts"}
            if not isinstance(s, dict) or set(s) != fields or s["schema"] != "nexus-bounded-episodic-v1":
                raise ValueError("unsupported snapshot schema")
            cap = s["max_contexts"]
            if type(cap) is not int or not 1 <= cap <= max_contexts:
                raise ValueError("snapshot context budget exceeds allowed limit")
            mem = cls(cap, s["max_order"], s["smoothing"])
            if type(s["observations"]) is not int or s["observations"] < 0:
                raise ValueError("invalid observation count")
            uni = s["unigram"]
            total = s["unigram_total"]
            if (not isinstance(uni, list) or len(uni) != 256 or not all(positive(n) and n >= 1 for n in uni)
                    or not positive(total) or not math.isclose(sum(uni), total, rel_tol=1e-9)):
                raise ValueError("invalid unigram statistics")
            rows = s["contexts"]
            if not isinstance(rows, list) or len(rows) > cap:
                raise ValueError("invalid context count")
            for key_hex, count, entries in rows:
                key = bytes.fromhex(key_hex)
                if not 1 <= len(key) <= mem.max_order or key in mem._table or not positive(count):
                    raise ValueError("invalid or duplicate context")
                if not isinstance(entries, list) or not 1 <= len(entries) <= 256:
                    raise ValueError("invalid context entries")
                bucket = {}
                for byte, value in entries:
                    if type(byte) is not int or not 0 <= byte <= 255 or byte in bucket or not positive(value):
                        raise ValueError("invalid or duplicate byte count")
                    bucket[byte] = value
                if not math.isclose(sum(bucket.values()), count, rel_tol=1e-9) or count > total:
                    raise ValueError("inconsistent context total")
                mem._table[key] = bucket
                mem._totals[key] = count
            mem._unigram = uni
            mem._unigram_total = total
            mem.num_observations = s["observations"]
            return mem
        except (TypeError, KeyError, OverflowError, UnicodeError, RecursionError) as exc:
            raise ValueError("malformed memory snapshot") from exc

    def save(self, path: Path, *, max_bytes: int):
        payload = self.snapshot(max_bytes=max_bytes)
        with Path(path).open("xb") as f:
            f.write(payload)

    @classmethod
    def load(cls, path: Path, *, max_bytes: int, max_contexts: int):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("invalid serialized-byte budget")
        with Path(path).open("rb") as f:
            payload = f.read(max_bytes + 1)
        return cls.restore(payload, max_bytes=max_bytes, max_contexts=max_contexts)
