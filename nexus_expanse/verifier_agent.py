"""Bounded synthesis over opaque raw-byte operators and checked outcomes."""
from dataclasses import dataclass
import struct

import numpy as np

from nexus_expanse.consolidation import ConsolidatingMemory
from nexus_expanse.hopfield_vsa import HopfieldVSAMemory


@dataclass(frozen=True)
class VerifierFeedback:
    program: bytes
    target: bytes
    states: tuple[bytes, ...]

    def __post_init__(self):
        if (not isinstance(self.program, bytes) or not 1 <= len(self.program) <= 3
                or not isinstance(self.target, bytes) or not self.target
                or len(self.states) != len(self.program) + 1
                or any(not isinstance(x, bytes) or len(x) != len(self.target) for x in self.states)):
            raise ValueError("invalid bounded verifier trace")

    @property
    def passed(self):
        return self.states[-1] == self.target

    @property
    def residual(self):
        return bytes(a ^ b for a, b in zip(self.states[-1], self.target, strict=True))

    @property
    def surprise(self):
        return sum(b.bit_count() for b in self.residual) / (8 * len(self.target))


class _SynthesisVSA(HopfieldVSAMemory):
    SLOTS = 64

    def observe_edge(self, symbol, operator, outcome):
        self._record((symbol, operator))
        self.outcomes.add(outcome)
        if self._write(symbol, operator, outcome, 1):
            self.roles.add(operator)


class VerifierGatedNexusAgent:
    """Compose remembered primitive transitions with a checked episodic shortcut."""

    BUDGET = 40_000
    EPISODIC_BUDGET = 18_000
    HEADER = struct.Struct("<4sBH")

    def __init__(self, *, gated=True, structured=False):
        self.gated = gated
        self._orders = [bytearray(256) for _ in range(3)] if structured else None
        self.episodic = ConsolidatingMemory(self.EPISODIC_BUDGET, "sparse_support", 5)
        self.relations = _SynthesisVSA(None)

    @property
    def structured(self):
        return self._orders is not None

    @staticmethod
    def _key(symbol, program):
        return b"\x00" + program + bytes((symbol,))

    def _transition(self, symbol, operator):
        episode = self.episodic._table.get(self._key(symbol, bytes((operator,))))
        recalled = self.relations._retrieve(symbol, operator)
        if recalled is None:
            return max(episode, key=episode.get) if episode else None
        distribution = recalled[0].copy()
        if episode:
            total = sum(episode.values())
            for result, count in episode.items():
                distribution[result] += count / total
        return int(np.argmax(distribution))

    @staticmethod
    def _order(row, order):
        return row[::-1] if order == 1 else row[1:] + row[:1] if order == 2 else row

    def propose(self, source, target, programs, tried):
        """Rank bounded pipelines by agreement with the raw goal."""
        edges = {}

        def predict(symbol, program):
            if len(program) > 1:
                shortcut = self.episodic._table.get(self._key(symbol, program))
                if shortcut:
                    return max(shortcut, key=shortcut.get)
            for operator in program:
                key = (symbol, operator)
                if key not in edges:
                    edges[key] = self._transition(symbol, operator)
                symbol = edges[key]
                if symbol is None:
                    break
            return symbol

        def score(program):
            if self._orders is None:
                values = [predict(symbol, program) for symbol in source]
            else:
                values = list(source)
                for op in program:
                    order = max(range(3), key=lambda i: self._orders[i][op])
                    values = self._order(values, order)
                    values = [predict(symbol, bytes((op,))) if symbol is not None else None
                              for symbol in values]
            hits = sum(x == y for x, y in zip(values, target, strict=True))
            misses = sum(x is not None for x in values) - hits
            return hits - misses, hits

        return max((p for p in programs if p not in tried), key=score)

    def observe(self, feedback):
        if not isinstance(feedback, VerifierFeedback):
            raise ValueError("admission requires verifier feedback")
        if self.gated and not feedback.passed:
            return False
        for operator, before, after in zip(feedback.program, feedback.states[:-1],
                                           feedback.states[1:], strict=True):
            if self._orders is not None:
                matches = [i for i in (1, 2) if self._order(before, i) == after and before != after]
                order = matches[0] if len(matches) == 1 else 0
                self._orders[order][operator] = min(255, self._orders[order][operator] + 1)
                before = self._order(before, order)
            for symbol, result in zip(before, after, strict=True):
                self.relations.observe_edge(symbol, operator, result)
                self.episodic.observe_transition(self._key(symbol, bytes((operator,))), result)
        if len(feedback.program) > 1 and self._orders is None:
            for symbol, result in zip(feedback.states[0], feedback.states[-1], strict=True):
                self.episodic.observe_transition(self._key(symbol, feedback.program), result)
        return True

    def snapshot(self):
        episode = self.episodic.snapshot(max_bytes=self.EPISODIC_BUDGET)
        tag = b"VG01" if self._orders is None else b"VG02"
        payload = self.HEADER.pack(tag, self.gated, len(episode)) + episode + self.relations.snapshot()
        if self._orders is not None:
            payload += b"".join(self._orders)
        if len(payload) > self.BUDGET:
            raise ValueError("synthesis adaptive budget exceeded")
        return payload

    @classmethod
    def restore(cls, payload):
        if not cls.HEADER.size <= len(payload) <= cls.BUDGET:
            raise ValueError("invalid synthesis snapshot size")
        tag, gated, n = cls.HEADER.unpack_from(payload)
        split = cls.HEADER.size + n
        end = len(payload) - (768 if tag == b"VG02" else 0)
        if tag not in (b"VG01", b"VG02") or gated not in (0, 1) or split >= end:
            raise ValueError("invalid synthesis snapshot header")
        model = cls(gated=bool(gated), structured=tag == b"VG02")
        model.episodic = ConsolidatingMemory.restore(payload[cls.HEADER.size:split],
                                                     max_bytes=cls.EPISODIC_BUDGET)
        model.relations.restore(payload[split:end])
        if model._orders is not None:
            model._orders = [bytearray(payload[i:i + 256]) for i in range(end, len(payload), 256)]
        return model
