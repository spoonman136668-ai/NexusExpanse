"""Bounded recent framing statistics and confidence-protected HRR writes (DEV)."""
from collections import Counter, deque
import struct

import numpy as np
import torch

from nexus_expanse.holographic_memory import HolographicRelationalMemory, hrr_bind, hrr_unbind


class HolographicRelationalMemoryV2(HolographicRelationalMemory):
    WINDOW = 16
    FRAME = 0.65
    STRONG = 0.75
    MARGIN = 0.35
    DAMP = 0.05
    _HEADER = struct.Struct('<4sIQB4d')

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.window = deque(maxlen=self.WINDOW)

    def _record(self, codes):
        row = frozenset(codes)
        if len(self.window) == self.WINDOW:
            self.doc_freq.subtract(self.window[0])
            self.doc_freq = +self.doc_freq
        self.window.append(row)
        self.doc_freq.update(row)
        self.records += 1

    def _content(self, rest):
        return [c for c in rest if self.doc_freq[c] <= self.FRAME * len(self.window)]

    def observe(self, codes, target):
        if not 0 <= target < 256 or any(not 0 <= c < 256 for c in codes):
            raise ValueError('raw bytes required')
        self._record(codes)
        self.outcomes.add(int(target))
        anchor, rest = self._select_anchor_and_candidates(codes)
        if anchor is None or len(self.window) < 3:
            return
        ops = self._content(rest)
        if not ops:
            return
        curr = self.codebook[anchor]
        for r in ops[:-1]:
            value = hrr_unbind(self.memory, hrr_bind(curr, self.rel_codebook[r]))
            _, res, y = self._cleanup_scores(value)
            if res >= self.resonance_threshold and y >= 0:
                curr = self.codebook[y]
        key = hrr_bind(curr, self.rel_codebook[ops[-1]])
        sims = self.obj_codebook @ hrr_unbind(self.memory, key)
        outcomes = sorted(self.outcomes)
        ranked = sorted(outcomes, key=lambda y: (-float(sims[y]), y))
        best = ranked[0]
        margin = float(sims[best] - sims[ranked[1]]) if len(ranked) > 1 else 0.0
        damp = self.DAMP if best != target and float(sims[best]) >= self.STRONG and margin >= self.MARGIN else 1.0
        err = torch.zeros_like(self.memory)
        for y in outcomes:
            err += ((1.0 if y == target else 0.0) - float(sims[y])) * self.obj_codebook[y]
        self.memory = self.memory + (self.eta * damp) * hrr_bind(key, err)

    @staticmethod
    def _mask(values):
        return sum(1 << int(v) for v in values).to_bytes(32, 'little')

    @staticmethod
    def _values(data):
        mask = int.from_bytes(data, 'little')
        return frozenset(i for i in range(256) if mask & (1 << i))

    def snapshot(self):
        header = self._HEADER.pack(b'HRR2', self.dim, self.records, len(self.window),
                                   self.eta, self.temperature, self.resonance_threshold, self.prior_weight)
        payload = header + self._mask(self.outcomes) + b''.join(self._mask(row) for row in self.window)
        payload += self.memory.detach().cpu().contiguous().numpy().astype('<f8', copy=False).tobytes()
        if len(payload) > 40000:
            raise ValueError('holographic state exceeds 40000 bytes')
        return payload

    def restore(self, payload):
        h = self._HEADER.size
        if not h + 32 <= len(payload) <= 40000:
            raise ValueError('invalid checkpoint length')
        magic, dim, records, n, *cfg = self._HEADER.unpack_from(payload)
        if (magic != b'HRR2' or dim != self.dim or n != min(records, self.WINDOW)
                or tuple(cfg) != (self.eta, self.temperature, self.resonance_threshold, self.prior_weight)
                or len(payload) != h + 32 * (1+n) + 8*dim):
            raise ValueError('incompatible checkpoint')
        outcomes = self._values(payload[h:h+32])
        window = deque((self._values(payload[h+32*(i+1):h+32*(i+2)]) for i in range(n)), maxlen=self.WINDOW)
        arr = np.frombuffer(payload[h+32*(n+1):], dtype='<f8').copy()
        if not np.isfinite(arr).all():
            raise ValueError('nonfinite memory')
        self.memory = torch.from_numpy(arr.astype(np.float64))
        self.records, self.outcomes, self.window = records, set(outcomes), window
        self.doc_freq = Counter(c for row in window for c in row)
