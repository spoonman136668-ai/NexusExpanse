"""Opt-in byte-budgeted episodic consolidation (DEV, no task labels)."""
from __future__ import annotations
import json
from nexus_expanse.persistent_memory import BoundedEpisodicByteMemory


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


class ConsolidatingMemory(BoundedEpisodicByteMemory):
    """Same posterior as episodic memory; vary admission and eviction only.

    The bound is exact serialized bytes, not resident Python heap. Reads are
    pure. Support eviction protects repeated contexts, but can preserve stale
    associations and starve novel ones. No task IDs or protected target labels.
    """
    POLICIES = ('lru', 'support', 'sparse_support')

    def __init__(self, budget=40000, policy='lru', max_order=32):
        if type(budget) is not int or not 4096 <= budget <= 10_000_000:
            raise ValueError('invalid byte budget')
        if policy not in self.POLICIES:
            raise ValueError('invalid policy')
        super().__init__(budget, max_order)
        self.budget = budget
        self.policy = policy
        self._sizes = {}
        self._row_bytes = 0
        self._prefix = _encode({'schema': 'nexus-consolidation-v1', 'policy': policy, 'budget': budget}) + b'\n'

    def _row_size(self, key):
        return len(_encode([key.hex(), self._totals[key], sorted(self._table[key].items())]))

    def serialized_size(self):
        header = {'schema': 'nexus-bounded-episodic-v1', 'max_contexts': self.max_contexts,
                  'max_order': self.max_order, 'smoothing': self.smoothing,
                  'observations': self.num_observations, 'unigram': self._unigram,
                  'unigram_total': self._unigram_total, 'contexts': []}
        return len(self._prefix) + len(_encode(header)) + self._row_bytes + max(0, len(self._table)-1)

    def observe_transition(self, context, next_byte, weight=1.0):
        if not isinstance(context, bytes) or type(next_byte) is not int or not 0 <= next_byte <= 255 or weight != 1.0:
            raise ValueError('only unit byte transitions supported')
        n = min(len(context), self.max_order)
        orders = range(1, n+1) if self.policy != 'sparse_support' else sorted({i for i in (1,2,4,8,16,32,n) if 0 < i <= n})
        self._unigram[next_byte] += 1.; self._unigram_total += 1.; self.num_observations += 1
        for order in orders:
            key = context[-order:]
            bucket = self._table.pop(key, {})
            self._row_bytes -= self._sizes.get(key, 0)
            bucket[next_byte] = bucket.get(next_byte, 0.) + 1.
            self._table[key] = bucket
            self._totals[key] += 1.
            self._sizes[key] = self._row_size(key)
            self._row_bytes += self._sizes[key]
        size = self.serialized_size()
        while size > self.budget and self._table:
            if self.policy == 'lru':
                key = next(iter(self._table))
            else:
                # Stable tie break: oldest observed context first.
                key = min(self._table, key=self._totals.__getitem__)
            size -= self._sizes[key] + int(len(self._table)>1)
            self._row_bytes -= self._sizes.pop(key)
            del self._table[key]; del self._totals[key]
        if size > self.budget:
            raise ValueError('unigram metadata exhausted budget')

    def snapshot(self, *, max_bytes):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('invalid external budget')
        limit = min(self.budget, max_bytes)
        payload = self._prefix + super().snapshot(max_bytes=limit-len(self._prefix))
        if len(payload) != self.serialized_size():
            raise RuntimeError('byte accounting mismatch')
        return payload

    @classmethod
    def restore(cls, payload, *, max_bytes, max_contexts=None):
        if type(max_bytes) is not int or len(payload)>max_bytes:
            raise ValueError('snapshot exceeds budget')
        try:
            head, raw = payload.split(b'\n', 1)
            meta = json.loads(head)
            if set(meta) != {'schema','policy','budget'} or meta['schema'] != 'nexus-consolidation-v1' or meta['budget'] > max_bytes:
                raise ValueError('invalid consolidation metadata')
            if max_contexts is not None and (type(max_contexts) is not int or meta['budget'] > max_contexts):
                raise ValueError('external context limit exceeded')
            inner = BoundedEpisodicByteMemory.restore(raw, max_bytes=max_bytes, max_contexts=meta['budget'])
            mem = cls(meta['budget'], meta['policy'], inner.max_order)
            if inner.max_contexts != mem.max_contexts or inner.smoothing != mem.smoothing:
                raise ValueError('incompatible inner configuration')
            mem._table, mem._totals = inner._table, inner._totals
            mem._unigram, mem._unigram_total = inner._unigram, inner._unigram_total
            mem.num_observations = inner.num_observations
            mem._sizes = {key: mem._row_size(key) for key in mem._table}
            mem._row_bytes = sum(mem._sizes.values())
            if mem.snapshot(max_bytes=max_bytes) != payload:
                raise ValueError('noncanonical checkpoint')
            return mem
        except (TypeError, KeyError, AttributeError, OverflowError, RecursionError) as exc:
            raise ValueError('invalid consolidation checkpoint') from exc
