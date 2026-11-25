"""Opt-in byte-budgeted episodic consolidation (DEV, no task labels)."""
from __future__ import annotations
import json
import math
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
        self._quines = {}
        self._protected_contexts = frozenset()
        self._set_quines({})

    QUINES_PER_DOMAIN = 8
    QUINE_DOMAINS = 3
    QUINE_LEDGER_LIMIT = 16

    def _set_quines(self, data):
        if not isinstance(data, dict) or len(data) > self.QUINE_DOMAINS:
            raise ValueError("invalid quine domain ledger")
        parsed = {}
        for domain, pairs in data.items():
            if (not isinstance(domain, str) or not 1 <= len(domain) <= 24 or not domain.isascii()
                    or not isinstance(pairs, list) or not 1 <= len(pairs) <= self.QUINES_PER_DOMAIN):
                raise ValueError("invalid quine domain")
            parsed[domain] = tuple((bytes.fromhex(ctx), target) for ctx, target in pairs)
            if (len({ctx[-2:] for ctx, _ in parsed[domain]}) != len(pairs)
                    or any(not 1 <= len(ctx) <= min(8, self.max_order) or type(target) is not int
                           or not 0 <= target <= 255 for ctx, target in parsed[domain])):
                raise ValueError("invalid quine witnesses")
        self._quines = parsed
        header = dict(schema="nexus-consolidation-v1", policy=self.policy, budget=self.budget)
        if parsed:
            header["quines"] = {domain: [[ctx.hex(), b] for ctx, b in pairs]
                                for domain, pairs in parsed.items()}
        self._prefix = _encode(header) + b"\n"

    @property
    def quine_contexts(self):
        return frozenset(ctx for rows in self._quines.values() for ctx, _ in rows)

    def quine_retention(self):
        result = {}
        for domain, pairs in sorted(self._quines.items()):
            records = []
            for ctx, target in pairs:
                distribution, _ = self.predict_probs(ctx)
                support = self._table.get(ctx, {}).get(target, 0)
                records.append(dict(context=ctx.hex(), byte=target, support=support,
                                    probability=float(distribution[target]),
                                    top1=int(distribution.argmax()) == target))
            result[domain] = records
        return result

    def consolidate_quines(self, domain, boundary):
        """Replay a bounded, distinct set of observed boundary witnesses."""
        rows = list(boundary)
        if not 1 <= len(rows) <= self.QUINE_LEDGER_LIMIT or any(
            not isinstance(ctx, bytes) or not 1 <= len(ctx) <= min(8, self.max_order)
            or type(byte) is not int or not 0 <= byte <= 255
            or not math.isfinite(nll) or nll < 0 or not math.isfinite(disagreement)
            or not 0 <= disagreement <= 1 for ctx, byte, nll, disagreement in rows
        ):
            raise ValueError("invalid observed quine candidates")
        chosen = {}
        for row in sorted(rows, key=lambda r: (-(r[2] + r[3]), r[0], r[1])):
            chosen.setdefault(row[0][-2:], row)
            if len(chosen) >= self.QUINES_PER_DOMAIN:
                break
        candidate = self.restore(self.snapshot(max_bytes=self.budget), max_bytes=self.budget)
        ledger = {d: [[ctx.hex(), b] for ctx, b in records] for d, records in self._quines.items()}
        ledger[domain] = [[ctx.hex(), b] for ctx, b, _, _ in chosen.values()]
        candidate._set_quines(ledger)
        candidate._protected_contexts = candidate.quine_contexts
        for _, pairs in sorted(candidate._quines.items()):
            for ctx, target in pairs:
                candidate.observe_transition(ctx, target)
        eligible = [key for key in candidate._table if key not in candidate._protected_contexts]
        removed = eligible[:len(candidate._table) // 8]
        for key in removed:
            candidate._drop_context(key)
        candidate._protected_contexts = frozenset()
        retained = candidate.quine_retention()
        if not all(row["support"] > 0 for records in retained.values() for row in records):
            raise ValueError("quine replay exceeded admissible support storage")
        candidate.snapshot(max_bytes=candidate.budget)
        self.__dict__.update(candidate.__dict__)
        return dict(selected=[dict(context=ctx.hex(), byte=b, surprisal=nll, disagreement=disagreement)
                              for ctx, b, nll, disagreement in chosen.values()],
                    removed_contexts=len(removed), witnesses=retained)

    def _drop_context(self, key):
        self._row_bytes -= self._sizes.pop(key)
        del self._table[key]
        del self._totals[key]

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
            candidates = (key for key in self._table if key not in self._protected_contexts)
            if self.policy == 'lru':
                key = next(candidates, None)
            else:
                # Stable tie break: oldest observed context first.
                key = min(candidates, key=self._totals.__getitem__, default=None)
            if key is None:
                break
            size -= self._sizes[key] + int(len(self._table)>1)
            self._drop_context(key)
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
            fields = {'schema', 'policy', 'budget'}
            if (set(meta) not in (fields, fields | {'quines'}) or meta['schema'] != 'nexus-consolidation-v1'
                    or meta['budget'] > max_bytes):
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
            mem._set_quines(meta.get('quines', {}))
            if mem.snapshot(max_bytes=max_bytes) != payload:
                raise ValueError('noncanonical checkpoint')
            return mem
        except (TypeError, KeyError, AttributeError, OverflowError, RecursionError) as exc:
            raise ValueError('invalid consolidation checkpoint') from exc
