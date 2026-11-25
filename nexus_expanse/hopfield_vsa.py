"""Bounded bipolar role binding and continuous Hopfield cleanup (DEV).

No tokenizer, byte-offset parser, raw pair lookup table, or task labels. The
slot bank stores packed bound keys and empirical target count distributions.
"""
from collections import Counter, deque
import struct

import numpy as np
import torch


class HopfieldVSAMemory:
    DIM = 256
    SLOTS = 128
    WINDOW = 16
    FRAME = .65
    BETA = 24.0
    RESONANCE = .75
    KEY_MARGIN = .10
    HOP_CONFIDENCE = .75
    HOP_SUPPORT = 2
    PRIOR_WEIGHT = .01
    BUDGET = 40000
    _HEADER = struct.Struct('<4sQBH')
    _PACKED = DIM // 8

    def __init__(self, model):
        self.model = model
        self.anchor_book = self._book(1337)
        self.relation_book = self._book(2337)
        self._popcount = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
        self.keys = np.zeros((self.SLOTS, self._PACKED), dtype=np.uint8)
        self.counts = np.zeros((self.SLOTS, 256), dtype=np.uint8)
        self.used = self.records = 0
        self.window = deque(maxlen=self.WINDOW)
        self.doc_freq = Counter()
        self.outcomes = set()
        self.roles = set()

    @classmethod
    def _book(cls, seed):
        return np.random.Generator(np.random.PCG64(seed)).integers(0, 2, (256, cls.DIM), dtype=np.int8)*2-1

    @property
    def live_codebook_bytes(self):
        return self.anchor_book.nbytes + self.relation_book.nbytes

    @property
    def live_slot_bytes(self):
        return self.keys.nbytes + self.counts.nbytes

    @torch.inference_mode()
    def encode(self, prefix):
        if not prefix: raise ValueError('nonempty raw-byte prefix required')
        logits = self.model.forward_neural(torch.tensor(list(prefix), dtype=torch.long)[None])[0]
        return tuple(prefix), logits[0,-1].double().softmax(0)

    @staticmethod
    def _validate(codes):
        if any(not isinstance(x, (int, np.integer)) or not 0 <= x < 256 for x in codes):
            raise ValueError('raw bytes required')

    def _record(self, codes):
        if len(self.window) == self.WINDOW:
            self.doc_freq.subtract(self.window[0]); self.doc_freq = +self.doc_freq
        row = frozenset(codes)
        self.window.append(row); self.doc_freq.update(row); self.records += 1

    def _key(self, anchor, rel):
        return np.packbits(self.anchor_book[anchor]*self.relation_book[rel] > 0, bitorder='little')

    def _similarities(self, query):
        # Bipolar normalized inner product = 1 - 2 Hamming distance / D.
        return 1.0 - (2.0/self.DIM)*self._popcount[np.bitwise_xor(self.keys[:self.used], query)].sum(axis=1)

    def _retrieve(self, anchor, rel):
        if not self.used: return None
        sims = self._similarities(self._key(anchor, rel))
        best = int(np.argmax(sims)); top = float(sims[best])
        second = float(np.max(np.delete(sims,best))) if self.used > 1 else -1.0
        if top < self.RESONANCE or top-second < self.KEY_MARGIN: return None
        weights = np.exp(self.BETA*(sims-top)); weights /= weights.sum()
        counts = self.counts[:self.used].astype(np.float64)
        totals = counts.sum(axis=1)
        probs = weights @ (counts/totals[:,None])
        return probs, int(totals[best]), top

    def _parse_roles(self, codes, discover=False):
        # Anchor and operator membership derive only from previous observations.
        # Framing filter governs novel-role discovery; established roles survive
        # a surface shift even if temporarily frequent in the recent window.
        anchor = None; rest = []
        for b in codes:
            if anchor is None:
                if b in self.outcomes: anchor = b
            else: rest.append(b)
        if anchor is None: return None, []
        known = [b for b in rest if b in self.roles]
        if known: return anchor, known if len(known) <= 2 else []
        if discover and len(self.window) >= 3:
            candidates = [b for b in rest if self.doc_freq[b] <= self.FRAME*len(self.window)]
            if len(candidates) == 1: return anchor, candidates
        return anchor, []

    def _resolve(self, anchor, roles):
        if anchor is None or not roles: return None
        if len(roles) == 2:
            first = self._retrieve(anchor, roles[0])
            if first is None: return None
            probs, support, _ = first; y = int(np.argmax(probs))
            if support < self.HOP_SUPPORT or probs[y] < self.HOP_CONFIDENCE: return None
            anchor = y
        return anchor, roles[-1]

    def predict(self, codes, prior):
        self._validate(codes)
        anchor, roles = self._parse_roles(codes)
        cue = self._resolve(anchor, roles)
        if cue is None: return prior
        result = self._retrieve(*cue)
        if result is None: return prior
        probs = torch.from_numpy(result[0]).to(device=prior.device, dtype=prior.dtype)
        return self.PRIOR_WEIGHT*prior + (1-self.PRIOR_WEIGHT)*probs

    def _write(self, anchor, rel, target, weight):
        key = self._key(anchor, rel)
        # Exact packed-key similarity selects the update slot, rather than a
        # raw-byte dictionary. Near matches are retrieval-only, not write aliases.
        sims = self._similarities(key)
        hits = np.flatnonzero(sims == 1.0)
        if len(hits): i = int(hits[0])
        else:
            if self.used == self.SLOTS: return False
            i = self.used; self.used += 1; self.keys[i] = key
        row = self.counts[i]
        if int(row[target])+weight > 255:
            row[:] = ((row.astype(np.uint16)+1)//2).astype(np.uint8)
        row[target] = int(row[target])+weight
        return True

    def observe(self, codes, target):
        self._validate(codes); self._validate((target,))
        self._record(codes)
        anchor, roles = self._parse_roles(codes, discover=True)
        cue = self._resolve(anchor, roles)
        # Target never participates in anchor/role resolution. Direct evidence
        # has twice the weight of an inferred edge; no last-write overwrite.
        if cue is not None and self._write(*cue, int(target), 2 if len(roles)==1 else 1):
            self.roles.add(roles[-1])
        self.outcomes.add(int(target))

    @staticmethod
    def _mask(values):
        return sum(1 << int(v) for v in values).to_bytes(32,'little')

    @staticmethod
    def _values(payload):
        mask = int.from_bytes(payload,'little')
        return frozenset(i for i in range(256) if mask & (1<<i))

    def snapshot(self):
        payload = self._HEADER.pack(b'HVS1',self.records,len(self.window),self.used)
        payload += self._mask(self.outcomes)+self._mask(self.roles)
        payload += b''.join(self._mask(row) for row in self.window)
        payload += self.keys[:self.used].tobytes()+self.counts[:self.used].tobytes()
        if len(payload)>self.BUDGET: raise ValueError('state budget exceeded')
        return payload

    def restore(self, payload):
        h=self._HEADER.size
        if not h+64 <= len(payload) <= self.BUDGET: raise ValueError('invalid checkpoint size')
        magic, records, n, used=self._HEADER.unpack_from(payload)
        offset=h+64+32*n
        if (magic!=b'HVS1' or n!=min(records,self.WINDOW) or used>min(records,self.SLOTS)
                or len(payload)!=offset+used*(self._PACKED+256)):
            raise ValueError('invalid checkpoint geometry')
        outcomes=set(self._values(payload[h:h+32])); roles=set(self._values(payload[h+32:h+64]))
        window=deque((self._values(payload[h+64+32*i:h+96+32*i]) for i in range(n)),maxlen=self.WINDOW)
        end=offset+used*self._PACKED
        keys=np.frombuffer(payload[offset:end],dtype=np.uint8).reshape(used,self._PACKED)
        counts=np.frombuffer(payload[end:],dtype=np.uint8).reshape(used,256)
        if (len({r.tobytes() for r in keys})!=used or np.any(counts.sum(axis=1)==0)
                or not set(np.flatnonzero(counts.any(axis=0))).issubset(outcomes) or (used and not roles)):
            raise ValueError('invalid checkpoint content')
        self.keys.fill(0); self.counts.fill(0)
        self.keys[:used]=keys; self.counts[:used]=counts
        self.used,self.records,self.window,self.outcomes,self.roles=used,records,window,outcomes,roles
        self.doc_freq=Counter(b for row in window for b in row)
