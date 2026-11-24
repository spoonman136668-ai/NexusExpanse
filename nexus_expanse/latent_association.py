"""DEV-only sparse transitions over winner-take-all backbone feature codes."""
from collections import Counter
import json

import torch
import torch.nn.functional as F


class LatentAssociation:
    def __init__(self, model):
        self.model = model
        with torch.inference_mode():
            # Fixed dictionary derived entirely from the supplied frozen backbone.
            _, h, _, _ = model.forward_neural(torch.arange(256)[:, None])
            self.prototypes = F.normalize(h[:, 0], dim=1)
        self.frequency = Counter()
        self.outcomes = set()
        self.edges = {}
        self.records = 0

    @torch.inference_mode()
    def encode(self, prefix):
        logits, h, _, _ = self.model.forward_neural(torch.tensor(list(prefix))[None])
        codes = (F.normalize(h[0], dim=1) @ self.prototypes.T).argmax(1)
        return tuple(codes.tolist()), logits[0, -1].double().softmax(0)

    def ignore_cue(self, anchor, code):
        return self.frequency[code] > .45 * max(1, self.records)

    def route(self, codes):
        anchor = None
        steps = []
        for code in codes:
            if anchor is None:
                if code in self.outcomes:
                    anchor = code
                continue
            # Frequent framing features provide little evidence of a relation.
            if self.ignore_cue(anchor, code):
                continue
            steps.append((anchor, code))
            counts = self.edges.get(steps[-1])
            if counts:
                anchor = counts.most_common(1)[0][0]
        return steps

    def predict(self, codes, prior):
        steps = self.route(codes)
        if not steps:
            return prior
        # Product of posterior predictive masses along the WTA path.
        mass = 1.
        for edge in steps[:-1]:
            counts = self.edges.get(edge)
            if not counts:
                return prior
            mass *= max(counts.values()) / (counts.total() + .1)
        counts = self.edges.get(steps[-1])
        if not counts:
            return prior
        scale = mass / (counts.total() + .1)
        probs = prior * (1 - scale * counts.total())
        for y, count in counts.items():
            probs[self.output_byte(y)] += scale * count
        return probs

    def output_byte(self, code):
        return code

    def observe(self, codes, target):
        path = self.route(codes)
        if path:
            self.edges.setdefault(path[-1], Counter())[target] += 1
        self.outcomes.add(target)
        self.frequency.update(set(codes))
        self.records += 1

    def snapshot(self):
        data = [1, self.records, sorted(self.outcomes), sorted(self.frequency.items()),
                [[x, r, list(counts.items())] for (x, r), counts in self.edges.items()]]
        payload = json.dumps(data, separators=(',', ':')).encode()
        if len(payload) > 40000:
            raise ValueError('adaptive state exceeds 40000 bytes')
        return payload

    def restore(self, payload):
        if len(payload) > 40000:
            raise ValueError('oversized adaptive state')
        version, self.records, outcomes, freq, rows = json.loads(payload)
        if version != 1:
            raise ValueError('invalid adaptive state version')
        self.outcomes = set(outcomes)
        self.frequency = Counter(dict(freq))
        self.edges = {(x, r): Counter(dict(pairs)) for x, r, pairs in rows}

class AlignedLatentAssociation(LatentAssociation):
    """Frozen byte-embedding identities for routing; contextual logits for the prior."""

    def __init__(self, model):
        self.model = model
        # Canonical names for the 256 frozen byte embeddings. Stable tie-break by
        # byte index makes this a bijection even for coincident coordinates.
        self.bytes_by_code = model.emb.weight.detach()[:, 0].argsort(stable=True).tolist()
        self.codes_by_byte = [0] * 256
        for code, byte in enumerate(self.bytes_by_code):
            self.codes_by_byte[byte] = code
        self.frequency = Counter()
        self.outcomes = set()
        self.edges = {}
        self.records = 0

    @torch.inference_mode()
    def encode(self, prefix):
        logits = self.model.forward_neural(torch.tensor(list(prefix))[None])[0]
        return tuple(self.codes_by_byte[b] for b in prefix), logits[0,-1].double().softmax(0)

    def output_byte(self, code):
        return self.bytes_by_code[code]

    def observe(self, codes, target):
        super().observe(codes, self.codes_by_byte[target])

    def ignore_cue(self, anchor, code):
        return (anchor, code) not in self.edges and super().ignore_cue(anchor, code)

    def snapshot(self):
        payload = b'A2' + super().snapshot()
        if len(payload) > 40000:
            raise ValueError('adaptive state exceeds 40000 bytes')
        return payload

    def restore(self, payload):
        if len(payload) > 40000 or not payload.startswith(b'A2'):
            raise ValueError('invalid aligned state')
        super().restore(payload[2:])
