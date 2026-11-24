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

    def route(self, codes):
        anchor = None
        steps = []
        for code in codes:
            if anchor is None:
                if code in self.outcomes:
                    anchor = code
                continue
            # Frequent framing features provide little evidence of a relation.
            if self.frequency[code] > .45 * max(1, self.records):
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
            probs[y] += scale * count
        return probs

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
