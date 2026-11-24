"""Opt-in streaming patch reader for a fixed raw-byte neural backbone."""
from collections import deque
import numpy as np
import torch


class BytePatchReader:
    def __init__(self, model, counts, threshold_bits=4.0):
        self.model = model
        self.counts = counts
        self.totals = counts.sum(axis=1) + 256.
        self.minimum_probability = 2. ** -threshold_bits
        self.events = deque(maxlen=model.max_seq_len)
        self.previous = None
        self.pending = None
        self.deep_calls = 0

    def reset_context(self):
        self.events.clear()
        self.previous = self.pending = None

    @torch.inference_mode()
    def forecast(self, observed):
        # Surprise belongs to the byte already observed, before the next target.
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
            tokens = torch.tensor(list(self.events))[None]
            return self.model.forward_neural(tokens)[0][0,-1].double().softmax(0)
        # Fold a run into its latest raw byte. The local expert predicts throughout.
        self.pending = observed
        return torch.from_numpy((self.counts[observed].astype(np.float64)+1.) / self.totals[observed])
