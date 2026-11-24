"""Bounded prequential two-expert arbitration; no target is used before scoring."""
from __future__ import annotations

import math


class ReliabilityArbitrator:
    """Bayesian expert mixture with fixed-share recovery and optional confidence bins.

    One scalar per bin tracks memory reliability. The fixed share gives a previously
    disfavored expert a chance to recover after distribution shift. State is owned by
    the caller; a fresh instance is required for an independent evaluation arm.
    """

    def __init__(self, bins: int = 1, share: float = 0.01):
        if type(bins) is not int or not 1 <= bins <= 16:
            raise ValueError("bins must be in 1..16")
        if not math.isfinite(share) or not 0 < share < 0.5:
            raise ValueError("share must be in (0, 0.5)")
        self.bins = bins
        self.share = share
        self._weights = [0.5] * bins
        self.updates = 0

    def _bin(self, confidence: float) -> int:
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid memory confidence")
        return min(self.bins - 1, int(confidence * self.bins))

    def weight(self, confidence: float) -> float:
        return self._weights[self._bin(confidence)]

    def observe(self, confidence: float, neural_likelihood: float, memory_likelihood: float) -> None:
        """Update only after observing the true outcome, using both pre-update forecasts."""
        idx = self._bin(confidence)
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in (neural_likelihood, memory_likelihood)):
            raise ValueError("invalid expert likelihood")
        w = self._weights[idx]
        # Match the evaluator's probability floor and avoid a zero denominator.
        a = w * max(memory_likelihood, 1e-12)
        b = (1 - w) * max(neural_likelihood, 1e-12)
        self._weights[idx] = self.share + (1 - 2 * self.share) * (a / (a + b))
        self.updates += 1
