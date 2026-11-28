"""Shared probability operations for causal byte decoding."""
import torch


def engram_forecast(probabilities, history, store, alpha):
    gain, match = store.lookup(history, max_alpha=alpha)
    if not gain:
        return probabilities, 0.0
    indices, counts = zip(*match.counts, strict=True)
    mixed = probabilities * (1 - gain)
    mixed[list(indices)] += mixed.new_tensor([gain * count / match.support for count in counts])
    return mixed / mixed.sum(), gain


def nucleus_byte(logs, temperature, top_p, generator):
    probabilities = ((logs - logs.max()) / temperature).softmax(0)
    sorted_probs, ids = probabilities.sort(descending=True, stable=True)
    keep = sorted_probs.cumsum(0) - sorted_probs < top_p
    selected = torch.multinomial(sorted_probs * keep, 1, generator=generator)
    return int(ids[selected].item())
