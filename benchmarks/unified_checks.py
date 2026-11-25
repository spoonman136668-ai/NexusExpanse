"""Portable prefix and streaming checks for the integrated runtime."""
import torch

EXPERTS = ("mixed", "patch", "episodic")
PREFIX_CUTS = (1, 7, 13, 25, 31)


@torch.inference_mode()
def exactness(unified, reference, span):
    tokens = torch.tensor([list(span)])
    logits, hidden = unified.forward_neural(tokens)[:2]
    result = dict(prefix_logits=0.0, prefix_hidden=0.0, streaming_logits=0.0,
                  unified_stream_probabilities=0.0)
    for cut in PREFIX_CUTS:
        if cut >= len(span):
            continue
        a, b = unified.forward_neural(tokens[:, :cut])[:2]
        result["prefix_logits"] = max(result["prefix_logits"], (a - logits[:, :cut]).abs().max().item())
        result["prefix_hidden"] = max(result["prefix_hidden"], (b - hidden[:, :cut]).abs().max().item())
    unified.reset_context()
    for position, byte in enumerate(span):
        decoded = unified.stream.append(byte)
        result["streaming_logits"] = max(result["streaming_logits"],
                                         (decoded - logits[0, position]).abs().max().item())
    unified.reset_context()
    reference.reader.cache = None
    for byte in span:
        a, b = unified.step(byte), reference.step(byte)
        for key in EXPERTS:
            error = (getattr(a, key) - getattr(b, key)).abs().max().item()
            result["unified_stream_probabilities"] = max(result["unified_stream_probabilities"], error)
    return result
