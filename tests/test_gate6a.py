"""Unit tests for Gate 6A HolographicRelationalMemory."""

from __future__ import annotations

import torch

from benchmarks.ablation_dev import CONFIG
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.holographic_memory import HolographicRelationalMemory, _make_unitary_codebook, hrr_bind, hrr_unbind


def test_unitary_hrr_bind_unbind_exact_recovery() -> None:
    cb = _make_unitary_codebook(32, 1024, seed=7)
    bound = hrr_bind(cb[3], cb[11])
    rec = hrr_unbind(bound, cb[3])
    sims = cb @ rec
    assert int(sims.argmax().item()) == 11
    assert float(sims[11].item()) > 0.99


def test_holographic_memory_2hop_composition_survives_shifted_punctuation_and_restores() -> None:
    torch.manual_seed(0)
    model = NexusExpanseByteCore(**CONFIG).eval()
    mem = HolographicRelationalMemory(model, dim=2048)

    # Teach 1-hop primitives with skin 0 ("A a? -> B", "B b? -> C")
    for _ in range(3):
        for p, y in ((b"A a?", ord("B")), (b"B b?", ord("C")), (b"C a?", ord("D")), (b"D b?", ord("A"))):
            codes, _ = mem.encode(p)
            mem.observe(codes, y)

    # Query unseen 2-hop composition with completely shifted punctuation ("A::a::b =>") -> should predict C!
    q_codes, q_prior = mem.encode(b"A::a::b =>")
    probs = mem.predict(q_codes, q_prior)
    assert int(probs.argmax().item()) == ord("C")

    # Verify compact snapshot <= 40,000 B and exact restore
    snap = mem.snapshot()
    assert len(snap) <= 40000
    mem2 = HolographicRelationalMemory(model, dim=2048)
    mem2.restore(snap)
    probs2 = mem2.predict(q_codes, q_prior)
    assert torch.allclose(probs, probs2, atol=1e-6)
