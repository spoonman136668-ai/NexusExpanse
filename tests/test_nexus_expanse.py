"""Unit and invariant tests for Nexus Expanse v0.1."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.memory import EpisodicByteMemory, HebbianFastWeightAdapter
from nexus_expanse.substrate import DirectMonarchLinear


def test_direct_monarch_is_full_rank() -> None:
    torch.manual_seed(0)
    layer = DirectMonarchLinear(128, 128, n_blocks=16, bias=False)
    with torch.inference_mode():
        w = layer(torch.eye(128))
    assert int(torch.linalg.matrix_rank(w).item()) == 128


def test_episodic_memory_zero_forgetting_on_disjoint_domain() -> None:
    mem = EpisodicByteMemory(max_order=6)
    domain_a = b"def alpha_protocol(x): return x + 1\n" * 5
    domain_b = b"SELECT * FROM telemetry WHERE status = 'OK';\n" * 5

    mem.ingest_stream(domain_a)
    p_before, _ = mem.predict_probs(b"def alpha_prot")
    prob_o_before = float(p_before[ord("o")].item())

    # Ingest completely disjoint domain B and verify Domain A recall holds >99% (negligible backoff drift)
    mem.ingest_stream(domain_b)
    p_after, _ = mem.predict_probs(b"def alpha_prot")
    prob_o_after = float(p_after[ord("o")].item())

    assert prob_o_before > 0.99
    assert prob_o_after > 0.99
    assert abs(prob_o_after - prob_o_before) < 0.005


def test_hebbian_fast_weight_only_updates_on_high_surprise() -> None:
    adapter = HebbianFastWeightAdapter(dim=32, lr=0.1)
    h = torch.randn(32)
    probs = torch.full((256,), 1.0 / 256.0)

    did_low = adapter.hebbian_update(h, probs, target_byte=65, surprise_nll=0.5, surprise_threshold=1.5)
    assert did_low is False and adapter.updates_applied == 0

    did_high = adapter.hebbian_update(h, probs, target_byte=65, surprise_nll=3.0, surprise_threshold=1.5)
    assert did_high is True and adapter.updates_applied == 1
    delta = adapter.forward_delta(h)
    assert float(delta[65].item()) > float(delta[66].item())


def test_nexus_core_strict_prefix_causality_and_tokenless_bytes() -> None:
    torch.manual_seed(0)
    core = NexusExpanseByteCore(n_layers=2, dim=64, n_heads=2, target_skip_rate=0.50, dropout=0.0)
    raw = bytes(range(256))[:32]
    xb = torch.tensor(list(raw), dtype=torch.long).unsqueeze(0)
    core.calibrate_gates([(xb[:, :-1], xb[:, 1:])])
    core.eval()
    with torch.inference_mode():
        full_logits, _, _, _ = core.forward_neural(xb)
        pref_logits, _, _, _ = core.forward_neural(xb[:, :16])
    assert torch.allclose(full_logits[:, :16], pref_logits, atol=1e-5)
