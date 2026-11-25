import numpy as np
import pytest
import torch

from benchmarks.gate14_unified_dev import exactness
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.deltanet import DeltaNetMonarchByteCore
from nexus_expanse.patching import CompactBigram
from nexus_expanse.unified import NexusExpanseUnified


def unified(kind, limit=32):
    torch.set_num_threads(1)
    torch.manual_seed(23)
    if kind == "attention":
        model = NexusExpanseByteCore(n_layers=2, dim=32, n_heads=4, ffn_mult=2,
                                      dropout=0.0, max_seq_len=limit)
    else:
        model = DeltaNetMonarchByteCore(n_layers=2, dim=32, n_heads=4)
    counts = np.zeros((256, 256), dtype="<u4")
    counts[1, 2] = 500
    return NexusExpanseUnified(model, CompactBigram.from_dense(counts))


@pytest.mark.parametrize("kind", ["attention", "delta"])
def test_unified_matches_fixed_causal_reference(kind):
    model = unified(kind)
    result = exactness(model, NexusExpanseUnified.restore(model.snapshot()), bytes([1, 2, *range(10, 26)]))
    assert all(v == 0.0 for v in result.values()), result


@pytest.mark.parametrize("kind", ["attention", "delta"])
def test_unified_midstream_restore_and_post_outcome_updates(kind):
    model = unified(kind, limit=8)
    first = model.step(1)
    assert model.episodic.num_observations == model.arbitrator.updates == 0
    assert torch.isclose(first.mixed.sum(), torch.tensor(1., dtype=torch.float64))
    model.step(2)
    assert model.reader.pending == 2
    assert model.episodic.num_observations == model.arbitrator.updates == 1
    for byte in range(5, 18):
        model.step(byte)
    model.step(1)
    model.step(2)
    restored = NexusExpanseUnified.restore(model.snapshot())
    assert restored.reader.pending == 2
    for byte in range(20, 31):
        expected, actual = model.step(byte), restored.step(byte)
        assert torch.equal(expected.mixed, actual.mixed)
        assert expected.confidence == actual.confidence
    assert restored.episodic.snapshot(max_bytes=40000) == model.episodic.snapshot(max_bytes=40000)
    restored.reset_context()
    updates = restored.arbitrator.updates
    restored.step(0)
    assert restored.arbitrator.updates == updates
