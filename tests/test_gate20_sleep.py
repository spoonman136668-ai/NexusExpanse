import pytest
import torch
from benchmarks.gate20_sleep_consolidation_dev import collect, create, fingerprint
from nexus_expanse.sleep_consolidation import adaptive_bytes, match_random_prune
from nexus_expanse.unified import NexusExpanseUnified


def test_sleep_replay_is_bounded_and_cold_scoring_does_not_learn():
    torch.set_num_threads(1)
    model = create(42)
    rows = [b"abcabc defdef abc\n", b"abcabcd fedfed\n"] * 3
    training, replay = collect(model, rows, adapt=True)
    assert replay and len(replay) <= 16
    before = model.episodic.num_observations
    report = model.consolidate_sleep(replay)
    assert model.episodic.num_observations == before + len(replay)
    assert report["episodic_entries"] > 0
    restored = NexusExpanseUnified.restore(model.snapshot())
    assert restored.sleep_budget and fingerprint(restored) == fingerprint(model)
    scored, unused = collect(restored, rows[:1], adapt=False)
    assert not unused and restored.episodic.num_observations == model.episodic.num_observations
    assert scored["peak_adaptive_bytes"] <= 40_000


def test_random_prune_matches_actual_serialized_bytes():
    torch.set_num_threads(1)
    a, b = create(43), create(43)
    rows = [b"raw memory probe raw memory\n", b"probe memory bytes raw\n"] * 2
    for model in (a, b):
        collect(model, rows, adapt=True)
    a.consolidate_sleep()
    match = match_random_prune(a, b, 43)
    assert match["random_prune"]["episodic_entries"] > 0
    assert adaptive_bytes(a) == adaptive_bytes(b) == match["target_bytes"]
    for model in (a, b):
        assert adaptive_bytes(NexusExpanseUnified.restore(model.snapshot())) == match["target_bytes"]


def test_sleep_refuses_inflight_and_invalid_ledgers():
    torch.set_num_threads(1)
    model = create(44)
    model.step(18)
    with pytest.raises(ValueError, match="reset"):
        model.consolidate_sleep()
    model.reset_context()
    with pytest.raises(ValueError, match="ledger"):
        model.consolidate_sleep([(b"x", 7, 1.)] * 17)
