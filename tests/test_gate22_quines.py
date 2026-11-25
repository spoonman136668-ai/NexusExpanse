import pytest
import torch

from benchmarks.gate20_sleep_consolidation_dev import NO_SLEEP, collect, create, fingerprint
from benchmarks.gate22_quine_sleep_dev import ARMS, FREQUENCY, QUINE, match_states
from nexus_expanse.consolidation import ConsolidatingMemory
from nexus_expanse.sleep_consolidation import adaptive_bytes
from nexus_expanse.unified import NexusExpanseUnified


def test_quines_rank_disagreement_and_save_supported_distinct_witnesses():
    torch.set_num_threads(1)
    mem = ConsolidatingMemory(8000, "sparse_support", 8)
    data = [(b"abxy", 40, 3.5, .9), (b"cdxy", 41, 4., .1), (b"abc", 42, 2., .1)]
    selected = mem.consolidate_quines("math", data)["selected"]
    assert [x["byte"] for x in selected] == [40, 42]
    mem.consolidate_quines("code", [(b"replay", 43, 4., .5)])
    checkpoint = mem.snapshot(max_bytes=8000)
    assert len(checkpoint) == mem.serialized_size()
    restored = ConsolidatingMemory.restore(checkpoint, max_bytes=8000)
    assert restored.snapshot(max_bytes=8000) == checkpoint
    assert restored.quine_retention() == mem.quine_retention()
    assert all(x["support"] > 0 for domain in restored.quine_retention().values() for x in domain)


def test_invalid_quine_input_preserves_existing_memory():
    mem = ConsolidatingMemory(8000, "sparse_support", 8)
    before = mem.snapshot(max_bytes=8000)
    with pytest.raises(ValueError, match="observed"):
        mem.consolidate_quines("code", [(b"invalid", 3, .4, 1.1)])
    assert mem.snapshot(max_bytes=8000) == before


def test_four_arm_matching_preserves_quines_and_no_sleep_relations():
    torch.set_num_threads(1)
    models = {a: create(42) for a in ARMS}
    rows = [b"raw byte evidence across domains\n", b"byte evidence stays raw\n"] * 3
    for a, model in models.items():
        _, ledger = collect(model, rows, adapt=True, track_disagreement=a == QUINE)
    models[QUINE].episodic.consolidate_quines("math", ledger)
    models[FREQUENCY].consolidate_sleep()
    base_rel = models[NO_SLEEP].reasoning.snapshot()
    result = match_states(models, 42)
    assert len(set(adaptive_bytes(m) for m in models.values())) == 1
    assert result["target_bytes"] == adaptive_bytes(models[QUINE])
    assert models[NO_SLEEP].reasoning.snapshot() == base_rel
    quine = models[QUINE]
    restored = NexusExpanseUnified.restore(quine.snapshot())
    assert fingerprint(restored) == fingerprint(quine)
    assert all(w["support"] > 0 for xs in restored.episodic.quine_retention().values() for w in xs)
