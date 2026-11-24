import json
import pytest
import torch
from nexus_expanse import BoundedEpisodicByteMemory, NexusExpanseByteCore


def test_context_budget_and_deterministic_observation_recency():
    m = BoundedEpisodicByteMemory(2, max_order=1)
    m.ingest_stream(b'abca')
    assert list(m._table) == [b'b', b'c']
    m.predict_probs(b'b')
    m.observe_transition(b'd', 65)
    assert list(m._table) == [b'c', b'd']
    m.observe_transition(b'c', 66)
    m.observe_transition(b'e', 67)
    assert list(m._table) == [b'c', b'e']
    assert set(m._totals) == set(m._table)
    assert m.num_observations == 6


def test_snapshot_exact_prediction_and_next_eviction(tmp_path):
    m = BoundedEpisodicByteMemory(32, max_order=4)
    m.ingest_stream(b'abcd efg abcd efg' * 12)
    path = tmp_path / 'memory.json'
    m.save(path, max_bytes=20000)
    restored = BoundedEpisodicByteMemory.load(path, max_bytes=20000, max_contexts=32)
    assert restored.snapshot(max_bytes=20000) == path.read_bytes()
    for ctx in (b'abcd', b'efg', b'zzzz', b''):
        p, c = m.predict_probs(ctx)
        q, d = restored.predict_probs(ctx)
        assert torch.equal(p, q) and c == d
    for obj in (m, restored):
        obj.ingest_stream(b'new unrelated suffix data')
    assert m.snapshot(max_bytes=20000) == restored.snapshot(max_bytes=20000)
    with pytest.raises(FileExistsError):
        m.save(path, max_bytes=20000)


def test_corrupt_or_overbudget_snapshot_rejected():
    m = BoundedEpisodicByteMemory(5)
    m.ingest_stream(b'abcd')
    blob = m.snapshot(max_bytes=10000)
    with pytest.raises(ValueError):
        m.snapshot(max_bytes=2)
    with pytest.raises(ValueError):
        m.restore(blob, max_bytes=len(blob)-1, max_contexts=5)
    with pytest.raises(ValueError):
        m.restore(blob, max_bytes=10000, max_contexts=4)
    state = json.loads(blob)
    state['contexts'][0][2][0][1] = -1
    with pytest.raises(ValueError):
        m.restore(json.dumps(state).encode(), max_bytes=10000, max_contexts=5)
    for invalid in (b'{}', b'not json', b'{"schema":1,"schema":2}'):
        with pytest.raises(ValueError):
            m.restore(invalid, max_bytes=10000, max_contexts=5)


def test_memory_budget_can_forget():
    m = BoundedEpisodicByteMemory(1, max_order=1)
    m.ingest_stream(b'ax' * 20 + b'a')
    m.observe_transition(b'a', ord('x'), 50)
    p, _ = m.predict_probs(b'a')
    m.observe_transition(b'b', ord('y'))
    q, _ = m.predict_probs(b'a')
    assert q[ord('x')] < p[ord('x')]


def test_core_accepts_restored_memory():
    m = BoundedEpisodicByteMemory(32)
    m.ingest_stream(b'abc' * 20)
    model = NexusExpanseByteCore(n_layers=1, dim=16, n_heads=2, max_seq_len=16)
    model.episodic = m
    a = model.evaluate_streaming(b'abcabc', seq_len=16)
    model.episodic = m.restore(m.snapshot(max_bytes=10000), max_bytes=10000, max_contexts=32)
    b = model.evaluate_streaming(b'abcabc', seq_len=16)
    assert a == b


def test_cold_process_restore(tmp_path):
    import subprocess
    import sys
    m = BoundedEpisodicByteMemory(32, max_order=4)
    m.ingest_stream(b'abc def abc def ' * 20)
    path = tmp_path / 'cold.json'
    m.save(path, max_bytes=20000)
    expected, confidence = m.predict_probs(b'abc')
    script = '''import json, sys
from pathlib import Path
from nexus_expanse import BoundedEpisodicByteMemory
m = BoundedEpisodicByteMemory.load(Path(sys.argv[1]), max_bytes=20000, max_contexts=32)
p, c = m.predict_probs(b'abc')
print(json.dumps([p.tolist(), c]))
'''
    p, c = json.loads(subprocess.check_output([sys.executable, '-c', script, str(path)], text=True))
    assert p == expected.tolist() and c == confidence
