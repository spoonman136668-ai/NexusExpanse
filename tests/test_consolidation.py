import pytest
import torch
from nexus_expanse.consolidation import ConsolidatingMemory

@pytest.mark.parametrize('policy', ConsolidatingMemory.POLICIES)
def test_budget_and_restore(policy):
    m=ConsolidatingMemory(4096,policy,16)
    data=bytes(range(256))*2
    for i in range(len(data)-1):
        m.observe_transition(data[max(0,i-15):i+1],data[i+1])
        assert m.serialized_size() <= 4096
    p=m.snapshot(max_bytes=4096)
    n=ConsolidatingMemory.restore(p,max_bytes=4096)
    assert n.snapshot(max_bytes=4096)==p
    assert torch.equal(m.predict_probs(data[-16:])[0],n.predict_probs(data[-16:])[0])
    assert m.snapshot(max_bytes=4096)==p
    m.observe_transition(b'next',42); n.observe_transition(b'next',42)
    assert m.snapshot(max_bytes=4096)==n.snapshot(max_bytes=4096)
    with pytest.raises(ValueError): ConsolidatingMemory.restore(p,max_bytes=4095)
    with pytest.raises(ValueError): ConsolidatingMemory.restore(p+b' ',max_bytes=4096)


def test_conflicting_outcomes_not_overwritten():
    m=ConsolidatingMemory(4096,'sparse_support')
    for x in (7,8,7,8): m.observe_transition(b'cue',x)
    p,_=m.predict_probs(b'cue')
    assert p[7]==p[8] and p[7]>p[9]
    assert m._table[b'cue']=={7:2.,8:2.}


def test_support_protects_repeated_context_under_pressure():
    m=ConsolidatingMemory(4096,'support',16)
    for _ in range(30): m.observe_transition(b'protected-context',7)
    for i in range(256): m.observe_transition(bytes([i])*16,8)
    assert m._table[b'rotected-context'][7]==30
