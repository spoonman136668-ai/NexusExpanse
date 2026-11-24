import copy
import pytest
import torch
from benchmarks.longitudinal_dev import fixture, cache, evaluate, shared_init, ORDER, REGIMES
from nexus_expanse.persistent_memory import BoundedEpisodicByteMemory

@pytest.mark.parametrize('cued',[True,False])
def test_fixture_critical_targets(cued):
    data,positions=fixture(0,cued)
    other,other_positions=fixture(1,cued)
    assert positions==other_positions and len(positions)==6
    for p in positions:
        assert data[p-2:p]==b'c=' and data[p]!=other[p]
        assert (data[p-12:p]!=other[p-12:p])==cued
    assert len(ORDER)==2*REGIMES

@pytest.mark.parametrize('learn',[False,True])
def test_cached_shared_matches_core(learn):
    torch.set_num_threads(1)
    m=shared_init(42,'monarch',True,None); n=copy.deepcopy(m)
    data,positions=fixture(0,True)
    result=evaluate(m,None,data,positions,cache(m,data),learn)
    ref=n.evaluate_streaming(data,seq_len=32,use_episodic=False,use_fast_weights=True,reset_fast_state=False,online_hebbian_update=learn)
    assert result['bpb']==pytest.approx(ref['bpb'],abs=1e-12)
    assert torch.equal(m.fast_weights.w_fast,n.fast_weights.w_fast)
    assert m.fast_weights.updates_applied==n.fast_weights.updates_applied

def test_frozen_memory_probe_has_no_writes():
    m=shared_init(42,'monarch',True,None); data,positions=fixture(0,True)
    mem=BoundedEpisodicByteMemory(256,32); f=cache(m,data)
    evaluate(m,mem,data,positions,f,True)
    before=mem.snapshot(max_bytes=40000)
    evaluate(m,mem,data,positions,f)
    assert before==mem.snapshot(max_bytes=40000)
