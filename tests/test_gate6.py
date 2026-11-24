
from benchmarks.ablation_dev import CONFIG
from benchmarks.kuramoto_parity_dev import sweep
from nexus_expanse import NexusExpanseByteCore
from nexus_expanse.holographic_memory import HolographicRelationalMemory
import torch


def test_phase_parity_and_long_stream():
    threads=torch.get_num_threads()
    try:
        result=sweep(2)
        assert result['fails']==[] and result['stream_max_diff']==result['prefix_max_diff']==0.
    finally:
        torch.set_num_threads(threads)


def test_holographic_query_is_read_only_and_superposition_restores():
    model=NexusExpanseByteCore(**CONFIG).eval()
    arm=HolographicRelationalMemory(model)
    arm.observe((1,2),1)
    arm.observe((1,2),3)
    before=arm.snapshot()
    assert arm.bundle.norm()>0
    prior=torch.full((256,),1/256,dtype=torch.float64)
    predict=arm.predict((1,2),prior)
    assert arm.snapshot()==before and abs(predict.sum().item()-1)<1e-12
    restored=HolographicRelationalMemory(model)
    restored.restore(before)
    assert torch.equal(predict,restored.predict((1,2),prior))
