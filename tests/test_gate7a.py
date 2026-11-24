import torch
import pytest
from nexus_expanse.holographic_memory_v2 import HolographicRelationalMemoryV2


def test_window_shift_and_exact_state():
    m=HolographicRelationalMemoryV2(None)
    for _ in range(16): m._record((1,2))
    m.memory[17]=0.1234567890123456
    before=m.memory.clone()
    for _ in range(16): m._record((3,4))
    assert m._content([1,2,3,4,5])==[1,2,5]
    assert torch.equal(before,m.memory)
    m.outcomes={7,8}
    payload=m.snapshot()
    assert len(payload)<=40000
    n=HolographicRelationalMemoryV2(None)
    n.restore(payload)
    assert n.snapshot()==payload
    assert n.doc_freq==m.doc_freq
    with pytest.raises(ValueError): n.restore(payload[:-1])


def test_all_bytes_window_is_bounded():
    m=HolographicRelationalMemoryV2(None)
    for _ in range(17): m._record(tuple(range(256)))
    m.outcomes=set(range(256))
    assert len(m.snapshot())<=40000
    assert len(m.window)==16
