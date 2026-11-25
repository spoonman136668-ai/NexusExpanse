import numpy as np
import pytest
import torch
from nexus_expanse.hopfield_vsa import HopfieldVSAMemory


def test_compact_codebooks_and_hopfield_cleanup():
    m=HopfieldVSAMemory(None)
    assert m.live_codebook_bytes==131072 and m.live_slot_bytes==36864
    assert set(np.unique(m.anchor_book))=={-1,1}
    for _ in range(3):
        assert m._write(1,9,2,2)
        assert m._write(2,10,3,2)
    assert m._write(1,9,4,1)
    m.roles={9,10}; m.outcomes={1,2,3,4}; m.records=16
    for _ in range(16): m.window.append(frozenset({1,9,240}))
    prior=torch.ones(256,dtype=torch.float64)/256
    before=m.snapshot()
    p=m.predict((1,251,9,252,10,253),prior)
    assert p.argmax().item()==3 and p[3]>.9 and abs(float(p.sum())-1)<1e-12
    assert m.snapshot()==before
    n=HopfieldVSAMemory(None); n.restore(before)
    assert n.snapshot()==before and torch.equal(n.predict((1,251,9,252,10,253),prior),p)
    with pytest.raises(ValueError): n.restore(before[:-1])


def test_role_discovery_and_shift_preserves_keys():
    m=HopfieldVSAMemory(None)
    rows=[((11,200,21,201),12),((12,200,22,201),11),((11,200,21,201),12),((12,200,22,201),11)]
    for _ in range(8):
        for prefix,target in rows: m.observe(prefix,target)
    prior=torch.ones(256,dtype=torch.float64)/256
    assert m.roles=={21,22}
    assert m.predict((11,250,21,251,22,252),prior).argmax().item()==11
    keys=m.keys[:m.used].copy()
    for _ in range(16): m.observe((11,250,21,251,22,252),11)
    assert np.array_equal(keys,m.keys[:m.used])
    assert m.doc_freq[200]==0 and m.doc_freq[250]==16
    assert m.predict((11,250,21,251,22,252),prior).argmax().item()==11


def test_full_slot_bank_bounded_and_no_overwrite():
    m=HopfieldVSAMemory(None)
    for r in range(128): assert m._write(7,r,8,2)
    m.records=128; m.outcomes={8}; m.roles=set(range(128)); m.window.extend([frozenset({7})]*16)
    before=m.snapshot(); assert len(before)<=40000
    assert not m._write(7,128,9,2) and m.snapshot()==before
    n=HopfieldVSAMemory(None); n.restore(before); assert n.snapshot()==before
    with pytest.raises(ValueError): m.observe((256,),1)


def test_unknown_or_uncertain_hop_abstains():
    m=HopfieldVSAMemory(None); prior=torch.ones(256,dtype=torch.float64)/256
    m.outcomes={1,2,3,4}; m.roles={9,10}
    m._write(1,9,2,2); m._write(1,9,4,2); m._write(2,10,3,2)
    assert torch.equal(m.predict((1,9,10),prior),prior)
    assert torch.equal(m.predict((1,255),prior),prior)
