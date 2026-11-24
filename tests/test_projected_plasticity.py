import pytest
import torch
from nexus_expanse import (ProjectedFastWeightAdapter, NexusExpanseByteCore,
                            snapshot_fast_weights, restore_fast_weights)


def test_projection_preserves_protected_logits_and_learns_orthogonal_direction():
    a=ProjectedFastWeightAdapter(4, rank=2)
    h=torch.tensor([1.,0.,0.,0.]); j=torch.tensor([1.,1.,0.,0.])
    p=torch.full((256,),1/256)
    assert a.hebbian_update(h,p,7,6.)
    confident=p.clone(); confident[7]=.9
    assert not a.hebbian_update(h,confident,7,.1)
    before=a.forward_delta(h).clone(); new=a.forward_delta(j).clone()
    assert a.hebbian_update(j,p,9,6.)
    assert torch.allclose(a.forward_delta(h),before,atol=1e-7)
    assert a.forward_delta(j)[9]>new[9]
    a.reset_fast_state(); assert not torch.count_nonzero(a.basis) and not torch.count_nonzero(a.w_fast)


def model(rank=2):
    torch.manual_seed(42)
    m=NexusExpanseByteCore(n_layers=1,dim=16,n_heads=2,dropout=0.,max_seq_len=16)
    m.fast_weights=ProjectedFastWeightAdapter(16,rank=rank)
    return m


def test_checkpoint_protects_basis_and_rejects_wrong_configuration():
    m=model(); a=m.fast_weights; h=torch.eye(16)[0]; p=torch.full((256,),1/256)
    a.hebbian_update(h,p,1,6.); confident=p.clone(); confident[1]=.9; a.hebbian_update(h,confident,1,.1)
    payload=snapshot_fast_weights(m,max_bytes=40000)
    other=model(); restore_fast_weights(other,payload,max_bytes=40000)
    assert snapshot_fast_weights(other,max_bytes=40000)==payload
    for b in (a,other.fast_weights): b.hebbian_update(torch.ones(16),p,2,6.)
    assert torch.equal(a.w_fast,other.fast_weights.w_fast)
    wrong=model(3)
    with pytest.raises(ValueError,match='mismatch'): restore_fast_weights(wrong,payload,max_bytes=40000)
    assert not torch.count_nonzero(wrong.fast_weights.w_fast)
    with pytest.raises(ValueError,match='budget'): snapshot_fast_weights(m,max_bytes=len(payload)-1)


def test_frozen_core_no_writes_and_invalid_basis_rejected():
    m=model(); a=m.fast_weights
    kw=dict(seq_len=8,use_episodic=False,use_fast_weights=True,reset_fast_state=False)
    m.evaluate_streaming(b'abcabcabc',online_hebbian_update=True,**kw)
    before=snapshot_fast_weights(m,max_bytes=40000)
    m.evaluate_streaming(b'abcabcabc',**kw)
    assert snapshot_fast_weights(m,max_bytes=40000)==before
    a.basis.fill_(1.)
    with pytest.raises(ValueError,match='orthonormal'): snapshot_fast_weights(m,max_bytes=40000)

@pytest.mark.parametrize('rank',[0,4,True])
def test_invalid_rank(rank):
    with pytest.raises(ValueError): ProjectedFastWeightAdapter(4,rank=rank)


def test_protection_cold_process(tmp_path):
    import subprocess, sys, json, hashlib
    m=model(); a=m.fast_weights; h=torch.eye(16)[0]; p=torch.full((256,),1/256)
    a.hebbian_update(h,p,1,6.); confident=p.clone(); confident[1]=.9; a.hebbian_update(h,confident,1,.1)
    payload=snapshot_fast_weights(m,max_bytes=40000)
    path=tmp_path/'protected.bin'; path.write_bytes(payload)
    script='''import sys, torch, json, hashlib
from nexus_expanse import NexusExpanseByteCore, ProjectedFastWeightAdapter, restore_fast_weights, snapshot_fast_weights
torch.set_num_threads(1); torch.manual_seed(42)
m=NexusExpanseByteCore(n_layers=1,dim=16,n_heads=2,dropout=0.,max_seq_len=16)
m.fast_weights=ProjectedFastWeightAdapter(16,rank=2)
restore_fast_weights(m,open(sys.argv[1],'rb').read(),max_bytes=40000)
print(json.dumps([hashlib.sha256(snapshot_fast_weights(m,max_bytes=40000)).hexdigest(),m.fast_weights.forward_delta(torch.ones(16)).tolist()]))
'''
    assert json.loads(subprocess.check_output([sys.executable,'-c',script,str(path)],text=True))==[hashlib.sha256(payload).hexdigest(),a.forward_delta(torch.ones(16)).tolist()]
