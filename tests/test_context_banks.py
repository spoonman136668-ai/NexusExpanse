import pytest
import torch
from nexus_expanse import (ContextBankFastWeightAdapter, NexusExpanseByteCore,
                           snapshot_fast_weights, restore_fast_weights)

def test_banks_same_budget_and_inactive_rows_unchanged():
    a=ContextBankFastWeightAdapter(16)
    x=b'abc'; y=next(bytes([i]) for i in range(256) if a._slice(bytes([i]))!=a._slice(x))
    h=torch.ones(16); p=torch.full((256,),1/256)
    a.hebbian_update(h,p,4,6,context=x)
    before=a.w_fast.clone(); pred=a.forward_delta(h,context=x).clone()
    a.hebbian_update(h,p,9,6,context=y)
    assert torch.equal(before[a._slice(x)],a.w_fast[a._slice(x)])
    assert torch.equal(pred,a.forward_delta(h,context=x))
    assert a.w_fast.numel()==16*256
    a.reset_fast_state(); assert not torch.count_nonzero(a.w_fast)

@pytest.mark.parametrize('banks,order',[(3,16),(2,0),(2,129)])
def test_bad_routing_config(banks,order):
    with pytest.raises(ValueError): ContextBankFastWeightAdapter(16,banks=banks,context_order=order)

def model(banks=2,order=16):
    torch.manual_seed(42)
    m=NexusExpanseByteCore(n_layers=1,dim=16,n_heads=2,dropout=0.,max_seq_len=16)
    m.fast_weights=ContextBankFastWeightAdapter(16,banks=banks,context_order=order)
    return m

def test_checkpoint_routes_and_restore():
    m=model(); m.evaluate_streaming(b'abcabcabc',seq_len=8,use_episodic=False,online_hebbian_update=True)
    p=snapshot_fast_weights(m,max_bytes=20000); other=model()
    restore_fast_weights(other,p,max_bytes=20000)
    assert snapshot_fast_weights(other,max_bytes=20000)==p
    kw=dict(seq_len=8,use_episodic=False,use_fast_weights=True,reset_fast_state=False)
    assert m.evaluate_streaming(b'abcabcabc',**kw)==other.evaluate_streaming(b'abcabcabc',**kw)
    wrong=model(order=8)
    with pytest.raises(ValueError,match='mismatch'): restore_fast_weights(wrong,p,max_bytes=20000)
    assert not torch.count_nonzero(wrong.fast_weights.w_fast)

def test_causal_route_same_prefix_different_targets():
    a=model(); b=model(); seen=[]
    original=a.fast_weights.forward_delta
    def record(h,*,context=b''):
        seen.append(context); return original(h,context=context)
    a.fast_weights.forward_delta=record
    kw=dict(seq_len=8,use_episodic=False,online_hebbian_update=True)
    a.evaluate_streaming(b'abcdefX',**kw)
    assert seen==[b'abcdef'[:i] for i in range(1,7)]
    assert a.fast_weights._slice(b'abcdef')==b.fast_weights._slice(b'abcdef')


def test_bank_cold_process(tmp_path):
    import subprocess, sys, json, hashlib
    m=model(); data=b'abcabcabcabc'
    kw=dict(seq_len=8,use_episodic=False,use_fast_weights=True,reset_fast_state=False)
    m.evaluate_streaming(data,online_hebbian_update=True,**kw)
    payload=snapshot_fast_weights(m,max_bytes=20000)
    path=tmp_path/'bank.bin'; path.write_bytes(payload)
    script='''import sys, torch, json, hashlib
from nexus_expanse import NexusExpanseByteCore, ContextBankFastWeightAdapter, restore_fast_weights, snapshot_fast_weights
torch.set_num_threads(1); torch.manual_seed(42)
m=NexusExpanseByteCore(n_layers=1,dim=16,n_heads=2,dropout=0.,max_seq_len=16)
m.fast_weights=ContextBankFastWeightAdapter(16)
restore_fast_weights(m,open(sys.argv[1],'rb').read(),max_bytes=20000)
print(json.dumps([hashlib.sha256(snapshot_fast_weights(m,max_bytes=20000)).hexdigest(),m.evaluate_streaming(b'abcabcabcabc',seq_len=8,use_episodic=False,use_fast_weights=True,reset_fast_state=False)]))
'''
    assert json.loads(subprocess.check_output([sys.executable,'-c',script,str(path)],text=True))==[hashlib.sha256(payload).hexdigest(),m.evaluate_streaming(data,**kw)]
