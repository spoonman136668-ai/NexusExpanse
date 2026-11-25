import torch
from nexus_expanse.jepa import KuramotoJEPA,encoder,latent,stream_latent,vicreg,state_hash
from benchmarks.gate9_jepa_corpus import manifest
from benchmarks.gate9_jepa_dev import pack,noncollapse

def test_manifest():
    m=manifest(); assert len(m['splits']['train'])==72 and len(m['splits']['eval'])==48
    assert m['sha256']=='4f6e68bc9cdb32ff45dd2026e512c12fcdc45fd90c29df6e96f7aab53678e9b5'
def test_collapse_and_covariance():
    v,c=vicreg(torch.zeros(16,64)); assert v>.98 and c==0
    assert noncollapse(torch.zeros(16,64).numpy())['effective_rank']==0
    v,c=vicreg(torch.eye(16)*4); assert v<.01 and c>0
def test_jepa_gradients_ema():
    torch.manual_seed(4); m=KuramotoJEPA(); x,l=pack([b'abc',b'def']); y,k=pack([b'ghi',b'jkl']); old=state_hash(m.target); loss=m.loss(x,l,y,k); loss.backward(); assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.context.parameters()); assert all(p.grad is None for p in m.target.parameters()); assert state_hash(m.target)==old
    with torch.no_grad(): next(m.context.parameters()).add_(1)
    m.update_target(); assert state_hash(m.target)!=old
    assert all(not a.ffn.enable_skip for a in m.context.layers)
def test_stream_latent():
    m=encoder().eval(); x,l=pack([b'raw bytes']);
    with torch.inference_mode(): assert torch.equal(latent(m,x,l)[0],stream_latent(m,b'raw bytes'))
def test_causal_padding():
    m=encoder().eval(); x,l=pack([b'abc']); y=torch.cat([x,torch.tensor([[99,100]])],1)
    with torch.inference_mode(): assert torch.equal(latent(m,x,l),latent(m,y,l))
