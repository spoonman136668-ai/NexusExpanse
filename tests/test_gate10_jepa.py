import math
from collections import Counter
import torch
from nexus_expanse.jepa_domain import DomainKuramotoJEPA, domain_vicreg, momentum
from nexus_expanse.jepa import latent, stream_latent
from benchmarks.gate10_jepa_corpus import corpus
from benchmarks.gate10_jepa_dev import balanced_batches,pack,retrieval
import numpy as np


def test_corpus_counterfactuals_and_balanced_steps():
    s=corpus(); assert len(s['train'])==360 and len(s['eval'])==120
    for d in ('math','science'):
        rows=[r for r in s['eval'] if r['domain']==d]; assert len(rows)==40
        for g in range(10):
            a=[r for r in rows if r['group']==g]; assert len({tuple(sorted(Counter(bytes.fromhex(r['ctx'])).items())) for r in a})==1
            assert set(r['label'] for r in a)==set(range(4 if d=='math' else 2))
    bs=list(balanced_batches(s['train'],42,1)); assert len(bs)==30
    assert all(Counter(r['domain'] for r in b)==dict(math=4,code=4,science=4) for b in bs)
    assert len({r['ctx'] for b in bs for r in b})==360


def test_domain_penalty_detects_domain_collapse():
    z=torch.cat([torch.ones(4,4)*i*10 for i in range(3)])
    d=torch.arange(3).repeat_interleave(4); v,c=domain_vicreg(z,d)
    assert float(v)>.98 and float(c)==0


def test_ema_schedule_and_isolation():
    torch.set_num_threads(1); torch.manual_seed(1)
    assert math.isclose(momentum(0),.90) and math.isclose(momentum(299),.99)
    assert all(momentum(i)<momentum(i+1) for i in range(299))
    m=DomainKuramotoJEPA(); x,l=pack([b'abc',b'bcd',b'cde',b'def',b'efg',b'fgh']); d=torch.arange(3).repeat_interleave(2)
    m.loss(x,l,x,l,d).backward(); assert all(p.grad is None for p in m.target.parameters())
    assert any(p.grad is not None for p in m.context.parameters())
    p,q=next(m.context.parameters()),next(m.target.parameters()); old=q.clone(); p.data.add_(1); m.update_target(0)
    assert torch.allclose(q,old*.9+p*.1)
    m.eval(); assert torch.equal(latent(m.context,*pack([b'abc']))[0],stream_latent(m.context,b'abc'))


def test_retrieval_same_domain_candidate_count():
    rows=corpus()['eval']; x=np.eye(120); r=retrieval(x,x,rows)
    assert r['top1']==r['mrr']==1 and all(len(v['ranks'])==40 for v in r['per_domain'].values())
