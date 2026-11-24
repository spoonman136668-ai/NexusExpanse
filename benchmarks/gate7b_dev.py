#!/usr/bin/env python3
"""Matched Gate 7B DEV; isolated scaling RSS; recurrent state never drops bytes."""
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from benchmarks.gate4b_dev import MANIFEST,read_split
from nexus_expanse.arbitration import ReliabilityArbitrator
from nexus_expanse.consolidation import ConsolidatingMemory
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.kuramoto_skip import CausalKuramotoMonarchByteCore
from nexus_expanse.patching import CompactBigram

POLICIES=('kuramoto_always','kuramoto_causal_skip','kuramoto_causal_skip_patch','kuramoto_random_skip')


def parity():
    sample=torch.tensor(list(b'def kuramoto_step(z, x): return z\n'))[None]
    pmax=smax=0.; comparisons=0
    for seed in range(50):
        torch.manual_seed(seed)
        m=CausalKuramotoMonarchByteCore(dim=32).eval(); m.random_seed=seed
        m.calibrate_gates([(sample[:,:16],sample[:,1:17])])
        for policy in ('always','causal','random'):
            m.policy=policy
            with torch.inference_mode():
                logits=m.forward_neural(sample)[0]
                for n in (1,7,13,25):
                    pref=m.forward_neural(sample[:,:n])[0]
                    pmax=max(pmax,float((pref-logits[:,:n]).abs().max()))
                z=m.init_stream_state()
                for i,b in enumerate(sample[0].tolist()):
                    p,z,_=m.stream_step(b,z,i)
                    smax=max(smax,float((p-logits[0,i].double().softmax(0)).abs().max())); comparisons+=1
    return dict(seeds=50,policies=3,stream_comparisons=comparisons,max_prefix_diff=pmax,max_stream_vs_batch_diff=smax)


def train(seed,spans):
    torch.manual_seed(seed)
    m=CausalKuramotoMonarchByteCore(dim=64,enable_skip=True)
    batches=[(torch.tensor(list(b[:-1]))[None],torch.tensor(list(b[1:]))[None]) for b in spans if len(b)>1]
    opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.01)
    for _ in range(12):
        m.train()
        for x,y in batches:
            opt.zero_grad(set_to_none=True); l,a=m.forward_train(x,y)
            (F.cross_entropy(l.reshape(-1,256),y.flatten())+.1*a).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
        m.calibrate_gates(batches)
    return m.eval()


def state_hash(m):
    h=hashlib.sha256()
    for key,v in sorted(m.state_dict().items()):
        h.update(key.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate(m,train_spans,dev,policy,online,seed):
    mem=ConsolidatingMemory(40000,'sparse_support')
    counts=np.zeros((256,256),dtype='<u4')
    for b in train_spans:
        for i in range(1,len(b)):
            mem.observe_transition(b[max(0,i-8):i],b[i]); counts[b[i-1],b[i]]+=1
    compact=CompactBigram.from_dense(counts)
    totals=compact.sum(axis=1).astype(np.float64)+256.
    arb=ReliabilityArbitrator(); m.random_seed=seed
    m.policy='always' if policy==POLICIES[0] else 'random' if policy==POLICIES[3] else 'causal'
    before=state_hash(m); bits=0.; count=deep=possible=readouts=0; per_span=[]
    start=time.perf_counter()
    with torch.inference_mode():
        for b in dev:
            z=m.init_stream_state(); prev=None; block_bits=0.
            for i,(obs,target) in enumerate(zip(b,b[1:])):
                wake=policy!=POLICIES[2] or prev is None or (float(compact[prev,obs])+1.)/totals[prev]<2.**-4
                p,z,decisions=m.stream_step(obs,z,i,readout=wake)
                if not wake: p=torch.from_numpy((compact[obs].astype(np.float64)+1.)/totals[obs])
                prev=obs; readouts+=int(wake); deep+=sum(decisions); possible+=len(decisions)
                ctx=b[max(0,i+1-8):i+1]; pm,confidence=mem.predict_probs(ctx); w=arb.weight(confidence)
                pn=float(p[target]); mp=float(pm[target]); loss=-math.log2(max((1-w)*pn+w*mp,1e-300))
                bits+=loss; block_bits+=loss; count+=1
                if online: arb.observe(confidence,pn,mp); mem.observe_transition(ctx,target)
            per_span.append(dict(transitions=len(b)-1,bpb=block_bits/(len(b)-1)))
    elapsed=time.perf_counter()-start
    assert before==state_hash(m)
    return dict(policy=policy,mode='online' if online else 'static',bpb=bits/count,transitions=count,ms_per_byte=elapsed*1000/count,skip_rate=1-deep/possible,readout_fraction=readouts/count,compact_bytes=len(compact.snapshot()) if policy==POLICIES[2] else 0,state_hash=before,per_span=per_span)


def scale(kind,dim,length):
    torch.manual_seed(42)
    if kind=='kuramoto':
        m=CausalKuramotoMonarchByteCore(dim=dim,enable_skip=False,max_seq_len=128).eval()
        def fresh(): return m.init_stream_state()
        def step(b,z,i): return m.stream_step(b,z,i)[1]
    else:
        m=NexusExpanseByteCore(dim=dim,n_layers=2,n_heads=4,ffn_mult=2,enable_skip=True,dropout=0.,max_seq_len=128).eval()
        for l in m.layers: l.ffn.tau_eval.fill_(-float('inf'))
        def fresh(): return m.new_kv_cache()
        def step(b,z,i): z.append(b).double().softmax(0); return z
    seq=np.random.default_rng(42).integers(0,256,size=length).tolist(); times=[]
    with torch.inference_mode():
        z=fresh()
        for i,b in enumerate(seq[:32]): z=step(b,z,i)
        for _ in range(3):
            z=fresh(); start=time.perf_counter()
            for i,b in enumerate(seq): z=step(b,z,i)
            times.append((time.perf_counter()-start)*1000/length)
    # Linux ru_maxrss is KiB. This worker is separately launched for each arm.
    return dict(kind=kind,dim=dim,length=length,ms_per_byte=statistics.median(times),repeats=times,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,window=128)


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if sys.argv[1]=='--scale':
        print(json.dumps(scale(sys.argv[2],int(sys.argv[3]),int(sys.argv[4])))); return
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
    exact=parity(); assert exact['max_prefix_diff']==exact['max_stream_vs_batch_diff']==0.
    data=read_split(); runs=[]
    for seed in (42,43,44):
        m=train(seed,data['train'])
        arms=[evaluate(m,data['train'],data['dev'],p,o,seed) for p in POLICIES for o in (False,True)]
        assert len({a['state_hash'] for a in arms})==1 and all(a['transitions']==2040 for a in arms)
        runs.append(dict(seed=seed,arms=arms))
    scaling=[json.loads(subprocess.check_output([sys.executable,__file__,'--scale',kind,str(dim),str(t)],text=True)) for dim in (128,256) for t in (128,512) for kind in ('kuramoto','kv')]
    checks=[]
    for r in runs:
        for mode in ('static','online'):
            a={v['policy']:v for v in r['arms'] if v['mode']==mode}; c=a[POLICIES[1]]; base=a[POLICIES[0]]; random=a[POLICIES[3]]
            checks.append(dict(seed=r['seed'],mode=mode,quality=c['bpb']<=base['bpb']+.05 and c['bpb']<=random['bpb']-.02,skip=.4<=c['skip_rate']<=.6,speed=c['ms_per_byte']<=base['ms_per_byte']/1.1))
    q=all(c['quality'] for c in checks); s=all(c['speed'] and c['skip'] for c in checks)
    status='PASS' if q and s else 'MIXED' if q or s else 'FAIL'
    with open(sys.argv[1],'x') as out:
        json.dump(dict(schema='nexus-gate7b-dev-v1',qualification='REUSED_MANIFEST_DEV',source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),torch=str(torch.__version__),parity=exact,status=status,checks=checks,runs=runs,scaling=scaling),out,indent=2,allow_nan=False); out.write('\n')
    print(exact,status)
if __name__=='__main__': main()
