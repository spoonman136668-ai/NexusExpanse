"""DEV-only longitudinal retention, with observable cues and ambiguous conflicts.

Predeclare eight regimes, forward then reverse acquisition, no writes during
probes; the decisive target-byte loss is separate from aggregate byte loss.
The cached frozen backbone is equivalent to core streaming for shared weights.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, platform, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.fast_retention_dev import BUDGET, EPOCHS, TRAIN, CONFIG, ROOT, batches, shared_init, torch, F
from nexus_expanse import snapshot_fast_weights, restore_fast_weights, backbone_identity
from nexus_expanse.persistent_memory import BoundedEpisodicByteMemory

SEEDS = (42, 43, 44)
REGIMES = 8
ORDER = tuple(range(REGIMES)) + tuple(reversed(range(REGIMES)))
LIMIT = .02
MIN_GAIN = .05
CONTEXT_CAP = 256
ARMS = ('shared_decay', 'shared_no_decay', 'frozen', 'memory32')

def fixture(regime, cued):
    if not 0 <= regime < REGIMES: raise ValueError('invalid regime')
    cue = bytes([65+regime]) if cued else b'X'
    record = b'r'+cue+b';a=1;b=2;c='+bytes([52+regime])+b';\n'
    return record*6, tuple(i+record.index(b'c=')+2 for i in range(0,len(record)*6,len(record)))

@torch.inference_mode()
def cache(model, data):
    model.eval()
    logits, hidden = [], []
    for i in range(0,len(data)-1,32):
        x=torch.tensor(list(data[i:min(i+32,len(data)-1)]))[None]
        l,h,_,_=model.forward_neural(x)
        logits.extend(l[0].double().unbind()); hidden.extend(h[0].unbind())
    return tuple(zip(logits,hidden))

@torch.inference_mode()
def evaluate(model, mem, data, targets, features, learn=False):
    total=critical=0.; targetset=set(targets)
    if len(features)!=len(data)-1 or not targets or not targetset <= set(range(1,len(data))):
        raise ValueError('invalid transitions')
    for i,(logits,h) in enumerate(features):
        if mem is not None:
            ctx=data[max(0,i+1-mem.max_order):i+1]
            p,_=mem.predict_probs(ctx)
        else:
            p=F.softmax(logits+model.fast_weights.forward_delta(h).double(),dim=-1)
        target=data[i+1]; loss=-math.log(max(float(p[target]),1e-12))
        total+=loss
        if i+1 in targetset: critical+=loss
        if learn:
            if mem is not None: mem.observe_transition(ctx,target)
            else: model.fast_weights.hebbian_update(h,p,target,loss,surprise_threshold=2.)
    return dict(bpb=total/((len(data)-1)*math.log(2)),target_bpb=critical/(len(targets)*math.log(2)))

def train(seed):
    m=shared_init(seed,'monarch',True,None)
    opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.01); bs=batches(TRAIN)
    for _ in range(EPOCHS):
        m.train()
        for x,y in bs:
            opt.zero_grad(set_to_none=True)
            logits,_,aux,_=m.forward_neural(x,targets=y)
            loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1))+.1*aux
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
        m.calibrate_gates(bs)
    m.eval(); return m

def run(seed):
    base=train(seed); results={}
    for cued in (True,False):
        data=[fixture(i,cued) for i in range(REGIMES)]
        features=[cache(base,d) for d,_ in data]; arms={}
        for arm in ARMS:
            m=copy.deepcopy(base); mem=None
            if arm=='shared_no_decay': m.fast_weights.decay=1.
            if arm=='memory32':
                mem=BoundedEpisodicByteMemory(CONTEXT_CAP,32); mem.ingest_stream(TRAIN)
            def score(i,learn=False): return evaluate(m,mem,*data[i],features[i],learn)
            initial=[score(i) for i in range(REGIMES)]; acquired={}; history=[]
            for step,i in enumerate(ORDER):
                before=score(i); online=score(i,learn=arm!='frozen'); after=score(i)
                if i not in acquired: acquired[i]=after
                # Frozen probes of all previously acquired regimes at each checkpoint.
                probes={str(j):score(j) for j in sorted(acquired)}
                if mem is None:
                    w=m.fast_weights.w_fast
                    payload=snapshot_fast_weights(m,max_bytes=BUDGET)
                    capacity=dict(bytes=len(payload),l2=float(w.norm()),max_abs=float(w.abs().max()),
                                  clamp_fraction=float((w.abs()>=2).float().mean()),updates=m.fast_weights.updates_applied)
                else:
                    payload=mem.snapshot(max_bytes=BUDGET)
                    capacity=dict(bytes=len(payload),contexts=mem.memory_entry_count(),updates=mem.num_observations)
                history.append(dict(regime=i,step=step,before=before,online=online,after=after,probes=probes,capacity=capacity))
            final=history[-1]['probes']; metric='target_bpb'
            gains=[initial[i][metric]-acquired[i][metric] for i in range(REGIMES)]
            drift=[final[str(i)][metric]-acquired[i][metric] for i in range(REGIMES)]
            worst=max(p[metric]-acquired[int(j)][metric] for r in history for j,p in r['probes'].items())
            if mem is None:
                restored=copy.deepcopy(base); restored.fast_weights.decay=m.fast_weights.decay
                restore_fast_weights(restored,payload,max_bytes=BUDGET)
                equality=all(evaluate(restored,None,*data[i],features[i])==final[str(i)] for i in range(REGIMES))
            else:
                restored=BoundedEpisodicByteMemory.restore(payload,max_bytes=BUDGET,max_contexts=CONTEXT_CAP)
                equality=all(evaluate(m,restored,*data[i],features[i])==final[str(i)] for i in range(REGIMES))
            assert equality
            arms[arm]=dict(initial=initial,acquired=acquired,history=history,first_target_gains=gains,final_target_drifts=drift,
                worst_target_drift=worst,mean_final_target_bpb=sum(v[metric] for v in final.values())/REGIMES,
                all_regimes_pass=min(gains)>=MIN_GAIN and worst<=LIMIT,exact_restore_equal=equality)
        results['cued' if cued else 'ambiguous']=arms
    return dict(seed=seed,backbone_sha256=backbone_identity(base),conditions=results)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    if git('status','--porcelain'): raise RuntimeError('clean tree required')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    out=dict(schema='nexus-longitudinal-dev-v1',qualification='DEVELOPMENT_ONLY',scientific_trial_executed=False,
        source_sha=git('rev-parse','HEAD'),source_tree=git('rev-parse','HEAD^{tree}'),python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform(),
        config=CONFIG,epochs=EPOCHS,order=ORDER,budget=BUDGET,memory_context_cap=CONTEXT_CAP,retention_limit=LIMIT,min_gain=MIN_GAIN,
        fixtures={f'{c}:{i}':hashlib.sha256(fixture(i,c)[0]).hexdigest() for c in (True,False) for i in range(REGIMES)},train_sha256=hashlib.sha256(TRAIN).hexdigest(),
        runs=[run(s) for s in SEEDS])
    out['seconds']=time.perf_counter()-start
    with args.output.open('x') as f: json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
if __name__=='__main__': main()
