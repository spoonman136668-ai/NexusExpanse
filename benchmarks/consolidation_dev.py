"""DEV-only consolidation, same v0.7 acquisition/probes and strict byte ceiling."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, platform, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.longitudinal_dev import (SEEDS, REGIMES, ORDER, LIMIT, MIN_GAIN, CONTEXT_CAP,
    BUDGET, EPOCHS, TRAIN, CONFIG, ROOT, torch, F, fixture, cache, train, evaluate)
from nexus_expanse import backbone_identity, snapshot_fast_weights, restore_fast_weights
from nexus_expanse.persistent_memory import BoundedEpisodicByteMemory
from nexus_expanse.consolidation import ConsolidatingMemory
ARMS=('frozen','shared_no_decay','memory32','lru','support','sparse_support')

@torch.inference_mode()
def first_target(model,mem,data,targets,features):
    i=targets[0]-1; logits,h=features[i]
    if mem is not None: p,_=mem.predict_probs(data[max(0,i+1-mem.max_order):i+1])
    else: p=F.softmax(logits+model.fast_weights.forward_delta(h).double(),dim=-1)
    return -math.log2(max(float(p[data[i+1]]),1e-12))

class FrozenView:
    """Ephemeral cache scoped to one immutable memory state."""
    def __init__(self, mem):
        self.mem=mem; self.max_order=mem.max_order; self.cache={}
    def predict_probs(self, context):
        if context not in self.cache: self.cache[context]=self.mem.predict_probs(context)
        return self.cache[context]

def run(seed, memory_reference=None):
    base=train(seed); results={}
    for cued in (True,False):
        data=[fixture(i,cued) for i in range(REGIMES)]; features=[cache(base,d) for d,_ in data]; arms={}
        for arm in ARMS:
            if memory_reference is not None and arm not in ('frozen','shared_no_decay'):
                arms[arm]=dict(memory_reference['conditions']['cued' if cued else 'ambiguous'][arm],deterministic_reuse_from_seed=memory_reference['seed'])
                continue
            m=copy.deepcopy(base); mem=None
            if arm=='shared_no_decay': m.fast_weights.decay=1.
            elif arm=='memory32': mem=BoundedEpisodicByteMemory(CONTEXT_CAP,32)
            elif arm in ConsolidatingMemory.POLICIES: mem=ConsolidatingMemory(BUDGET,arm,32)
            if mem is not None: mem.ingest_stream(TRAIN)
            view=FrozenView(mem) if mem is not None else None
            def score(i,learn=False): return evaluate(m,mem if learn else view,*data[i],features[i],learn)
            def first(i): return first_target(m,view,*data[i],features[i])
            initial=[score(i) for i in range(REGIMES)]; acquired={}; history=[]
            for step,i in enumerate(ORDER):
                before=score(i); online=score(i,learn=arm!='frozen')
                if view is not None: view.cache.clear()
                after=score(i)
                if i not in acquired: acquired[i]=after
                probes={str(j):score(j) for j in sorted(acquired)}
                if mem is None:
                    payload=snapshot_fast_weights(m,max_bytes=BUDGET); contexts=None
                else:
                    payload=mem.snapshot(max_bytes=BUDGET); contexts=mem.memory_entry_count()
                history.append(dict(step=step,regime=i,before=before,online=online,after=after,probes=probes,
                    first_target_probes={str(j):first(j) for j in sorted(acquired)},bytes=len(payload),contexts=contexts))
            final=history[-1]['probes']; metric='target_bpb'
            gains=[initial[i][metric]-acquired[i][metric] for i in range(REGIMES)]
            worst=max(p[metric]-acquired[int(j)][metric] for row in history for j,p in row['probes'].items())
            if mem is None:
                restored=copy.deepcopy(base); restored.fast_weights.decay=m.fast_weights.decay
                restore_fast_weights(restored,payload,max_bytes=BUDGET)
                equal=all(evaluate(restored,None,*data[i],features[i])==final[str(i)] for i in range(REGIMES))
            else:
                restored=type(mem).restore(payload,max_bytes=BUDGET,max_contexts=CONTEXT_CAP if arm=='memory32' else BUDGET)
                equal=all(evaluate(m,restored,*data[i],features[i])==final[str(i)] for i in range(REGIMES))
            assert equal
            arms[arm]=dict(initial=initial,acquired=acquired,history=history,first_target_gains=gains,
                worst_target_drift=worst,min_first_gain=min(gains),mean_final_target_bpb=sum(v[metric] for v in final.values())/REGIMES,
                mean_final_bpb=sum(v['bpb'] for v in final.values())/REGIMES,
                mean_final_first_target_bpb=sum(history[-1]['first_target_probes'].values())/REGIMES,
                max_bytes=max(row['bytes'] for row in history),all_regimes_pass=min(gains)>=MIN_GAIN and worst<=LIMIT,exact_restore_equal=equal)
        results['cued' if cued else 'ambiguous']=arms
    return dict(seed=seed,backbone_sha256=backbone_identity(base),conditions=results)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    def git(*a): return subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
    if git('status','--porcelain'): raise RuntimeError('clean tree required')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.perf_counter()
    first=run(SEEDS[0]); runs=[first]+[run(s,first) for s in SEEDS[1:]]
    out=dict(schema='nexus-consolidation-dev-v1',qualification='DEVELOPMENT_ONLY',scientific_trial_executed=False,
        source_sha=git('rev-parse','HEAD'),source_tree=git('rev-parse','HEAD^{tree}'),python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform(),
        config=CONFIG,epochs=EPOCHS,order=ORDER,budget=BUDGET,retention_limit=LIMIT,min_gain=MIN_GAIN,arms=ARMS,
        fixtures={f'{c}:{i}':hashlib.sha256(fixture(i,c)[0]).hexdigest() for c in (True,False) for i in range(REGIMES)},train_sha256=hashlib.sha256(TRAIN).hexdigest(),runs=runs, memory_only_seed_independent=True)
    out['seconds']=time.perf_counter()-start
    with args.output.open('x') as f: json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
if __name__=='__main__': main()
