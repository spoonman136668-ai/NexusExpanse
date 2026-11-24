"""Bit-exact next-logit comparison against the unmodified full-prefix reference."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.ablation_dev import CONFIG, torch
from nexus_expanse import NexusExpanseByteCore


@torch.inference_mode()
def compare(seed,kind,gated,limit,positions):
    torch.manual_seed(seed)
    config=dict(CONFIG,dim=64,max_seq_len=limit)
    m=NexusExpanseByteCore(**config,ffn_kind=kind).eval()
    raw=torch.randint(0,256,(1,limit+2))
    if gated:
        m.calibrate_gates([(raw[:,:limit],raw[:,1:limit+1])])
    cache=m.new_kv_cache()
    worst=0.
    mismatch=0
    for i,b in enumerate(raw[0,:max(positions)].tolist(),1):
        actual=cache.append(b)
        if i in positions:
            reference=m.forward_neural(raw[:,max(0,i-limit):i])[0][0,-1]
            err=(actual-reference).abs().max().item()
            worst=max(worst,err)
            mismatch+=int(not torch.equal(actual,reference))
    return worst,mismatch


def sweep():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fails=[]
    worst=0.
    comparisons=0
    for seed in range(100):
        for kind in ('dense','monarch'):
            for gated in (False,True):
                for limit,positions in [(32,(1,2,7,13,31,32,33)),(256,(1,2,7,13,32))]:
                    err,n=compare(seed,kind,gated,limit,positions)
                    worst=max(worst,err)
                    comparisons+=len(positions)
                    if n:
                        fails.append(dict(seed=seed,kind=kind,gated=gated,limit=limit,max_diff=err,mismatches=n))
    return dict(seeds=100,comparisons=comparisons,max_diff=worst,fails=fails)


if __name__=='__main__':
    result=sweep()
    print(json.dumps(result))
    sys.exit(bool(result['fails']))
