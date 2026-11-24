"""Known synthetic development-only A/B/A fast-weight retention diagnostic."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.ablation_dev import ROOT, TRAIN, CONFIG, batches, shared_init, torch, F
from nexus_expanse import snapshot_fast_weights, restore_fast_weights, backbone_identity

A = b'a=1;b=2;c=4;\n' * 16
B = b'a=1;b=2;c=9;\n' * 16
BUDGET = 40000
EPOCHS = 12


def run(seed):
    m = shared_init(seed, 'monarch', True, None)
    opt = torch.optim.AdamW(m.parameters(), lr=.003, weight_decay=.01)
    bs = batches(TRAIN)
    for _ in range(EPOCHS):
        m.train()
        for x,y in bs:
            opt.zero_grad(set_to_none=True)
            logits, _, aux, _ = m.forward_neural(x, targets=y)
            loss = F.cross_entropy(logits.reshape(-1,256), y.reshape(-1)) + .1*aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            opt.step()
        m.calibrate_gates(bs)
    baseline = copy.deepcopy(m)
    kw = dict(seq_len=32, use_episodic=False, use_fast_weights=True, reset_fast_state=False)
    def score(model,data): return model.evaluate_streaming(data, **kw)['bpb']
    def learn(model,data): return model.evaluate_streaming(data, online_hebbian_update=True, **kw)
    before = score(m,A)
    learn_a = learn(m,A)
    after_a = score(m,A)
    snap = snapshot_fast_weights(m,max_bytes=BUDGET)
    restored = copy.deepcopy(baseline)
    restore_fast_weights(restored,snap,max_bytes=BUDGET)
    restored_a = score(restored,A)
    assert restored_a == after_a
    reset_a = score(baseline,A)
    b_before = score(restored,B)
    learn_b = learn(restored,B)
    b_after = score(restored,B)
    a_after_b = score(restored,A)
    # Frozen retention control has no writes or decay during B.
    frozen = copy.deepcopy(m)
    score(frozen,B)
    frozen_a = score(frozen,A)
    assert frozen_a == after_a
    return dict(seed=seed, backbone_sha256=backbone_identity(m), optimizer_steps=EPOCHS*len(bs),
                checkpoint_bytes=len(snap), fast_tensor_bytes=m.fast_weights.w_fast.numel()*4,
                updates_a=learn_a['hebbian_updates'], updates_b=learn_b['hebbian_updates']-learn_a['hebbian_updates'],
                a_before=before,a_after_learning=after_a,a_restored=restored_a,a_reset=reset_a,
                b_before=b_before,b_after=b_after,a_after_b=a_after_b,a_frozen_control=frozen_a,
                a_interference_bpb=a_after_b-after_a)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    if git('status','--porcelain'): raise RuntimeError('clean tree required')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    out=dict(schema='nexus-fast-retention-dev-v1',qualification='DEVELOPMENT_ONLY',scientific_trial_executed=False,
             source_sha=git('rev-parse','HEAD'), source_tree=git('rev-parse','HEAD^{tree}'),
             torch=str(torch.__version__),python=platform.python_version(),platform=platform.platform(),device='cpu',threads=1,
             config=CONFIG,epochs=EPOCHS,checkpoint_budget=BUDGET,
             fixtures={k:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in [('train',TRAIN),('a',A),('b',B)]},
             runs=[run(s) for s in (42,43,44)])
    out['seconds']=time.perf_counter()-start
    with args.output.open('x') as f: json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

if __name__=='__main__': main()
