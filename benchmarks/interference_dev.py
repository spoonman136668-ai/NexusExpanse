"""Known development-only interference decomposition and matched-storage routing."""
from __future__ import annotations
import argparse, copy, hashlib, json, platform, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.fast_retention_dev import A, B, BUDGET, EPOCHS, TRAIN, CONFIG, ROOT, batches, shared_init, torch, F
from nexus_expanse import ContextBankFastWeightAdapter, snapshot_fast_weights, restore_fast_weights, backbone_identity

SEEDS = (42,43,44)
LIMIT = .02
MIN_B_GAIN = .05

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
    kw = dict(seq_len=32, use_episodic=False, use_fast_weights=True, reset_fast_state=False)
    def score(model,data): return model.evaluate_streaming(data, **kw)['bpb']
    def learn(model,data): return model.evaluate_streaming(data, online_hebbian_update=True, **kw)
    arms = {}
    for kind in ('shared','shared_no_decay','bank','bank_no_decay'):
        model = copy.deepcopy(m)
        if kind.startswith('bank'):
            model.fast_weights = ContextBankFastWeightAdapter(m.dim, lr=m.fast_weights.lr, decay=m.fast_weights.decay)
        if kind.endswith('no_decay'): model.fast_weights.decay = 1.
        a0 = score(model,A)
        learn(model,A)
        a1 = score(model,A); b0 = score(model,B)
        payload = snapshot_fast_weights(model,max_bytes=BUDGET)
        frozen = copy.deepcopy(model); frozen.fast_weights.reset_fast_state()
        restore_fast_weights(frozen,payload,max_bytes=BUDGET)
        assert score(frozen,A) == a1
        if kind == 'shared':
            decay_only = copy.deepcopy(model)
            # Exactly the original number of decay operations, with no writes.
            for _ in range(len(B)-1): decay_only.fast_weights.w_fast.mul_(decay_only.fast_weights.decay)
            no_decay_b = copy.deepcopy(model); no_decay_b.fast_weights.decay = 1.
            learn(no_decay_b,B)
            arms['shared_decay_only_after_a'] = dict(a_after_a=a1,a_after_b=score(decay_only,A),b_before=b0,b_after=score(decay_only,B))
            arms['shared_no_decay_after_a'] = dict(a_after_a=a1,a_after_b=score(no_decay_b,A),b_before=b0,b_after=score(no_decay_b,B))
        learn(model,B)
        a2 = score(model,A); b1 = score(model,B)
        arms[kind] = dict(a_before=a0,a_after_a=a1,a_after_b=a2,b_before=b0,b_after=b1,
                          checkpoint_bytes=len(payload),tensor_bytes=model.fast_weights.w_fast.numel()*4,
                          updates=model.fast_weights.updates_applied,exact_restore_equal=True)
    for r in arms.values():
        r['a_drift'] = r['a_after_b']-r['a_after_a']; r['b_gain'] = r['b_before']-r['b_after']
        r['retention_and_learning_pass'] = r['a_drift'] <= LIMIT and r['b_gain'] >= MIN_B_GAIN
    return dict(seed=seed, backbone_sha256=backbone_identity(m), optimizer_steps=EPOCHS*len(bs), arms=arms)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    if git('status','--porcelain'): raise RuntimeError('clean tree required')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    out=dict(schema='nexus-interference-dev-v1',qualification='DEVELOPMENT_ONLY',scientific_trial_executed=False,
             source_sha=git('rev-parse','HEAD'),source_tree=git('rev-parse','HEAD^{tree}'),python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform(),
             config=CONFIG,epochs=EPOCHS,budget=BUDGET,banks=2,context_order=16,retention_limit=LIMIT,min_b_gain=MIN_B_GAIN,
             fixtures={k:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in [('train',TRAIN),('a',A),('b',B)]},runs=[run(s) for s in SEEDS])
    out['seconds']=time.perf_counter()-start
    with args.output.open('x') as f: json.dump(out,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
if __name__=='__main__': main()
