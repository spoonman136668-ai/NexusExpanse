#!/usr/bin/env python3
"""Frozen Gate 4A reused fixture: Hopfield VSA vs all matched controls."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.gate4a_dev import SEEDS, cold, run, torch
from nexus_expanse.hopfield_vsa import HopfieldVSAMemory

BASE='8e2b6a52edf5f65d6c6d33de950934d1c4dd5786'
OLD='evidence/gate7a-holographic-v2-dev-001.json'

def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()

def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if sys.argv[1]=='--cold':
        cold(Path(sys.argv[2]),candidate_cls=HopfieldVSAMemory); return
    if git('status','--porcelain'): raise RuntimeError('clean preregistered tree required')
    previous_bytes=subprocess.check_output(['git','show',f'{BASE}:{OLD}'],cwd=ROOT)
    previous=json.loads(previous_bytes); old={r['seed']:r for r in previous['runs']}
    probe=HopfieldVSAMemory(None); runs=[]
    for seed in SEEDS:
        candidate=run(seed,candidate_cls=HopfieldVSAMemory,cold_script=__file__)
        for prior in (old[seed]['v1'],old[seed]['v2']):
            for key in ('fixture_digest','backbone','corrupted'):
                if candidate[key]!=prior[key]: raise ValueError('historical control custody mismatch: '+key)
            for key in candidate['arms']:
                if key!='latent_wta' and candidate['arms'][key]!=prior['arms'][key]:
                    raise ValueError('control mismatch: '+key)
        c=candidate['arms']['latent_wta']; s=candidate['arms']['sparse_support']; u=c['scores']['unseen']
        margin=s['scores']['unseen']['bpb']-u['bpb']
        quality=u['accuracy']>=.75 and margin>=3.
        retention=c['worst_drift']<=.02
        engineering=candidate['cold_equal'] and c['max_bytes']<=40000 and probe.live_codebook_bytes<512*1024
        runs.append(dict(seed=seed,candidate=candidate,gate6a_v1=old[seed]['v1']['arms']['latent_wta'],gate7a_v2=old[seed]['v2']['arms']['latent_wta'],matched_controls=True,unseen_gain_vs_sparse=margin,quality=quality,retention=retention,engineering=engineering))
        print(f'seed={seed} unseen={u} drift={c["worst_drift"]:.6f} max_bytes={c["max_bytes"]}',flush=True)
    e=all(r['engineering'] for r in runs); q=all(r['quality'] for r in runs); t=all(r['retention'] for r in runs)
    status='PASS' if e and q and t else 'MIXED' if e and (q or t) else 'FAIL'
    result=dict(schema='nexus-gate8a-hopfield-vsa-dev-v1',qualification='REUSED_FIXTURE_DEV',source_sha=git('rev-parse','HEAD'),source_tree=git('rev-parse','HEAD^{tree}'),torch=str(torch.__version__),historical_source=BASE,historical_evidence_sha256=hashlib.sha256(previous_bytes).hexdigest(),live_codebook_bytes=probe.live_codebook_bytes,live_slot_bytes=probe.live_slot_bytes,status=status,runs=runs)
    with open(sys.argv[1],'x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(status,flush=True)
if __name__=='__main__': main()
