#!/usr/bin/env python3
"""Frozen Gate 4A fixture, V1/V2 matched controls; no new holdout."""
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.gate4a_dev import SEEDS, cold, run, torch
from nexus_expanse.holographic_memory import HolographicRelationalMemory
from nexus_expanse.holographic_memory_v2 import HolographicRelationalMemoryV2


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if sys.argv[1] in ('--cold', '--cold-v1'):
        cold(Path(sys.argv[2]), candidate_cls=HolographicRelationalMemoryV2 if sys.argv[1]=='--cold' else HolographicRelationalMemory)
        return
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
    # V1 uses its original cold subprocess entry point. Deterministic train is rechecked.
    runs=[]
    for seed in SEEDS:
        v1=run(seed,candidate_cls=HolographicRelationalMemory,cold_script=str(ROOT/'benchmarks/gate6a_dev.py'))
        v2=run(seed,candidate_cls=HolographicRelationalMemoryV2,cold_script=__file__)
        for key in ('fixture_digest','backbone','corrupted'):
            assert v1[key]==v2[key], key
        for key in v2['arms']:
            if key != 'latent_wta':
                assert v1['arms'][key]==v2['arms'][key], key
        c=v2['arms']['latent_wta']; s=v2['arms']['sparse_support']
        quality=c['scores']['unseen']['accuracy']>=.75 and s['scores']['unseen']['bpb']-c['scores']['unseen']['bpb']>=3.0
        retention=c['worst_drift']<=.02
        engineering=v2['cold_equal'] and c['max_bytes']<=40000
        runs.append(dict(seed=seed,v1=v1,v2=v2,quality=quality,retention=retention,engineering=engineering))
    e=all(r['engineering'] for r in runs); q=all(r['quality'] for r in runs); t=all(r['retention'] for r in runs)
    status='PASS' if e and q and t else 'MIXED' if e and (q or t) else 'FAIL'
    with open(sys.argv[1],'x') as out:
        json.dump(dict(schema='nexus-gate7a-dev-v1',qualification='REUSED_FIXTURE_DEV',source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),torch=str(torch.__version__),status=status,runs=runs),out,indent=2,allow_nan=False)
        out.write('\n')
    print(status)
if __name__=='__main__': main()
