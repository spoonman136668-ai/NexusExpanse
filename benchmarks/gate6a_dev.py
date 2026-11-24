"""Preregistered bipolar binding versus paired Gate 5 relational controls."""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.gate4a_dev import cold, run, SEEDS, torch
from benchmarks.gate5a_dev import compare as run_controls
from nexus_expanse.holographic_memory import HolographicRelationalMemory


def compare(seed):
    controls=run_controls(seed)
    result=run(seed,HolographicRelationalMemory,__file__)
    assert (controls['backbone'],controls['fixture_digest'])==(result['backbone'],result['fixture_digest'])
    arms=result['arms']
    arms['holographic']=arms.pop('latent_wta')
    arms['aligned_v2']=controls['arms']['aligned_v2']
    candidate=arms['holographic']
    score=candidate['scores']['unseen']
    checks=dict(accuracy=score['accuracy']>=.90,
                sparse_gain=arms['sparse_support']['scores']['unseen']['bpb']-score['bpb']>=.05,
                retention=candidate['worst_drift']<=.02)
    result.pop('margin')
    result.update(checks=checks,status='PASS' if all(checks.values()) else
                  'MIXED' if checks['accuracy'] and checks['sparse_gain'] else 'FAIL')
    return result


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if sys.argv[1]=='--cold':
        cold(Path(sys.argv[2]),HolographicRelationalMemory)
    else:
        assert not subprocess.check_output(['git','status','--porcelain'])
        with open(sys.argv[1],'x') as out:
            json.dump(dict(qualification='DEV',source_sha=subprocess.check_output(
                ['git','rev-parse','HEAD'],text=True).strip(),runs=[compare(s) for s in SEEDS]),
                out,allow_nan=False,indent=2)
            out.write('\n')
