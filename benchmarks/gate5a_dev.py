"""Paired replay of the frozen Gate 4A diagnostic."""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.gate4a_dev import cold, run, SEEDS, torch
from nexus_expanse.latent_association import AlignedLatentAssociation


def compare(seed):
    baseline = run(seed)
    aligned = run(seed, AlignedLatentAssociation, __file__)
    assert baseline['fixture_digest'] == aligned['fixture_digest']
    assert baseline['backbone'] == aligned['backbone']
    arms = aligned['arms']
    v2 = arms.pop('latent_wta')
    arms['aligned_v2'] = v2
    arms['gate4_v1'] = baseline['arms']['latent_wta']
    unseen = v2['scores']['unseen']
    checks = dict(accuracy=unseen['accuracy'] >= .90,
                  sparse_margin=arms['sparse_support']['scores']['unseen']['bpb']-unseen['bpb'] >= .05,
                  symbolic_margin=arms['symbolic_chain']['scores']['unseen']['bpb']-unseen['bpb'] >= .05,
                  retention=v2['worst_drift'] <= .02)
    del aligned['margin']
    aligned.update(checks=checks, status='PASS' if all(checks.values()) else
                   'MIXED' if checks['accuracy'] and checks['sparse_margin'] else 'FAIL')
    return aligned


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if sys.argv[1] == '--cold':
        cold(Path(sys.argv[2]), AlignedLatentAssociation)
    else:
        assert not subprocess.check_output(['git','status','--porcelain'])
        with open(sys.argv[1],'x') as out:
            runs=[compare(s) for s in SEEDS]
            json.dump(dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                           qualification='DEV', runs=runs),out,indent=2,allow_nan=False)
            out.write('\n')
