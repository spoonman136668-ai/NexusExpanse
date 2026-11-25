"""Counterfactual tensor-group replacements on newly trained incompatible donors."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks import checkpoint_transfer_study as study
from nexus_expanse.checkpoint_transport import load_skill
from nexus_expanse.donor_assimilation import forbid_donor_forward
import torch

GROUPS=('embeddings','attention','norms','ffn','readout')


def group(key):
    if key.startswith(('emb.','pos_emb.')):
        return 'embeddings'
    if key.startswith('readout.'):
        return 'readout'
    if 'norm.' in key:
        return 'norms'
    if '.attn.' in key:
        return 'attention'
    if '.fc1.' in key or '.fc2.' in key:
        return 'ffn'
    raise ValueError('unclassified tensor '+key)


def run(dest):
    method=json.loads((study.OUT/'checkpoint-transfer-screen.json').read_text())['ranking'][0][-1]
    splits,_=study.corpus()
    runs=[]
    for seed in range(801,806):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            subprocess.run([sys.executable,study.__file__,'producer',str(path),str(seed),'rule'],check=True)
            p={l:torch.load(path/(l+'.pt'),weights_only=True) for l in ('A','B')}
            refs={l:torch.load(path/(l+'.reference'),weights_only=True)['test'] for l in p}
            assert torch.equal(p['A']['byte_to_row'],p['B']['byte_to_row'])
            with forbid_donor_forward() as guard:
                for label,other in [('A','B'),('B','A')]:
                    tables=study.rules(seed)
                    _,test=study.eval_sets('rule',seed,splits,tables[label=='B'])
                    _,opposite=study.eval_sets('rule',seed,splits,tables[other=='B'])
                    scores={}
                    for replacement in ('none',*GROUPS,'all'):
                        state={k:(p[other]['weights'][k] if replacement=='all' or group(k)==replacement else v)
                               for k,v in p[label]['weights'].items()}
                        mixed=path/'mixed.pt'
                        torch.save(dict(weights=state,byte_to_row=p[label]['byte_to_row']),mixed)
                        model=load_skill(mixed,method)
                        own,_=study.evaluate(model,test,refs[label])
                        swapped,_=study.evaluate(model,opposite,refs[other])
                        scores[replacement]={'own':own,'opposite':swapped}
                    runs.append(dict(seed=seed,source=label,replaced_from=other,scores=scores,
                        donor_test=study.score_logits(refs[label],test[1]),forbidden_calls=guard.donor_forward_calls))
    study.atomic(dest,dict(method=method,source=study.git('rev-parse','HEAD').decode().strip(),runs=runs))


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    run(Path(sys.argv[1]))
