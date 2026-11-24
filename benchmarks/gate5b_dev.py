"""Frozen matched cached-policy trial on the Gate 4B manifest."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.gate4b_dev import CONFIG, MANIFEST, measure, prepare, read_split, torch, train_one, worker
from benchmarks.kv_parity_dev import sweep

STRONG = dict(CONFIG,dim=64,max_seq_len=256)


def run():
    assert not subprocess.check_output(['git','status','--porcelain'])
    parity=sweep()
    data=read_split()
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)
        compact=prepare(path,data['train'],compact=True)
        mem=[worker(path,'memory','memory_only',mode,__file__) for mode in ('static','online')]
        runs=[]
        for seed in (42,43,44):
            for kind in ('dense','monarch'):
                row=train_one(path,seed,kind,data['train'],STRONG,12,full_spans=True)
                row['arms']=[worker(path,kind,policy,mode,__file__) for policy in ('always','skip','patch')
                             for mode in ('static','online')]
                assert all(a['trained_hash']==row['state_hash'] for a in row['arms'])
                assert all(a['patch_predictor_serialized_bytes']+a['patch_cursor_bytes']<25000 for a in row['arms'])
                runs.append(row)
    return dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                qualification='REUSED_MANIFEST_DEV',torch=str(torch.__version__),config=STRONG,epochs=12,
                manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                kv_parity=parity,compact=compact,memory_only=mem,runs=runs)


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if sys.argv[1]=='--arm':
        print(json.dumps(measure(Path(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5]=='online',STRONG,True)))
    else:
        with open(sys.argv[1],'x') as out:
            json.dump(run(),out,allow_nan=False,indent=2)
            out.write('\n')
