"""Deterministic development check for existing gated-backbone prefix instability."""
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from benchmarks.ablation_dev import CONFIG, TRAIN
from nexus_expanse.core import NexusExpanseByteCore


def run():
    torch.set_num_threads(1)
    fails=[]
    for seed in range(100):
        torch.manual_seed(seed)
        m=NexusExpanseByteCore(**CONFIG,ffn_kind='monarch',enable_skip=True)
        x=torch.tensor(list(TRAIN[:32]))[None]
        m.calibrate_gates([(x[:,:-1],x[:,1:])]); m.eval()
        with torch.inference_mode():
            a=m.forward_neural(x)[0]; b=m.forward_neural(x[:,:13])[0]
        if not torch.allclose(a[:,:13],b,atol=1e-5):
            fails.append({'seed':seed,'max_diff':float((a[:,:13]-b).abs().max())})
    return {'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'torch':str(torch.__version__),'seeds':100,'fails':fails}

if __name__=='__main__':
    result=run()
    print(json.dumps(result))
    sys.exit(bool(result['fails']))
