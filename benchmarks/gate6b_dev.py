"""Matched real-corpus phase recurrence and cached attention CPU DEV trial."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.gate5b_dev import STRONG
from benchmarks.ablation_dev import shared_init
from benchmarks.gate4b_dev import MANIFEST, measure, prepare, read_split, torch, train_one, worker
from benchmarks.kuramoto_parity_dev import sweep
from nexus_expanse import NexusExpanseByteCore
from nexus_expanse.kuramoto import KuramotoMonarchByteCore

POLICIES={'dense':('always',),'dense_matched':('always',),'monarch':('always','patch'),
          'kuramoto':('recurrent','patch')}


def build(kind,config):
    if kind=='kuramoto':
        return KuramotoMonarchByteCore(**{k:config[k] for k in ('dim','n_layers','ffn_mult','max_seq_len')})
    net=NexusExpanseByteCore(**config,ffn_kind='dense' if kind=='dense_matched' else kind)
    if kind=='dense_matched':
        # Exactly 3,264 FFN parameters/layer at d=64; both dense maps are active.
        for layer in net.layers:
            layer.ffn.fc1=torch.nn.Linear(64,25,bias=False)
            layer.ffn.fc2=torch.nn.Linear(25,64)
    return net


def initialize(seed,kind,config):
    ref=shared_init(seed,'monarch',False,None,config)
    if kind in ('dense','monarch'):
        return shared_init(seed,kind,True,ref,config)
    torch.manual_seed(seed)
    model=build(kind,config)
    # Match all tensors with the same semantics and shape. The second byte
    # drive has a distinct role and has its own initialization.
    state=model.state_dict()
    for name,value in ref.state_dict().items():
        if name in state and state[name].shape==value.shape:
            state[name]=value
    model.load_state_dict(state)
    return model


def run():
    assert not subprocess.check_output(['git','status','--porcelain'])
    parity=sweep()
    assert not parity['fails']
    data=read_split()
    runs=[]
    with tempfile.TemporaryDirectory() as temp:
        directory=Path(temp)
        compact=prepare(directory,data['train'],compact=True)
        memory=[worker(directory,'memory','memory_only',mode,__file__) for mode in ('static','online')]
        for seed in (42,43,44):
            by_kind={}
            for kind,policies in POLICIES.items():
                print(f'training seed {seed} {kind}',flush=True)
                row=train_one(directory,seed,kind,data['train'],STRONG,12,True,initialize)
                row['arms']=[worker(directory,kind,policy,mode,__file__) for policy in policies
                             for mode in ('static','online')]
                assert all(a['trained_hash']==row['state_hash'] for a in row['arms'])
                by_kind[kind]=row
            assert len({by_kind[k]['trainable_parameters'] for k in ('kuramoto','monarch','dense_matched')})==1
            for arm in by_kind['kuramoto']['arms']:
                policy='patch' if arm['policy']=='patch' else 'always'
                ref=next(a for a in by_kind['monarch']['arms'] if a['policy']==policy and a['mode']==arm['mode'])
                arm['speedup']=ref['ms_per_byte']/arm['ms_per_byte']
                arm['bpb_delta']=arm['bpb']-ref['bpb']
                arm['rss_ratio']=arm['peak_rss_mb']/ref['peak_rss_mb']
                quality=arm['bpb_delta']<=.05
                speed=arm['speedup']>=2
                resource_ok=arm['rss_ratio']<=1.10
                arm['status']=('PASS' if quality and speed and resource_ok else
                               'MIXED' if resource_ok and (quality or speed) else 'FAIL')
            runs.extend(by_kind.values())
    return dict(qualification='REUSED_MANIFEST_DEV',source_sha=subprocess.check_output(
        ['git','rev-parse','HEAD'],text=True).strip(),torch=str(torch.__version__),config=STRONG,
        epochs=12,manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        compact=compact,parity=parity,memory_only=memory,runs=runs)


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if sys.argv[1]=='--arm':
        kind=sys.argv[3]
        print(json.dumps(measure(Path(sys.argv[2]),kind,sys.argv[4],sys.argv[5]=='online',
                                 STRONG,True,build,kind=='kuramoto')))
    else:
        with open(sys.argv[1],'x') as out:
            json.dump(run(),out,indent=2,allow_nan=False)
            out.write('\n')
