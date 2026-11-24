"""Gate 7B completion: immutable epoch checkpoints and independent per-seed receipts.

No model or benchmark policy changes. Run training shards separately; run timed
arms serially on an idle host. Artifacts belong outside the tracked working tree.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks import gate7b_dev as frozen
import torch
import torch.nn.functional as F

SEEDS = (42, 43, 44)
BASE = '8e2b6a52edf5f65d6c6d33de950934d1c4dd5786'
PROTOCOL = 'b7cfd5aa19ceea4203aaeeaedc5cc495ed1fa8e2'
ORIGINAL = 'evidence/gate7b-kuramoto-skip-dev-001.json'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity():
    if git('status', '--porcelain'):
        raise RuntimeError('dirty source')
    for path in ('nexus_expanse/kuramoto.py', 'nexus_expanse/kuramoto_skip.py',
                 'benchmarks/gate7b_dev.py', 'docs/GATE7B_PREREG.md'):
        if git('rev-parse', f'HEAD:{path}') != git('rev-parse', f'{PROTOCOL}:{path}'):
            raise RuntimeError('frozen source drift: ' + path)
    return dict(source_sha=git('rev-parse', 'HEAD'), tree=git('rev-parse', 'HEAD^{tree}'),
                protocol_sha=PROTOCOL, manifest_sha256=digest(frozen.MANIFEST),
                torch=str(torch.__version__), platform=platform.platform(),
                cpu=platform.processor(), threads=torch.get_num_threads())


def write_json(path, obj):
    # Exclusive create; interrupted writes retain a .partial file, never a valid receipt.
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix(path.suffix + '.partial')
    with temp.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.link(temp, path); temp.unlink()


def train_shard(seed, spans, directory, ident, limit=12):
    """Exactly frozen training order, with immutable epoch boundaries and RNG custody."""
    torch.manual_seed(seed)
    m = frozen.CausalKuramotoMonarchByteCore(dim=64, enable_skip=True)
    batches = [(torch.tensor(list(b[:-1]))[None], torch.tensor(list(b[1:]))[None]) for b in spans if len(b)>1]
    opt = torch.optim.AdamW(m.parameters(), lr=.003, weight_decay=.01)
    prefix = f'seed-{seed}-epoch-'
    files = sorted(directory.glob(prefix + '[0-9][0-9].pt'))
    epoch = 0
    if files:
        expected = [directory / f'{prefix}{i:02}.pt' for i in range(1, len(files)+1)]
        if files != expected or len(files)>12:
            raise ValueError('checkpoint lineage gap')
        last = None
        for i, path in enumerate(files, 1):
            meta = json.loads(path.with_suffix('.json').read_text())
            if meta != dict(identity=ident, seed=seed, epoch=i, sha256=digest(path), parent=last):
                raise ValueError('checkpoint custody mismatch')
            last = meta['sha256']
        ck = torch.load(files[-1], weights_only=True)
        if ck['epoch'] != len(files): raise ValueError('epoch mismatch')
        m.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        torch.set_rng_state(ck['rng']); epoch = ck['epoch']
    else:
        last = None
    for i in range(epoch, min(epoch + limit, 12)):
        start = time.perf_counter()
        m.train()
        for x,y in batches:
            opt.zero_grad(set_to_none=True); l,a=m.forward_train(x,y)
            (F.cross_entropy(l.reshape(-1,256),y.flatten())+.1*a).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
        m.calibrate_gates(batches)
        path = directory / f'{prefix}{i+1:02}.pt'
        with path.open('xb') as f:
            torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(), rng=torch.get_rng_state(), epoch=i+1), f)
            f.flush(); os.fsync(f.fileno())
        sha = digest(path)
        write_json(path.with_suffix('.json'), dict(identity=ident, seed=seed, epoch=i+1, sha256=sha, parent=last))
        last = sha
        print(f'seed={seed} epoch={i+1}/12 seconds={time.perf_counter()-start:.3f} sha256={sha}', flush=True)
    return m.eval(), min(epoch + limit, 12)


def arm(directory, seed, policy, mode, ident):
    path = directory / f'seed-{seed}-epoch-12.pt'
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta['identity'] != ident or meta['seed'] != seed or meta['epoch'] != 12 or meta['sha256'] != digest(path):
        raise ValueError('final checkpoint mismatch')
    m = frozen.CausalKuramotoMonarchByteCore(dim=64, enable_skip=True).eval()
    m.load_state_dict(torch.load(path, weights_only=True)['model'])
    data = frozen.read_split()
    result = frozen.evaluate(m, data['train'], data['dev'], policy, mode=='online', seed)
    result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return result


def evaluate_seed(directory, seed, ident):
    data = frozen.read_split()
    _, epoch = train_shard(seed, data['train'], directory, ident, limit=0)
    if epoch != 12: raise ValueError('seed training incomplete')
    arms = []
    for policy in frozen.POLICIES:
        for mode in ('static', 'online'):
            p = directory / f'seed-{seed}-{policy}-{mode}.json'
            if not p.exists():
                result = json.loads(subprocess.check_output([sys.executable, __file__, '--arm', str(seed), policy, mode, '--directory', str(directory)], text=True))
                write_json(p, dict(identity=ident, seed=seed, result=result))
            receipt = json.loads(p.read_text())
            if receipt['identity'] != ident or receipt['seed'] != seed or (receipt['result']['policy'], receipt['result']['mode']) != (policy, mode):
                raise ValueError('arm identity mismatch')
            arms.append(receipt['result'])
    if len({a['state_hash'] for a in arms}) != 1 or not all(a['transitions']==2040 for a in arms):
        raise ValueError('unmatched arms')
    write_json(directory / f'seed-{seed}.json', dict(identity=ident, seed=seed, arms=arms))


def merge(directory, ident, out):
    runs=[]; checks=[]; scaling=[]
    for seed in SEEDS:
        receipt=json.loads((directory/f'seed-{seed}.json').read_text())
        if receipt['identity'] != ident or receipt['seed'] != seed: raise ValueError('seed identity mismatch')
        arms=receipt['arms']; runs.append(dict(seed=seed,arms=arms))
        if len(arms)!=8 or len({a['state_hash'] for a in arms})!=1 or any(a['transitions']!=2040 for a in arms): raise ValueError('invalid arms')
        for mode in ('static','online'):
            a={v['policy']:v for v in arms if v['mode']==mode}; c=a[frozen.POLICIES[1]]; base=a[frozen.POLICIES[0]]; random=a[frozen.POLICIES[3]]
            checks.append(dict(seed=seed,mode=mode,quality=c['bpb']<=base['bpb']+.05 and c['bpb']<=random['bpb']-.02,skip=.4<=c['skip_rate']<=.6,speed=c['ms_per_byte']<=base['ms_per_byte']/1.1))
    prior=json.loads(subprocess.check_output(['git','show',f'{BASE}:{ORIGINAL}'],cwd=ROOT,text=True))
    scaling.extend(prior['scaling'])
    for dim in (128,256):
        receipt=json.loads((directory/f'scale-kv-{dim}-512.json').read_text())
        if receipt['identity']!=ident: raise ValueError('scale identity mismatch')
        scaling.append(receipt['result'])
    exact=json.loads((directory/'parity.json').read_text())
    if exact['identity']!=ident or exact['result']['max_prefix_diff']!=0 or exact['result']['max_stream_vs_batch_diff']!=0: raise ValueError('parity failed')
    q=all(c['quality'] for c in checks); s=all(c['speed'] and c['skip'] for c in checks)
    status='PASS' if q and s else 'MIXED' if q or s else 'FAIL'
    write_json(out,dict(schema='nexus-gate7b-dev-v1',qualification='REUSED_MANIFEST_DEV',**ident,status=status,parity=exact['result'],runs=runs,checks=checks,scaling=scaling,prior_incomplete_source=BASE,scaling_note='First six cells historical; last two newly measured or explicitly timed out.'))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--directory',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True); g.add_argument('--seed',type=int,choices=SEEDS); g.add_argument('--train-seed',type=int,choices=SEEDS); g.add_argument('--arm',nargs=3); g.add_argument('--scale-kv',type=int,choices=(128,256)); g.add_argument('--parity',action='store_true'); g.add_argument('--merge',type=Path)
    p.add_argument('--epochs-per-call',type=int,default=12); a=p.parse_args()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    directory=a.directory.resolve(); directory.mkdir(parents=True,exist_ok=True); ident=identity()
    if a.epochs_per_call not in range(1,13): p.error('epochs-per-call must be 1..12')
    if a.arm:
        seed,policy,mode=a.arm; print(json.dumps(arm(directory,int(seed),policy,mode,ident),allow_nan=False)); return
    if a.train_seed is not None or a.seed is not None:
        seed=a.train_seed if a.train_seed is not None else a.seed
        if (directory/f'seed-{seed}.json').exists(): raise FileExistsError('seed already completed')
        data=frozen.read_split(); _,epoch=train_shard(seed,data['train'],directory,ident,a.epochs_per_call)
        if a.seed is not None and epoch==12: evaluate_seed(directory,seed,ident)
    elif a.scale_kv:
        dim=a.scale_kv; path=directory/f'scale-kv-{dim}-512.json'
        if path.exists(): raise FileExistsError(path)
        start=time.perf_counter()
        try:
            result=json.loads(subprocess.check_output([sys.executable,str(ROOT/'benchmarks/gate7b_dev.py'),'--scale','kv',str(dim),'512'],text=True,timeout=240))
        except subprocess.TimeoutExpired:
            result=dict(kind='kv',dim=dim,length=512,status='TIMEOUT',elapsed_seconds=time.perf_counter()-start,budget_seconds=240)
        write_json(path,dict(identity=ident,result=result))
    elif a.parity:
        write_json(directory/'parity.json',dict(identity=ident,result=frozen.parity()))
    else: merge(directory,ident,a.merge)

if __name__=='__main__': main()
