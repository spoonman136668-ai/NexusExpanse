"""Noisy composition with a change of byte surface and fresh-process return."""
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.relational_dev import AssociationArm, ARMS, SEEDS, train, torch
from benchmarks.ablation_dev import CONFIG
from nexus_expanse import NexusExpanseByteCore, backbone_identity
from nexus_expanse.latent_association import LatentAssociation


def fixture(seed):
    rng = random.Random(seed)
    maps = [rng.sample(range(16), 16) for _ in range(4)]
    def make(x, a, b, skin):
        y = maps[a][x] if b is None else maps[b][maps[a][x]]
        parts = [bytes([65+x]), bytes([97+a])]
        if b is not None:
            parts.append(bytes([97+b]))
        prefix = (b' ' if skin == 0 else b'::').join(parts) + (b'?' if skin == 0 else b' =>')
        canonical = bytes([254,65+x,97+a,0 if b is None else 97+b,253])
        return prefix, canonical, 65+y
    singles = [make(x,a,None,0) for x in range(16) for a in range(4)]
    pairs = [make(x,a,b,1) for x in range(16) for a in range(4) for b in range(4)]
    seen, novel = [], []
    for i,row in enumerate(pairs):
        x, ab = divmod(i,16)
        a,b = divmod(ab,4)
        (seen if (x+a+b)%4==0 else novel).append(row)
    phases = []
    for group, repeats in [(singles,4),(seen,2)]:
        phase = []
        for _ in range(repeats):
            batch = group.copy()
            rng.shuffle(batch)
            for p,c,y in batch:
                observed = 65+(y-65+rng.randrange(1,16))%16 if rng.random()<.1 else y
                phase.append((p,c,y,observed))
        phases.append(phase)
    shifted = [make(x,a,None,1) for x in range(16) for a in range(4)]
    assert set(r[1] for r in seen).isdisjoint(r[1] for r in novel)
    return singles, seen, novel, shifted, phases


def digest(probs):
    h = hashlib.sha256()
    for p in probs:
        h.update(p.contiguous().numpy().tobytes())
    return h.hexdigest()


def cold(directory, candidate_cls=LatentAssociation):
    model = NexusExpanseByteCore(**CONFIG).eval()
    model.load_state_dict(torch.load(directory/'model', weights_only=True))
    arm = candidate_cls(model)
    arm.restore((directory/'state').read_bytes())
    queries = json.loads((directory/'queries').read_text())
    print(digest(arm.predict(*arm.encode(bytes.fromhex(p))) for p in queries))


def run(seed, candidate_cls=LatentAssociation, cold_script=__file__):
    singles, seen, novel, shifted, phases = fixture(seed)
    model = train(seed)
    candidate = candidate_cls(model)
    rows = singles + seen + novel + shifted
    features = {p:candidate.encode(p) for p,_,_ in rows}
    arms = {}
    for name in (*ARMS, 'latent_wta'):
        control = None if name == 'latent_wta' else AssociationArm(name)
        def key(p,c):
            return c if name in ('local_chain','symbolic_chain') else p
        def predict(row):
            p,c,_ = row
            codes,prior = features[p]
            if control is None:
                return candidate.predict(codes,prior)
            if name == 'sparse_support':
                return control.memory.predict_probs(p)[0]
            if name == 'lookup':
                counts = control.table.get(p)
                if not counts:
                    return prior
                out = .01*prior
                for byte,n in counts.items():
                    out[byte] += .99*n/counts.total()
                return out
            return prior if name == 'frozen' else control.predict(key(p,c),prior)
        def score(row):
            probs = predict(row)
            return -math.log2(max(float(probs[row[2]]),1e-300)), int(probs.argmax())==row[2]
        anchor, worst = {}, {}
        losses, sizes = [], []
        for phase in phases:
            for i,(p,c,y,observed) in enumerate(phase,1):
                row = (p,c,y)
                losses.append(-math.log2(max(float(predict(row)[observed]),1e-300)))
                if control is None:
                    candidate.observe(features[p][0], observed)
                    sizes.append(len(candidate.snapshot()))
                elif name in ('symbolic_chain','local_chain','sparse_support','lookup'):
                    control.observe(key(p,c)+bytes([observed]))
                    sizes.append(control.byte_size())
                if row not in anchor:
                    anchor[row] = score(row)[0]
                    worst[row] = 0.
                if i%16==0:
                    for probe in anchor:
                        worst[probe]=max(worst[probe],score(probe)[0]-anchor[probe])
        metrics = {}
        for label,group in [('cold_return',singles),('shifted_primitive',shifted),('seen',seen),('unseen',novel)]:
            scores=[score(r) for r in group]
            metrics[label]=dict(bpb=sum(v for v,_ in scores)/len(group),
                                accuracy=sum(v for _,v in scores)/len(group))
        arms[name]=dict(scores=metrics,max_bytes=max(sizes,default=0),worst_drift=max(worst.values()),
                        per_association_drift={p.hex():v for (p,_,_),v in worst.items()},
                        online_bpb=sum(losses)/len(losses))
        assert arms[name]['max_bytes']<=40000
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)
        torch.save(model.state_dict(),path/'model')
        (path/'state').write_bytes(candidate.snapshot())
        (path/'queries').write_text(json.dumps([p.hex() for p,_,_ in rows]))
        expected=digest(candidate.predict(*features[p]) for p,_,_ in rows)
        actual=subprocess.check_output([sys.executable,cold_script,'--cold',tmp],text=True).strip()
        assert actual==expected
    result=arms['latent_wta']
    margin=min(arms[k]['scores']['unseen']['bpb'] for k in ARMS)-result['scores']['unseen']['bpb']
    quality=margin>=.05
    retention=result['worst_drift']<=.02
    status='PASS' if quality and retention else 'MIXED' if quality or retention else 'FAIL'
    return dict(seed=seed,backbone=backbone_identity(model),cold_digest=actual,cold_equal=True,
                fixture_digest=hashlib.sha256(repr((phases,novel)).encode()).hexdigest(),
                corrupted=sum(y!=o for phase in phases for _,_,y,o in phase),
                arms=arms,margin=margin,status=status)


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if sys.argv[1]=='--cold':
        cold(Path(sys.argv[2]))
    else:
        assert not subprocess.check_output(['git','status','--porcelain'])
        runs=[run(seed) for seed in SEEDS]
        status='PASS' if all(r['status']=='PASS' for r in runs) else 'FAIL' if all(r['status']=='FAIL' for r in runs) else 'MIXED'
        with open(sys.argv[1],'x') as out:
            json.dump(dict(qualification='DEV',source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                           torch=str(torch.__version__),status=status,runs=runs),out,allow_nan=False,indent=2)
            out.write('\n')
