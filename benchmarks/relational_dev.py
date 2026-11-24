"""Preregistered DEV experiment on composition of two observable byte relations."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.longitudinal_dev import BUDGET, SEEDS, train, torch
from nexus_expanse import backbone_identity
from nexus_expanse.consolidation import ConsolidatingMemory

N = 16
ARMS = ('frozen', 'sparse_support', 'lookup', 'symbolic_chain', 'local_chain')


def fixture(seed):
    rng = random.Random(seed)
    mappings = [rng.sample(range(N), N) for _ in range(4)]

    def record(x, a, b):
        y = mappings[a][x]
        if b is not None:
            y = mappings[b][y]
        return bytes([254, 65+x, 97+a, 0 if b is None else 97+b, 253, 65+y])

    singles = [record(x, a, None) for x in range(N) for a in range(4)]
    pairs = [record(x, a, b) for x in range(N) for a in range(4) for b in range(4)]
    seen = [r for r in pairs if (r[1]-65+r[2]-97+r[3]-97) % 4 == 0]
    novel = [r for r in pairs if r not in seen]
    rng.shuffle(singles)
    rng.shuffle(seen)
    assert len(novel) == 192 and len(seen) == len(singles) == 64
    assert {r[:-1] for r in seen}.isdisjoint(r[:-1] for r in novel)
    return singles, seen, novel


class AssociationArm:
    def __init__(self, name):
        self.name = name
        self.table = {}
        self.weights = torch.zeros(4, N, N) if name == 'local_chain' else None
        self.memory = ConsolidatingMemory(BUDGET, 'sparse_support') if name == 'sparse_support' else None

    def predict(self, prefix, prior):
        x, a, b = prefix[1]-65, prefix[2]-97, prefix[3]
        if self.memory is not None:
            return self.memory.predict_probs(prefix)[0]
        if self.weights is not None:
            activation = self.weights[a, x].softmax(0)
            if b:
                activation = activation @ self.weights[b-97].softmax(1)
            probs = .01 * prior
            probs[65:65+N] += .99 * activation.double()
            return probs
        if self.name == 'symbolic_chain':
            counts = self.table.get(bytes([x, a]))
            if counts and b:
                counts = self.table.get(bytes([counts.most_common(1)[0][0]-65, b-97]))
        else:
            counts = self.table.get(prefix)
        if not counts:
            return prior
        probs = .01 * prior
        for byte, count in counts.items():
            probs[byte] += .99 * count / counts.total()
        return probs

    def observe(self, row):
        x, a, b, y = row[1]-65, row[2]-97, row[3], row[-1]
        if self.memory is not None:
            for i in range(1, len(row)):
                self.memory.observe_transition(row[:i], row[i])
        elif self.weights is not None:
            # A cue selects the local output matrix. A composed cue supplies
            # soft presynaptic activity from the previously learned first edge.
            pre = torch.zeros(N)
            pre[x] = 1
            if b:
                pre = self.weights[a, x].softmax(0)
                a = b-97
            post = torch.zeros(N)
            post[y-65] = 1
            self.weights[a] += 3 * pre[:, None] * (post - self.weights[a].softmax(1))
        elif self.name == 'lookup' or (self.name == 'symbolic_chain' and not b):
            key = row[:-1] if self.name == 'lookup' else bytes([x, a])
            self.table.setdefault(key, Counter())[y] += 1

    def byte_size(self):
        if self.memory is not None:
            return len(self.memory.snapshot(max_bytes=BUDGET))
        if self.weights is not None:
            return len(self.weights.numpy().tobytes())
        return len(json.dumps({k.hex(): dict(v) for k,v in self.table.items()},
                              separators=(',', ':')).encode()) if self.table else 0


@torch.inference_mode()
def priors(model, rows):
    prefixes = [r[:-1] for r in rows]
    prior = {}
    for i in range(0,len(rows),32):
        group = prefixes[i:i+32]
        logits = model.forward_neural(torch.tensor([list(p) for p in group]))[0]
        prior.update(zip(group, logits[:, -1].double().softmax(-1), strict=True))
    return prior


def probe(arm, row, neural):
    p = arm.predict(row[:-1], neural[row[:-1]])
    return -math.log2(max(float(p[row[-1]]), 1e-12)), int(p.argmax()) == row[-1]


def run(seed):
    singles, seen, novel = fixture(seed)
    model = train(seed)
    neural = priors(model, singles + seen + novel)
    result = {}
    for name in ARMS:
        arm = AssociationArm(name)
        acquired, worst = {}, {}
        sizes = []
        online = []
        for group in (singles, seen):
            for start in range(0, len(group), 8):
                batch = group[start:start+8]
                for row in batch:
                    online.append(probe(arm, row, neural)[0])
                    arm.observe(row)
                    acquired[row] = probe(arm, row, neural)[0]
                    worst[row] = 0.
                for row, initial in acquired.items():
                    worst[row] = max(worst[row], probe(arm, row, neural)[0] - initial)
                sizes.append(arm.byte_size())
                assert sizes[-1] <= BUDGET
        scores = {}
        for group_name, group in [('primitive', singles), ('repeated_pair', seen), ('unseen_pair', novel)]:
            measurements = [probe(arm, row, neural) for row in group]
            scores[group_name] = dict(first_target_bpb=sum(s[0] for s in measurements)/len(group),
                                      first_target_accuracy=sum(s[1] for s in measurements)/len(group))
        result[name] = dict(scores=scores, max_bytes=max(sizes), prequential_target_bpb=sum(online)/len(online),
                            worst_drift=max(worst.values()),
                            per_association_drift={r[:-1].hex(): v for r,v in worst.items()})
    def loss(name):
        return result[name]['scores']['unseen_pair']['first_target_bpb']
    margin = min(loss(name) for name in ARMS if name != 'local_chain') - loss('local_chain')
    return dict(seed=seed, fixture_sha256=hashlib.sha256(b''.join(singles+seen+novel)).hexdigest(),
                backbone_sha256=backbone_identity(model), arms=result, strongest_control_margin=margin,
                passed=margin >= .05 and result['local_chain']['worst_drift'] <= .02)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    if git('status', '--porcelain'):
        raise RuntimeError('clean preregistered tree required')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = dict(schema='nexus-relational-dev-v1', qualification='DEVELOPMENT_ONLY',
                  source_sha=git('rev-parse','HEAD'), source_tree=git('rev-parse','HEAD^{tree}'),
                  platform=platform.platform(), torch=str(torch.__version__), runs=[run(s) for s in SEEDS])
    result['passed'] = all(r['passed'] for r in result['runs'])
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')
