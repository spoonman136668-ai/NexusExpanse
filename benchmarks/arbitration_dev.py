"""Matched development-only arbitration comparison on the existing known fixture."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

from ablation_dev import (ROOT, TRAIN, DEV, CONFIG, batches, shared_init, state_hash,
                          memory_only, EpisodicByteMemory, torch, F)
from nexus_expanse import ReliabilityArbitrator


def run(seed):
    reference = shared_init(seed, 'monarch', False, None)
    memory = EpisodicByteMemory(max_order=CONFIG['memory_max_order'])
    memory.ingest_stream(TRAIN)
    train_batches = batches(TRAIN)
    result = {'seed': seed, 'memory_only': memory_only(memory, DEV, True), 'backbones': {}}
    for kind in ('monarch', 'dense'):
        for skip in (False, True):
            m = shared_init(seed, kind, skip, reference)
            m.episodic = copy.deepcopy(memory)
            opt = torch.optim.AdamW(m.parameters(), lr=.003, weight_decay=.01)
            for _ in range(3):
                m.train()
                for x, y in train_batches:
                    opt.zero_grad(set_to_none=True)
                    logits, _, aux, _ = m.forward_neural(x, targets=y)
                    loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)) + .1 * aux
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
                    opt.step()
                if skip:
                    m.calibrate_gates(train_batches)
            state = state_hash(m)
            arms = {}
            for name, bins, episodic in [('neural', 0, False), ('fixed', 0, True),
                                          ('global', 1, True), ('confidence', 4, True)]:
                model = copy.deepcopy(m)
                assert state_hash(model) == state
                a = ReliabilityArbitrator(bins=bins, share=.01) if bins else None
                start = time.perf_counter()
                score = model.evaluate_streaming(DEV, seq_len=32, use_episodic=episodic,
                    online_episodic_update=episodic, arbitrator=a, online_arbitration_update=bool(a))
                score['eval_seconds'] = time.perf_counter() - start
                arms[name] = score
            result['backbones'][f'{kind}_{"skip" if skip else "full"}'] = {
                'trained_state_sha256': state, 'arms': arms}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    if git('status', '--porcelain'):
        raise RuntimeError('development evidence requires a clean Git tree')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    result = {'schema':'nexus-arbitration-dev-v1', 'qualification':'DEVELOPMENT_ONLY',
              'scientific_trial_executed':False, 'source_sha':git('rev-parse','HEAD'),
              'source_tree':git('rev-parse','HEAD^{tree}'), 'config':CONFIG,
              'device':'cpu', 'threads':1, 'torch':torch.__version__, 'python':platform.python_version(),
              'platform':platform.platform(), 'epochs':3, 'share':.01, 'bins':[1,4],
              'fixture':{'train_sha256':hashlib.sha256(TRAIN).hexdigest(),
                         'dev_sha256':hashlib.sha256(DEV).hexdigest(),
                         'train_bytes':len(TRAIN), 'dev_bytes':len(DEV)},
              'runs':[run(s) for s in (42,43,44)]}
    result['seconds'] = time.perf_counter() - start
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')

if __name__ == '__main__':
    main()
