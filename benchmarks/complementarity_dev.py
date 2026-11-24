"""Stronger training and complementary expert comparison on known development fixtures."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import platform
import random
import sys
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.ablation_dev import (ROOT, TRAIN, DEV, CONFIG, batches, shared_init, state_hash,
                          memory_only, EpisodicByteMemory, torch, F)
from nexus_expanse import ReliabilityArbitrator


EPOCHS = 40


def complementary_fixture():
    # Position 12 depends on position 0, beyond an eight-byte suffix.
    # Aligned artificial records, deliberately not a generalization benchmark.
    rng = random.Random(8401)
    def record(flag):
        return bytes([65 + flag]) + b'.' * 11 + bytes([88 + flag]) + bytes(rng.choices(b'0123456789', k=18)) + b'\n'
    train = b''.join(record(i % 2) for i in range(32))
    familiar = b''.join(record(i % 2) for i in range(16))
    novel = bytes(range(128,159)) + b'\n'
    return train, familiar + novel * 16


def run(seed, train=TRAIN, dev=DEV, epochs=EPOCHS):
    reference = shared_init(seed, 'monarch', False, None)
    memory = EpisodicByteMemory(max_order=CONFIG['memory_max_order'])
    memory.ingest_stream(train)
    train_batches = batches(train)
    result = {'seed': seed, 'memory_only': memory_only(memory, dev, True), 'backbones': {}}
    for kind in ('monarch', 'dense'):
        for skip in (False, True):
            m = shared_init(seed, kind, skip, reference)
            m.episodic = copy.deepcopy(memory)
            opt = torch.optim.AdamW(m.parameters(), lr=.003, weight_decay=.01)
            for _ in range(epochs):
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
                score = model.evaluate_streaming(dev, seq_len=32, use_episodic=episodic,
                    online_episodic_update=episodic, arbitrator=a, online_arbitration_update=bool(a))
                score['eval_seconds'] = time.perf_counter() - start
                arms[name] = score
            result['backbones'][f'{kind}_{"skip" if skip else "full"}'] = {
                'trained_state_sha256': state, 'optimizer_steps': epochs * len(train_batches), 'arms': arms}
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
    result = {'schema':'nexus-complementarity-dev-v1', 'qualification':'DEVELOPMENT_ONLY',
              'scientific_trial_executed':False, 'source_sha':git('rev-parse','HEAD'),
              'source_tree':git('rev-parse','HEAD^{tree}'), 'config':CONFIG,
              'device':'cpu', 'threads':1, 'torch':torch.__version__, 'python':platform.python_version(),
              'platform':platform.platform(), 'epochs':EPOCHS, 'share':.01, 'bins':[1,4], 'fixtures':{}}
    for name, (train, dev) in [('original', (TRAIN, DEV)), ('complementary', complementary_fixture())]:
        result['fixtures'][name] = {'train_sha256':hashlib.sha256(train).hexdigest(),
                         'dev_sha256':hashlib.sha256(dev).hexdigest(),
                         'train_bytes':len(train), 'dev_bytes':len(dev),
                         'runs':[run(seed, train, dev) for seed in (42,43,44)]}
    result['seconds'] = time.perf_counter() - start
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')

if __name__ == '__main__':
    main()
