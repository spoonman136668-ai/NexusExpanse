import hashlib
import struct
import subprocess
import sys
import json

import pytest
import torch

from nexus_expanse import (NexusExpanseByteCore, backbone_identity, snapshot_fast_weights,
                          restore_fast_weights, save_fast_weights, load_fast_weights)

BUDGET = 20000


def model(seed=42, **kwargs):
    torch.manual_seed(seed)
    return NexusExpanseByteCore(n_layers=1, dim=16, n_heads=2, dropout=0., max_seq_len=16, **kwargs)


def trained():
    m = model()
    m.evaluate_streaming(b'abcabcabcabcabcabc', seq_len=8, use_episodic=False, online_hebbian_update=True)
    assert m.fast_weights.updates_applied > 0
    return m


def frozen(m):
    return m.evaluate_streaming(b'abcabcabcabcabcabc', seq_len=8, use_episodic=False,
                                use_fast_weights=True, reset_fast_state=False)


def test_exact_restore_and_frozen_prediction():
    m = trained()
    fresh = model()
    p = snapshot_fast_weights(m, max_bytes=BUDGET)
    restore_fast_weights(fresh, p, max_bytes=BUDGET)
    assert torch.equal(fresh.fast_weights.w_fast, m.fast_weights.w_fast)
    assert frozen(m) == frozen(fresh)
    assert snapshot_fast_weights(fresh, max_bytes=BUDGET) == p
    assert backbone_identity(m) == backbone_identity(fresh)
    assert frozen(m)['bpb'] != frozen(model())['bpb']


def test_cold_process(tmp_path):
    m = trained()
    path = tmp_path / 'fast.bin'
    save_fast_weights(m, path, max_bytes=BUDGET)
    script = '''import sys, torch, hashlib, json
from nexus_expanse import NexusExpanseByteCore, load_fast_weights, snapshot_fast_weights
torch.set_num_threads(1)
torch.manual_seed(42)
m=NexusExpanseByteCore(n_layers=1,dim=16,n_heads=2,dropout=0.,max_seq_len=16)
load_fast_weights(m,sys.argv[1],max_bytes=20000)
print(json.dumps([hashlib.sha256(snapshot_fast_weights(m,max_bytes=20000)).hexdigest(), m.evaluate_streaming(b'abcabcabcabcabcabc',seq_len=8,use_episodic=False,use_fast_weights=True,reset_fast_state=False)]))
'''
    assert json.loads(subprocess.check_output([sys.executable, '-c', script, str(path)], text=True)) == [hashlib.sha256(path.read_bytes()).hexdigest(), frozen(m)]
    with pytest.raises(FileExistsError):
        save_fast_weights(m, path, max_bytes=BUDGET)


@pytest.mark.parametrize('change', ['seed', 'lr', 'skip', 'gate', 'weight'])
def test_wrong_backbone_rejected_without_mutation(change):
    m = trained()
    other = model(43) if change == 'seed' else model(fast_lr=.2) if change == 'lr' else model(enable_skip=False) if change == 'skip' else model()
    if change == 'gate':
        other.layers[0].ffn.tau_eval.fill_(0.)
    if change == 'weight':
        with torch.no_grad():
            other.emb.weight[0,0] += 1
    before = other.fast_weights.w_fast.clone()
    with pytest.raises(ValueError, match='mismatch'):
        restore_fast_weights(other, snapshot_fast_weights(m, max_bytes=BUDGET), max_bytes=BUDGET)
    assert torch.equal(before, other.fast_weights.w_fast)
    assert other.fast_weights.updates_applied == 0


@pytest.mark.parametrize('mode', ['truncated', 'flipped', 'oversized', 'nan', 'out_of_range'])
def test_corrupt_checkpoint_rejected_without_mutation(mode):
    m = trained()
    p = snapshot_fast_weights(m, max_bytes=BUDGET)
    if mode == 'truncated': p = p[:-1]
    elif mode == 'flipped': p = p[:-1] + bytes([p[-1]^1])
    elif mode == 'oversized': p += b' ' * BUDGET
    else:
        body = p[:-36] + struct.pack('<f', float('nan') if mode == 'nan' else 3.)
        p = body + hashlib.sha256(body).digest()
    old = m.fast_weights.w_fast.clone()
    count = m.fast_weights.updates_applied
    with pytest.raises(ValueError):
        restore_fast_weights(m, p, max_bytes=BUDGET)
    assert torch.equal(old, m.fast_weights.w_fast) and count == m.fast_weights.updates_applied


def test_budget_and_default_reset(tmp_path):
    m = trained()
    with pytest.raises(ValueError): snapshot_fast_weights(m, max_bytes=10)
    path = tmp_path / 'fast.bin'
    save_fast_weights(m, path, max_bytes=BUDGET)
    with pytest.raises(ValueError): load_fast_weights(m, path, max_bytes=10)
    m.evaluate_streaming(b'abc', seq_len=8, use_episodic=False)
    assert m.fast_weights.updates_applied == 0 and not torch.count_nonzero(m.fast_weights.w_fast)


def test_continued_adaptation_matches_uninterrupted_state():
    m = trained(); fresh = model()
    restore_fast_weights(fresh, snapshot_fast_weights(m, max_bytes=BUDGET), max_bytes=BUDGET)
    kw = dict(seq_len=8, use_episodic=False, online_hebbian_update=True, reset_fast_state=False)
    assert m.evaluate_streaming(b'xyzxyzxyzxyz', **kw) == fresh.evaluate_streaming(b'xyzxyzxyzxyz', **kw)
    assert snapshot_fast_weights(m, max_bytes=BUDGET) == snapshot_fast_weights(fresh, max_bytes=BUDGET)
