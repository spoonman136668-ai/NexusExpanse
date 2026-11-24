import copy
import math
import pytest
import torch

from benchmarks.ablation_dev import ARMS, CONFIG, TRAIN, DEV, batches, run_dev, shared_init
from nexus_expanse.core import NexusExpanseByteCore


def test_training_remainder_coverage():
    bs = batches(bytes(range(80)))
    assert sum(y.numel() for _, y in bs) == 79
    assert torch.cat([y.flatten() for _, y in bs]).tolist() == list(range(1, 80))


@pytest.mark.parametrize('kind', ['dense', 'monarch'])
@pytest.mark.parametrize('skip', [False, True])
def test_matched_backbone_prefix(kind, skip):
    m = NexusExpanseByteCore(**CONFIG, ffn_kind=kind, enable_skip=skip)
    x = torch.tensor(list(TRAIN[:32]))[None]
    m.calibrate_gates([(x[:, :-1], x[:, 1:])])
    m.eval()
    with torch.inference_mode():
        a = m.forward_neural(x)[0]
        b = m.forward_neural(x[:, :13])[0]
    assert torch.allclose(a[:, :13], b, atol=1e-5)


def test_same_initial_shared_parameters():
    ref = shared_init(42, 'monarch', False, None)
    dense = shared_init(42, 'dense', False, ref)
    for name, value in ref.state_dict().items():
        if '.ffn.fc' not in name:
            assert torch.equal(value, dense.state_dict()[name])


def test_first_prediction_precedes_writes_and_frozen_fast_state():
    m = NexusExpanseByteCore(**CONFIG)
    m.episodic.ingest_stream(TRAIN)
    frozen = copy.deepcopy(m)
    a = m.evaluate_streaming(b'xy', seq_len=32, online_episodic_update=True, online_hebbian_update=True)
    b = frozen.evaluate_streaming(b'xy', seq_len=32)
    assert a['bpb'] == b['bpb']
    assert m.episodic.num_observations == frozen.episodic.num_observations + 1
    m.evaluate_streaming(DEV, seq_len=32, online_hebbian_update=True)
    saved = m.fast_weights.w_fast.clone()
    writes = m.fast_weights.updates_applied
    assert writes > 0
    m.evaluate_streaming(DEV, seq_len=32, use_fast_weights=True, reset_fast_state=False)
    assert torch.equal(saved, m.fast_weights.w_fast)
    assert m.fast_weights.updates_applied == writes


def test_invalid_inputs():
    m = NexusExpanseByteCore(**CONFIG)
    for kwargs in ({'seq_len': 0}, {'seq_len': 33}, {'mem_blend_cap': 1.1},
                   {'online_episodic_update': True, 'use_episodic': False},
                   {'online_hebbian_update': True, 'use_fast_weights': False}):
        with pytest.raises(ValueError):
            m.evaluate_streaming(b'xy', **({"seq_len": 32} | kwargs))
    with pytest.raises(ValueError):
        m.evaluate_streaming(b'x', seq_len=32)


def test_development_full_factorial():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        r = run_dev(42, epochs=1)
    finally:
        torch.set_num_threads(old)
    assert len(r['backbones']) == 4
    for backbone in r['backbones'].values():
        assert set(backbone['arms']) == set(ARMS)
        for name, score in backbone['arms'].items():
            assert score['bytes'] == len(DEV) - 1 and math.isfinite(score['bpb'])
            if not ARMS[name][2]:
                assert score['hebbian_updates'] == 0
        assert backbone['arms']['neural']['episodic_entries'] == backbone['arms']['static_episodic']['episodic_entries']
    assert r['backbones']['monarch_full']['ffn_bytes'] < r['backbones']['dense_full']['ffn_bytes']
