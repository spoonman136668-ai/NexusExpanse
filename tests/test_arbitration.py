import copy
import math
import pytest
import torch

from benchmarks.ablation_dev import CONFIG, TRAIN
from nexus_expanse import NexusExpanseByteCore, ReliabilityArbitrator


def test_reliability_reverses_and_stays_bounded():
    a = ReliabilityArbitrator()
    for _ in range(30):
        a.observe(.8, .01, .9)
    assert .95 < a.weight(.8) <= .99
    for _ in range(30):
        a.observe(.8, .9, .01)
    assert .01 <= a.weight(.8) < .05
    assert a.updates == 60


def test_bins_isolated_and_ties_finite():
    a = ReliabilityArbitrator(4)
    a.observe(.99, 0, 1)
    assert a.weight(1) > .98
    assert a.weight(0) == .5
    a.observe(0, 0, 0)
    assert a.weight(0) == .5


@pytest.mark.parametrize('args', [(0,.01), (17,.01), (1,0), (1,.5), (1,float('nan'))])
def test_invalid_config(args):
    with pytest.raises(ValueError):
        ReliabilityArbitrator(*args)


def test_bad_likelihood_does_not_mutate():
    a = ReliabilityArbitrator()
    for p in (float('nan'), -1, 2):
        with pytest.raises(ValueError):
            a.observe(.5,p,.5)
    assert a.updates == 0 and a.weight(.5) == .5
    with pytest.raises(ValueError):
        a.weight(float('nan'))


def test_causal_first_prediction_and_frozen_arbitration():
    m = NexusExpanseByteCore(**CONFIG).eval()
    m.episodic.ingest_stream(TRAIN)
    a = ReliabilityArbitrator(4)
    b = copy.deepcopy(a)
    x = copy.deepcopy(m).evaluate_streaming(b'ab', seq_len=32, arbitrator=a, online_arbitration_update=True,
                                             online_episodic_update=True)
    y = m.evaluate_streaming(b'ab', seq_len=32, arbitrator=b)
    assert x['bpb'] == y['bpb'] and x['mean_memory_weight'] == .5
    assert x['arbitration_updates'] == 1 and y['arbitration_updates'] == 0
    assert b.updates == 0
    assert math.isfinite(x['bpb'])


def test_default_unchanged_and_invalid_combinations():
    m = NexusExpanseByteCore(**CONFIG).eval()
    m.episodic.ingest_stream(TRAIN)
    x = m.evaluate_streaming(b'abc', seq_len=32)
    y = m.evaluate_streaming(b'abc', seq_len=32, arbitrator=None)
    assert x == y
    for kw in ({'online_arbitration_update': True}, {'arbitrator': ReliabilityArbitrator(), 'use_episodic': False}):
        with pytest.raises(ValueError):
            m.evaluate_streaming(b'ab', seq_len=32, **kw)
