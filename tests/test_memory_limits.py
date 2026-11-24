import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nexus_expanse.memory import EpisodicByteMemory, HebbianFastWeightAdapter


def test_conflicting_observations_can_change_old_prediction():
    mem = EpisodicByteMemory(max_order=3)
    for _ in range(20):
        mem.observe_transition(b"abc", 65)
    before, _ = mem.predict_probs(b"abc")
    for _ in range(40):
        mem.observe_transition(b"abc", 66)
    after, _ = mem.predict_probs(b"abc")
    assert after[65] < before[65]
    assert torch.isclose(after.sum(), torch.tensor(1., dtype=torch.float64))


def test_fast_state_reset_is_not_durable_retention():
    adapter = HebbianFastWeightAdapter(8)
    h = torch.ones(8)
    adapter.hebbian_update(h, torch.full((256,), 1/256), 65, 3.)
    assert adapter.updates_applied == 1
    adapter.reset_fast_state()
    assert adapter.updates_applied == 0
    assert torch.count_nonzero(adapter.w_fast) == 0
