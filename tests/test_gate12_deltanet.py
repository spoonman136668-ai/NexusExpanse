import json

import pytest
import torch

from benchmarks.gate12_deltanet_dev import DELTA, decision, parity, setup
from benchmarks.gate_custody import EpochRun
from nexus_expanse.deltanet import DeltaNetMonarchByteCore, GatedDeltaNetLayer


def test_delta_rule_lookup_and_overwrite():
    state = torch.zeros((2, 2), dtype=torch.float64)
    k1, k2 = torch.eye(2, dtype=state.dtype)
    a, b = torch.tensor([2., -3.], dtype=state.dtype), torch.tensor([7., 4.], dtype=state.dtype)
    one = state.new_tensor(1.)
    update = GatedDeltaNetLayer.update
    state = update(state, k1, a, one, one)
    state = update(state, k2, b, one, one)
    assert torch.equal(state @ k1, a)
    assert torch.equal(state @ k2, b)
    state = update(state, k1, -a, one, one)
    assert torch.equal(state @ k1, -a)
    assert torch.equal(state @ k2, b)


def test_parallel_training_matches_recurrence_and_backpropagates():
    setup()
    torch.manual_seed(8)
    layer = GatedDeltaNetLayer(32).eval()
    x = torch.randn(3, 9, 32)
    state = layer.init_state(3, x.device)
    frames = []
    for part in x.unbind(1):
        frame, state = layer.forward_step(part, state)
        frames.append(frame)
    parallel = layer(x)
    torch.testing.assert_close(parallel, torch.stack(frames, 1), rtol=1e-5, atol=1e-6)
    parallel.square().mean().backward()
    grad = layer.input_proj.weight.grad
    assert torch.isfinite(grad).all() and (grad.abs().sum(1) > 0).all()


def test_exact_eval_and_independent_streams():
    setup()
    check = parity(seeds=2)
    assert check["passed"], check
    assert [r["state_tensor_bytes"] for r in check["runs"]] == [16384, 16384, 65536, 65536]
    model = DeltaNetMonarchByteCore().eval()
    a, b = model.init_stream_state(), model.init_stream_state()
    _, _, after = model.step_logits(1, a)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    assert any(torch.count_nonzero(x) for x in after)
    with pytest.raises(ValueError, match="raw byte"):
        model.step_logits(256, a)


def test_gate12_predeclared_decision_and_eligibility():
    def run(math, causal, speed):
        return dict(capacity={DELTA: dict(heldout=dict(math=dict(accuracy=math),
                                                      science=dict(accuracy=causal)))},
                    timing={"64": dict(speedup=speed), "128": dict(speedup=speed)})
    assert decision([run(.85, .90, 2)], {"passed": True})["status"] == "PASS"
    mixed = decision([run(.25, .50, 3)], {"passed": True})
    assert mixed["status"] == "MIXED" and not mixed["delta_eligible_for_gate13"]
    assert decision([run(.25, .50, 1)], {"passed": True})["status"] == "FAIL"
    assert decision([run(.90, .90, 3)], {"passed": False})["status"] == "FAIL"


def test_checkpoint_refuses_ambiguous_consumed_epoch(tmp_path):
    run = EpochRun(tmp_path, "trial", {"source_sha": "frozen"}, 2, 1)
    intent, _, _ = run.paths(1)
    intent.write_text(json.dumps({"epoch": 1}))
    with pytest.raises(RuntimeError, match="replay refused"):
        run.history()
