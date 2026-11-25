import torch
from benchmarks.gate15_hybrid_dev import parity
from benchmarks.gate18_hybrid_v2_dev import decision, rate
from nexus_expanse.hybrid_backbone import HybridDeltaAttentionCore


def test_stabilization_is_opt_in_and_differentiable():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    baseline = HybridDeltaAttentionCore(dim=32)
    torch.manual_seed(23)
    new = HybridDeltaAttentionCore(dim=32, stabilized=True)
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, new.state_dict()[key])
    assert (new.layers[0].layer_scale == .5).all()
    logits = new.forward_neural(torch.tensor([[4, 6, 2, 6]]))[0]
    logits.square().mean().backward()
    assert new.layers[0].layer_scale.grad.abs().sum() > 0
    assert new.layers[0].readout_norm.weight.grad.abs().sum() > 0
    assert parity(1, stabilized=True)["passed"]


def test_warmup_and_cosine_boundaries():
    assert rate(1, 3000) == .003 / 300
    assert rate(300, 3000) == .003
    assert rate(301, 3000) < .003
    assert rate(3000, 3000) == 0.


def test_frozen_five_seed_decision():
    def row(seed, score):
        return dict(seed=seed, v2=dict(heldout={"math": dict(accuracy=score), "science": dict(accuracy=1.)}),
                    timing={"64": dict(speedup=6., cache_reduction=.50)})
    values = [row(s, .95) for s in range(42, 47)]
    assert decision(values, True)["status"] == "PASS"
    values[-1] = row(46, .925)
    assert decision(values, True)["status"] == "MIXED"
    values[0] = row(42, .925)
    assert decision(values, True)["status"] == "FAIL"
