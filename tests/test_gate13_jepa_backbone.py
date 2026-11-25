import torch

from benchmarks.gate12_deltanet_dev import ATTENTION, DELTA
from benchmarks.gate13_jepa_proven_backbone_dev import (
    ARMS, CE, JEPA, UNTRAINED, batches, comparisons, corpus_rows,
    encoder_of, histogram_control, model_for,
)
from nexus_expanse.jepa import state_hash
from nexus_expanse.jepa_backbone import last_hidden


def test_expanded_schedule_and_recomputed_histogram():
    train, fit, evaluation = corpus_rows()
    observed = [r for b in batches(train, 42, 1) for r in b]
    assert len(observed) == len(train) == 2520
    assert {r["ctx"] for r in observed} == {r["ctx"] for r in train}
    assert not {r["ctx"] for r in train} & {r["ctx"] for r in evaluation}
    assert histogram_control(fit, evaluation)["retrieval"]["top1"] == .20


def test_matched_encoder_initialization_and_ema_gradient_isolation():
    torch.set_num_threads(1)
    ctx = torch.arange(30).reshape(6, 5)
    lengths = torch.tensor([4, 5, 3, 4, 2, 5])
    domains = torch.tensor([0, 0, 1, 1, 2, 2])
    for backbone in (ATTENTION, DELTA):
        assert len({state_hash(encoder_of(model_for(42, backbone, arm))) for arm in ARMS}) == 1
        model = model_for(42, backbone, JEPA).train()
        assert not model.target.training
        assert last_hidden(model.context, ctx, lengths).shape == (6, 64)
        loss = model.loss(ctx, lengths, ctx.flip(1), lengths, domains)
        loss.backward()
        assert all(p.grad is None for p in model.target.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.context.parameters())
        source, target = next(model.context.parameters()), next(model.target.parameters())
        old = target.detach().clone()
        with torch.no_grad():
            source.add_(1)
        model.update_target(0, 600)
        torch.testing.assert_close(target, .9 * old + .1 * source)


def test_jepa_decision_requires_all_seeds_and_all_domain_cells():
    def arm(retrieval, math, causal):
        return dict(retrieval=dict(top1=retrieval), math=dict(accuracy=math),
                    science=dict(accuracy=causal),
                    noncollapse={d: dict(effective_rank=8.0, mean_std=.5) for d in ("math", "code", "science")})
    runs = [dict(seed=s, arms={JEPA: arm(.30, .50, .70), CE: arm(.25, .50, .70),
                             UNTRAINED: arm(.19, .40, .50)}) for s in (42, 43, 44)]
    assert comparisons(runs, JEPA)["status"] == "PASS"
    runs[0]["arms"][CE]["retrieval"]["top1"] = .26
    assert comparisons(runs, JEPA)["status"] == "MIXED"
    runs[1]["arms"][JEPA]["noncollapse"]["code"]["mean_std"] = .49
    assert comparisons(runs, JEPA)["status"] == "FAIL"
