"""Causal forecast, filter, sparse accounting and cold CLI checks."""
import json
import math
import subprocess
import sys

import pytest
import torch

from nexus_expanse.checkpoint_transport import ImportedSkill
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.multi_skill_router import MultiSkillBank, load_bank, save_bank


ROUTERS = ("hmm_bma", "hmm_sparse")


@pytest.fixture(scope="module", autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_bank():
    return MultiSkillBank(
        ImportedSkill(),
        {name: ImportedSkill() for name in ("rule_A", "rule_B", "markdown_prose")},
    )


def constant_emission_bank():
    bank = make_bank()
    for model, (a, b) in zip(bank.skills.values(), ((0.72, 0.24), (0.24, 0.72), (0.01, 0.01), (0.01, 0.01)), strict=True):
        logits = torch.full((256,), math.log((1 - a - b) / 254))
        logits[11] = math.log(a)
        logits[22] = math.log(b)

        def forecast(prefix, logits=logits):
            return logits.expand(*prefix.shape, -1)

        model.forward = forecast
    return bank


@pytest.mark.parametrize("seed", range(50))
def test_hmm_exact_prefix_causality(seed):
    torch.manual_seed(seed)
    bank = make_bank()
    length = bank.context + 6 if seed == 49 else 12 + seed % 7
    raw = torch.randint(0, 256, (2, length), dtype=torch.uint8)
    cut = length - 2 if seed == 49 else 1 + seed % (length - 1)
    changed = raw.clone()
    changed[:, cut:] = (changed[:, cut:].long() + seed + 1) % 256
    for name in ROUTERS:
        route = getattr(bank, f"forward_{name}")
        original = route(raw)
        alternate = route(changed)
        short = route(raw[:, :cut], include_next=True)
        for a, b, c in zip(original[:2], alternate[:2], short[:2], strict=True):
            assert torch.equal(a[:, :cut + 1], b[:, :cut + 1]), (seed, name)
            assert torch.equal(a[:, :cut + 1], c), (seed, name)


def test_predictive_prior_matches_first_order_filter():
    bank = constant_emission_bank()
    raw = torch.tensor([[11, 11, 22, 11, 22, 22], [22, 11, 11, 11, 22, 11]])
    probability = 0.05
    logp, weights = bank.forward_hmm_bma(raw, probability, include_next=True)
    experts = bank.component_log_probs(raw, include_next=True).double().exp()
    transition = torch.full((4, 4), probability / 3, dtype=torch.float64).fill_diagonal_(1 - probability)
    state = torch.full((2, 4), 0.25, dtype=torch.float64)
    for t in range(raw.size(1) + 1):
        torch.testing.assert_close(weights[:, t].double(), state, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(logp[:, t].double().exp(), (experts[:, t] * state.unsqueeze(-1)).sum(-2),
                                   rtol=5e-6, atol=1e-8)
        if t < raw.size(1):
            observed = raw[:, t].view(-1, 1, 1).expand(-1, 4, 1)
            state *= experts[:, t].gather(-1, observed).squeeze(-1)
            state = (state / state.sum(-1, keepdim=True)) @ transition
    _, sparse_weights, calls = bank.forward_hmm_sparse(raw, probe_interval=1, include_next=True)
    torch.testing.assert_close(sparse_weights, weights, rtol=1e-6, atol=1e-7)
    assert calls == 4 * raw.size(1) / (raw.size(1) + 1)


def test_sparse_probe_replays_four_bytes_and_other_steps_call_one_expert():
    bank = constant_emission_bank()
    raw = torch.tensor([[11] * 6 + [22] * 4 + [11] * 7])
    calls = [0] * 4

    def count(index):
        def hook(_model, args):
            calls[index] += args[0].size(0)
        return hook

    hooks = [skill.register_forward_pre_hook(count(k)) for k, skill in enumerate(bank.skills.values())]
    try:
        _, weights, average = bank.forward_hmm_sparse(raw, surprise_threshold_bits=100, probe_interval=5)
    finally:
        for hook in hooks:
            hook.remove()

    # At the first probe the prior four observed targets are all 11. At the
    # second probe they are all 22, enough to overcome the sticky prior.
    assert weights[0, 5].argmax() == 0
    assert weights[0, 9, 0] > weights[0, 9, 1]
    assert weights[0, 10].argmax() == 1
    assert all(count >= 3 for count in calls)
    assert sum(calls) == raw.numel() - 1 + 3 * 3
    assert average == sum(calls) / raw.numel() < len(bank.skills)


def test_dense_hmm_switches_faster_than_discounted_bayes():
    bank = constant_emission_bank()
    raw = torch.tensor([[11] * 32 + [22] * 32])
    states = [bank.forward_hmm_bma(raw)[1], bank.forward_bayes_bma(raw, gamma=0.85)[1]]
    delays = []
    for weights in states:
        active = weights[0].argmax(-1)
        assert active[31] == 0 and active[-1] == 1
        delays.append(next(start for start in range(25) if (active[32 + start:40 + start] == 1).all()))
    assert 0 < delays[0] < delays[1]


def test_hmm_cold_restore_with_existing_bank_format(tmp_path):
    torch.manual_seed(1600)
    with forbid_donor_forward() as guard:
        original = make_bank()
        path = tmp_path / "bank.pt"
        save_bank(original, path)
        restored = load_bank(path)
        assert original.skill_names == restored.skill_names
        raw = torch.randint(256, (2, 17))
        for name in ROUTERS:
            a = getattr(original, f"forward_{name}")(raw, switch_prob=0.17)
            b = getattr(restored, f"forward_{name}")(raw, switch_prob=0.17)
            assert all(torch.equal(x, y) for x, y in zip(a[:2], b[:2], strict=True))
            assert a[2:] == b[2:]
    assert guard.donor_forward_calls == 0


@pytest.mark.parametrize("router", ROUTERS)
def test_hmm_route_stream_cli(tmp_path, router):
    bank = make_bank()
    snapshot = tmp_path / "bank.pt"
    save_bank(bank, snapshot)
    command = [
        sys.executable, "-m", "nexus_expanse.cli", "route-stream", "--bank", str(snapshot),
        "--router", router, "--input-hex", "00ff10", "--input-hex", "a002fe07",
    ]
    report = json.loads(subprocess.check_output(command, text=True))
    assert report["router"] == router
    assert report["prefix_max_abs_diff"] == 0.0
    assert report["donor_forward_calls"] == 0
    assert [chunk["end_offset"] for chunk in report["chunks"]] == [3, 7]
    raw = torch.tensor([list(bytes.fromhex("00ff10a002fe07"))])
    logp, weights, *calls = getattr(bank, f"forward_{router}")(raw, include_next=True)
    assert report["avg_active_skills_per_byte"] == (calls[0] if calls else 4 * 7 / 8)
    for chunk in report["chunks"]:
        pos = chunk["end_offset"]
        assert chunk["routing_weights"] == pytest.approx(dict(zip(bank.skill_names, weights[0, pos].tolist(), strict=True)))
        assert chunk["active_skill"] == bank.skill_names[int(weights[0, pos].argmax())]
        assert chunk["next_byte"] == int(logp[0, pos].argmax())


def test_single_expert_and_switch_probability_endpoints():
    raw = torch.tensor([[0, 255, 2]])
    for bank in (MultiSkillBank(ImportedSkill(), {}), make_bank()):
        for name in ROUTERS:
            route = getattr(bank, f"forward_{name}")
            for probability in (0.0, 1.0):
                logp, weights, *_ = route(raw, switch_prob=probability)
                assert logp.isfinite().all()
                torch.testing.assert_close(weights.sum(-1), torch.ones_like(raw, dtype=torch.float32))
            with pytest.raises(ValueError, match="switch_prob"):
                route(raw, switch_prob=float("nan"))
