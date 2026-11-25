"""Phase 3A invariants on untrained native receivers and static donor checkpoints."""
import math

import pytest
import torch

from nexus_expanse.checkpoint_transport import ImportedSkill, load_skill
from nexus_expanse.donor_assimilation import DenseTokenDonorLM, forbid_donor_forward
from nexus_expanse.multi_skill_router import MultiSkillBank, load_bank, save_bank


ROUTERS = ("bayes_bma", "bayes_argmax", "surprise_sparse")


@pytest.fixture(scope="module", autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def random_bank(**kwargs):
    return MultiSkillBank(
        ImportedSkill(),
        {name: ImportedSkill() for name in ("rule_A", "rule_B", "markdown_prose")},
        **kwargs,
    )


@pytest.mark.parametrize("seed", range(50))
def test_prefix_causality_exact_for_all_routers(seed):
    torch.manual_seed(seed)
    bank = random_bank()
    raw = torch.randint(0, 256, (2, 24))
    cut = 1 + seed % (raw.size(1) - 1)
    alternate = raw.clone()
    alternate[:, cut:] = (alternate[:, cut:] + 1 + seed) % 256
    for name in ROUTERS:
        route = getattr(bank, f"forward_{name}")
        original = route(raw)
        perturbed = route(alternate)
        shortened = route(raw[:, :cut], include_next=True)
        for first, second, short in zip(original[:2], perturbed[:2], shortened[:2], strict=True):
            assert torch.equal(first[:, :cut + 1], second[:, :cut + 1]), (seed, name)
            assert torch.equal(first[:, :cut + 1], short), (seed, name)


def test_bayesian_updates_use_observed_byte_before_next_forecast():
    torch.manual_seed(80)
    bank = random_bank()
    raw = torch.tensor([[255, 1, 17, 9]], dtype=torch.uint8)
    forecasts = bank.component_log_probs(raw)
    bma, weights = bank.forward_bayes_bma(raw, gamma=0.5)
    argmax, active = bank.forward_bayes_argmax(raw, gamma=0.5)
    assert torch.equal(weights[0, 0], torch.full((4,), 0.25))
    scores = torch.zeros(4)
    for t in range(raw.size(1)):
        assert torch.allclose(weights[0, t], scores.softmax(-1), atol=1e-7, rtol=0)
        assert active[0, t] == scores.argmax()
        assert torch.equal(argmax[0, t], forecasts[0, t, active[0, t]])
        expected = (weights[0, t, :, None] * forecasts[0, t].exp()).sum(0)
        assert torch.allclose(bma[0, t].exp(), expected, atol=1e-7, rtol=0)
        scores = 0.5 * scores + forecasts[0, t, :, raw[0, t].long()]


def test_static_import_and_cold_round_trip(tmp_path):
    torch.manual_seed(81)
    paths = {}
    for name in ("base_python", "rule_A", "rule_B", "markdown_prose"):
        donor = DenseTokenDonorLM(256, 32, 2, 4, 2, 64)
        path = tmp_path / f"{name}.pt"
        rows = torch.randperm(256)
        torch.save(dict(weights=donor.state_dict(), byte_to_row=rows), path)
        paths[name] = path
        del donor
    with forbid_donor_forward() as guard:
        skills = {name: load_skill(path, "residual8") for name, path in paths.items()}
        bank = MultiSkillBank(
            skills.pop("base_python"), skills, gamma=0.72,
            surprise_threshold_bits=4.25, probe_interval=7,
        )
        assert all(not parameter.requires_grad for parameter in bank.parameters())
        raw = torch.randint(0, 256, (2, 17))
        snapshot = tmp_path / "bank.pt"
        save_bank(bank, snapshot)
        restored = load_bank(snapshot)
        assert restored.skill_names == bank.skill_names
        assert restored.gamma.hex() == bank.gamma.hex()
        assert restored.surprise_threshold_bits.hex() == bank.surprise_threshold_bits.hex()
        assert restored.probe_interval == bank.probe_interval
        assert restored.state_dict().keys() == bank.state_dict().keys()
        assert all(torch.equal(value, restored.state_dict()[name])
                   for name, value in bank.state_dict().items())
        for name in ROUTERS:
            reference = getattr(bank, f"forward_{name}")(raw)
            actual = getattr(restored, f"forward_{name}")(raw)
            for first, second in zip(reference[:2], actual[:2], strict=True):
                assert torch.equal(first, second)
            assert reference[2:] == actual[2:]
        assert guard.donor_forward_calls == 0


def test_sparse_runs_only_one_skill_between_probes():
    torch.manual_seed(82)
    bank = random_bank()
    with torch.no_grad():
        bank.base.core.pos_emb.weight.zero_()
        direction = bank.base.core.forward_neural(torch.zeros(1, 1, dtype=torch.long))[1][0, 0]
        bank.base.readout.weight.zero_()
        bank.base.readout.weight[0] = 16 * direction
    expert_calls = [0] * len(bank.skills)

    def count(index):
        def hook(_model, args):
            expert_calls[index] += args[0].size(0)
        return hook

    hooks = [skill.register_forward_pre_hook(count(i)) for i, skill in enumerate(bank.skills.values())]
    raw = torch.zeros(2, 48, dtype=torch.long)
    try:
        forecasts, active, average = bank.forward_surprise_sparse(raw)
    finally:
        for hook in hooks:
            hook.remove()
    assert torch.equal(active, torch.zeros_like(active))
    assert (forecasts[:, 1:, 0] > -3.5 * math.log(2)).all()
    # The initial surprise and offsets 16 and 32 each schedule a full probe.
    assert expert_calls == [94, 6, 6, 6]
    assert average == sum(expert_calls) / raw.numel() < len(bank.skills)


def test_byte_validation_and_rolling_context():
    torch.manual_seed(83)
    bank = random_bank()
    with pytest.raises(ValueError, match="raw bytes"):
        bank.forward_bayes_bma(torch.tensor([[256]]))
    raw = torch.randint(0, 256, (1, bank.context + 4), dtype=torch.uint8)
    shorter = raw[:, :-1]
    for name in ROUTERS:
        route = getattr(bank, f"forward_{name}")
        full_result = route(raw, include_next=True)
        prefix_result = route(shorter, include_next=True)
        assert torch.equal(full_result[0][:, :-1], prefix_result[0])
        assert torch.equal(full_result[1][:, :-1], prefix_result[1])
