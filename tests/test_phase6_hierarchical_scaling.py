"""Exact compatibility, causality and group-mass checks for hierarchical routing."""
import json
import subprocess
import sys

import pytest
import torch

from nexus_expanse.checkpoint_transport import ImportedSkill
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.jepa import state_hash
from nexus_expanse.multi_skill_router import MultiSkillBank, load_bank
from nexus_expanse.skill_birth import EPISODIC_BIRTH, OpenWorldSkillBank


RESIDENT = ("base_python", "rule_A", "rule_B", "markdown_prose")
BORN = ("rule_C", "rule_D", "rule_E")


@pytest.fixture(scope="module", autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def build_bank(born_count=0):
    born = tuple(f"new_{index}" for index in range(born_count))
    return MultiSkillBank(ImportedSkill(), {name: ImportedSkill() for name in (*RESIDENT[1:], *born)},
                          resident_names=RESIDENT)


@pytest.mark.parametrize("seed", range(50))
def test_k7_exact_prefix_causality(seed):
    torch.manual_seed(seed)
    bank = build_bank(3)
    length = 70 if seed == 49 else 12 + seed % 7
    raw = torch.randint(0, 256, (2, length), dtype=torch.uint8)
    cut = length - 2 if seed == 49 else 1 + seed % (length - 1)
    changed = raw.clone()
    changed[:, cut:] = (changed[:, cut:].long() + seed + 1) % 256

    original = bank.forward_hmm_bma_hierarchical(raw)
    perturbed = bank.forward_hmm_bma_hierarchical(changed)
    truncated = bank.forward_hmm_bma_hierarchical(raw[:, :cut], include_next=True)
    for first, second, third in zip(original, perturbed, truncated, strict=True):
        assert torch.equal(first[:, :cut + 1], second[:, :cut + 1]), seed
        assert torch.equal(first[:, :cut + 1], third), seed
    assert original[1].size(-1) == 7


@pytest.mark.parametrize("switch_prob", (0.0, 0.05, 1.0))
def test_unexpanded_bank_is_bit_exact_with_standard_hmm(switch_prob):
    torch.manual_seed(1900)
    bank = build_bank()
    raw = torch.randint(0, 256, (2, 14))
    standard = bank.forward_hmm_bma(raw, switch_prob, include_next=True)
    hierarchical = bank.forward_hmm_bma_hierarchical(raw, switch_prob, include_next=True)
    assert all(torch.equal(a, b) for a, b in zip(standard, hierarchical, strict=True))


@pytest.mark.parametrize("born_count", (1, 3, 10))
def test_born_slots_share_a_fixed_budget(born_count):
    bank = build_bank(born_count)
    raw = torch.empty((1, 0), dtype=torch.long)
    _, weights = bank.forward_hmm_bma_hierarchical(raw, include_next=True)
    torch.testing.assert_close(weights[0, 0, :4], torch.full((4,), 0.245))
    torch.testing.assert_close(weights[0, 0, 4:], torch.full((born_count,), 0.02 / born_count))


def test_filter_matches_explicit_transition_and_interleaved_groups():
    torch.manual_seed(1901)
    bank = MultiSkillBank(
        ImportedSkill(), {name: ImportedSkill() for name in ("new_0", "peer", "new_1")},
        resident_names=("base_python", "peer"),
    )
    raw = torch.tensor([[1, 0, 255, 42, 6], [5, 2, 193, 33, 0]])
    switch, mass, share = 0.2, 0.1, 0.3
    components = bank.component_log_probs(raw, include_next=True).double().exp()
    forecast, weights = bank.forward_hmm_bma_hierarchical(raw, switch, mass, share, include_next=True)
    transition = torch.tensor([
        [0.8, 0.03, 0.14, 0.03],
        [0.2 / 3, 0.8, 0.2 / 3, 0.2 / 3],
        [0.14, 0.03, 0.8, 0.03],
        [0.2 / 3, 0.2 / 3, 0.2 / 3, 0.8],
    ], dtype=torch.float64)
    state = torch.tensor([[0.45, 0.05, 0.45, 0.05]] * 2, dtype=torch.float64)
    for t in range(raw.size(1) + 1):
        torch.testing.assert_close(weights[:, t].double(), state, atol=1e-7, rtol=2e-6)
        expected = (components[:, t] * state.unsqueeze(-1)).sum(-2)
        torch.testing.assert_close(forecast[:, t].double().exp(), expected, atol=1e-8, rtol=5e-6)
        if t < raw.size(1):
            where = raw[:, t].view(-1, 1, 1).expand(-1, 4, 1)
            state = state * components[:, t].gather(-1, where).squeeze(-1)
            state = (state / state.sum(-1, keepdim=True)) @ transition


def test_single_resident_switches_to_born_and_invalid_groups_are_rejected():
    bank = MultiSkillBank(ImportedSkill(), {"born": ImportedSkill()}, resident_names=("base_python",))
    raw = torch.tensor([[255]])
    _, weights = bank.forward_hmm_bma_hierarchical(raw, 0.2, 0.0, 0.1, include_next=True)
    torch.testing.assert_close(weights, torch.tensor([[[1.0, 0.0], [0.8, 0.2]]]))
    for group in ((), ("born",), ("base_python", "unknown"), ("base_python", "base_python")):
        with pytest.raises(ValueError, match="resident names"):
            MultiSkillBank(ImportedSkill(), {"born": ImportedSkill()}, resident_names=group)


def test_sequential_episodic_birth_preserves_residents_and_cold_state(tmp_path):
    torch.manual_seed(1902)
    bank = build_bank()
    world = OpenWorldSkillBank(bank)
    probe = torch.randint(256, (2, 15))
    before_hashes = [state_hash(skill) for skill in bank.skills.values()]
    before = bank.component_log_probs(probe)
    with forbid_donor_forward() as guard:
        for index, name in enumerate(BORN):
            encounter = torch.tensor([[176 + index, 192 + index, 208, 224 + index] * 5])
            world.spawn_and_adapt_skill(name, encounter, steps=1, mode=EPISODIC_BIRTH)
        assert world.resident_names == RESIDENT
        assert bank.skill_names == (*RESIDENT, *BORN)
        assert before_hashes == [state_hash(bank.skills[name]) for name in RESIDENT]
        assert torch.equal(before, bank.component_log_probs(probe)[:, :, :4])
        path = tmp_path / "expanded.pt"
        world.save_bank(path)
        restored = OpenWorldSkillBank.load_bank(path)
        assert restored.resident_names == RESIDENT
        assert state_hash(restored.bank) == state_hash(bank)
        actual = restored.bank.forward_hmm_bma_hierarchical(probe, include_next=True)
        expected = bank.forward_hmm_bma_hierarchical(probe, include_next=True)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
    assert guard.donor_forward_calls == 0


def test_legacy_snapshot_defaults_to_all_resident_and_cli_uses_saved_groups(tmp_path):
    torch.manual_seed(1903)
    world = OpenWorldSkillBank(build_bank(3))
    path = tmp_path / "bank.pt"
    world.save_bank(path)
    package = torch.load(path, weights_only=True)
    assert package.pop("resident_names") == list(RESIDENT)
    legacy = tmp_path / "legacy.pt"
    torch.save(package, legacy)
    restored = load_bank(legacy)
    assert restored.resident_names == restored.skill_names

    command = [sys.executable, "-m", "nexus_expanse.cli", "route-stream", "--bank", str(path),
               "--router", "hmm_bma_hierarchical", "--input-hex", "ff00fe80", "--input-hex", "123456"]
    output = json.loads(subprocess.check_output(command, text=True, timeout=60))
    assert output["prefix_max_abs_diff"] == 0.0
    assert output["donor_forward_calls"] == 0
    assert output["router"] == "hmm_bma_hierarchical"
    raw = torch.tensor([list(bytes.fromhex("ff00fe80123456"))])
    logp, weights = world.bank.forward_hmm_bma_hierarchical(raw, include_next=True)
    for item, end in zip(output["chunks"], (4, 7), strict=True):
        assert item["end_offset"] == end
        expected = dict(zip(world.bank.skill_names, weights[0, end].tolist(), strict=True))
        assert item["routing_weights"] == pytest.approx(expected, abs=1e-6)
        assert item["next_byte"] == int(logp[0, end].argmax())
