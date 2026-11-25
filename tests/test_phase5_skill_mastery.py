"""Expanded-bank causality, resident isolation and episodic snapshot tests."""
import json
import subprocess
import sys

import pytest
import torch

from nexus_expanse import skill_birth
from nexus_expanse.checkpoint_transport import ImportedSkill
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.jepa import state_hash
from nexus_expanse.multi_skill_router import BANK_V1, BANK_V2, MultiSkillBank, load_bank, save_bank
from nexus_expanse.skill_birth import CONSOLIDATED_BIRTH, EPISODIC_BIRTH, EpisodicAssistedBornSkill, OpenWorldSkillBank


@pytest.fixture(scope="module", autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_bank():
    return OpenWorldSkillBank(MultiSkillBank(
        ImportedSkill(),
        {name: ImportedSkill() for name in ("rule_A", "rule_B", "markdown_prose")},
    ))


@pytest.mark.parametrize("seed", range(50))
def test_expanded_episodic_hmm_prefix_is_exact(seed):
    torch.manual_seed(seed)
    world = make_bank()
    motif = torch.randint(0, 256, (2, 5))
    encounter = motif.repeat(1, 4)
    skill = world.spawn_and_adapt_skill("skill_novel", encounter, steps=1, mode=EPISODIC_BIRTH)
    assert isinstance(skill, EpisodicAssistedBornSkill)
    assert skill.keys.numel() > 0
    length = 70 if seed == 49 else 10 + seed % 7
    raw = motif.repeat(1, (length + 4) // 5)[:, :length].contiguous()
    cut = length - 2 if seed == 49 else 3 + seed % (length - 3)
    alternate = raw.clone()
    alternate[:, cut:] = (alternate[:, cut:] + seed + 1) % 256
    original = world(raw)
    changed = world(alternate)
    truncated = world(raw[:, :cut], include_next=True)
    for a, b, c in zip(original, changed, truncated, strict=True):
        assert torch.equal(a[:, :cut + 1], b[:, :cut + 1]), seed
        assert torch.equal(a[:, :cut + 1], c), seed
    assert original[1].size(-1) == 5


def test_suffix_table_blends_a_causal_hit_and_ignores_conflicts():
    torch.manual_seed(1800)
    neural = ImportedSkill().eval()
    encounter = torch.tensor([[250, 251, 252, 255, 4, 4, 4, 6],
                              [250, 251, 252, 255, 4, 4, 4, 7]])
    skill = EpisodicAssistedBornSkill.from_encounter(neural, encounter)
    raw = torch.tensor([[250, 251, 252, 254, 4, 4, 4, 6]])
    with torch.inference_mode():
        original = neural(raw)
        assisted = skill(raw)
        probabilities = assisted[0, 2].softmax(-1)
        expected = original[0, 2].softmax(-1) * 0.1
        expected[255] += 0.9
    torch.testing.assert_close(probabilities, expected)
    assert probabilities.argmax() == 255
    assert torch.equal(assisted[:, 6], original[:, 6])
    assert torch.equal(assisted[:, :2], original[:, :2])


def test_birth_modes_preserve_all_resident_skill_hashes_and_forecasts(monkeypatch):
    torch.manual_seed(1810)
    world = make_bank()
    raw = torch.tensor([[176, 192, 208, 227] * 4])
    probe = torch.randint(256, (2, 13))
    residents = world.bank.skill_names
    before = [state_hash(world.bank.skills[name]) for name in residents]
    forecasts = world.bank.component_log_probs(probe)
    steps_used = []
    native_fit = skill_birth.adapt_skill

    def check_fit(candidate, observed, steps):
        assert torch.equal(observed, raw)
        steps_used.append(steps)
        native_fit(candidate, observed, 1)

    monkeypatch.setattr(skill_birth, "adapt_skill", check_fit)
    with forbid_donor_forward() as guard:
        episodic = world.spawn_and_adapt_skill("episodic", raw, mode=EPISODIC_BIRTH)
        neural = world.spawn_and_adapt_skill("consolidated", raw, mode=CONSOLIDATED_BIRTH)
        after = world.bank.component_log_probs(probe)
    assert guard.donor_forward_calls == 0
    assert steps_used == [32, 192]
    assert isinstance(episodic, EpisodicAssistedBornSkill)
    assert type(neural) is ImportedSkill
    assert before == [state_hash(world.bank.skills[name]) for name in residents]
    assert torch.equal(forecasts, after[:, :, :4])
    assert all(not parameter.requires_grad for parameter in world.parameters())


def test_episode_cold_restore_and_cli_preserve_bank_formats(tmp_path):
    torch.manual_seed(1811)
    world = make_bank()
    original = tmp_path / "legacy.pt"
    save_bank(world.bank, original)
    assert torch.load(original, weights_only=True)["schema"] == BANK_V1
    assert state_hash(load_bank(original)) == state_hash(world.bank)

    raw = torch.tensor([[255, 0, 254, 128] * 4])
    world.spawn_and_adapt_skill("skill_novel", raw, steps=2, mode=EPISODIC_BIRTH)
    path = tmp_path / "episodic.pt"
    world.save_bank(path)
    assert torch.load(path, weights_only=True)["schema"] == BANK_V2
    expected = world(raw, include_next=True)
    with forbid_donor_forward() as guard:
        cold = OpenWorldSkillBank.load_bank(path)
        assert cold.bank.skill_names == world.bank.skill_names
        assert state_hash(cold.bank) == state_hash(world.bank)
        assert all(torch.equal(a, b) for a, b in zip(expected, cold(raw, include_next=True), strict=True))
    assert guard.donor_forward_calls == 0

    command = [sys.executable, "-m", "nexus_expanse.cli", "route-stream", "--bank", str(path),
               "--router", "hmm_bma", "--input-hex", "ff00fe80ff00fe80"]
    output = json.loads(subprocess.check_output(command, text=True, timeout=60))
    assert output["prefix_max_abs_diff"] == 0.0
    assert output["donor_forward_calls"] == 0
    assert tuple(output["chunks"][0]["routing_weights"]) == cold.bank.skill_names


@pytest.mark.parametrize(("name", "pattern"), (
    ("rule_A", b")P}\x91"),
    ("markdown_prose", b"## Notes\n"),
))
def test_in_distribution_probes_do_not_birth_episodic_skills(name, pattern):
    world = make_bank()
    logits = torch.full((256,), -12.0)
    logits[list(set(pattern))] = 2.0
    world.bank.skills[name].forward = lambda raw: logits.expand(*raw.shape, -1)
    raw = torch.tensor([list(pattern * 20)])
    assert not world.detect_novelty(raw).item()
    assert world.observe("skill_novel", raw, mode=EPISODIC_BIRTH) is None
    assert len(world.bank.skills) == 4
