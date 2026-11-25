"""Exact retention, causality, novelty controls and expanded snapshot checks."""
import pytest
import torch

from nexus_expanse.checkpoint_transport import ImportedSkill
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.jepa import state_hash
from nexus_expanse.multi_skill_router import MultiSkillBank, load_bank
from nexus_expanse.skill_birth import OpenWorldSkillBank


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


def test_birth_preserves_every_resident_hash_and_forecast():
    torch.manual_seed(1700)
    world = make_bank()
    probe = torch.randint(256, (2, 14))
    old = {name: state_hash(skill) for name, skill in world.bank.skills.items()}
    before = world.bank.component_log_probs(probe)
    training = torch.tensor([[176, 192, 208, 224, 177, 192, 208, 225] * 3])
    with forbid_donor_forward() as guard:
        born = world.spawn_and_adapt_skill("skill_novel", training, steps=4)
    assert guard.donor_forward_calls == 0
    assert world.bank.skills["skill_novel"] is born
    assert len(world.bank.skills) == 5
    assert state_hash(born) != old[world.bank.base_name]
    assert all(state_hash(world.bank.skills[name]) == expected for name, expected in old.items())
    assert torch.equal(world.bank.component_log_probs(probe)[:, :, :4], before)
    assert all(not parameter.requires_grad for parameter in world.parameters())


@pytest.mark.parametrize("seed", range(50))
def test_expanded_hmm_has_exact_prefix_causality(seed):
    torch.manual_seed(seed)
    world = make_bank()
    world.spawn_and_adapt_skill("skill_novel", torch.randint(256, (1, 10)), steps=1)
    length = 70 if seed == 49 else 10 + seed % 7
    raw = torch.randint(256, (2, length), dtype=torch.uint8)
    cut = length - 2 if seed == 49 else 1 + seed % (length - 1)
    changed = raw.clone()
    changed[:, cut:] = (changed[:, cut:].long() + 1 + seed) % 256
    full = world(raw)
    mutated = world(changed)
    truncated = world(raw[:, :cut], include_next=True)
    for output, other, short in zip(full, mutated, truncated, strict=True):
        assert torch.equal(output[:, :cut + 1], other[:, :cut + 1])
        assert torch.equal(output[:, :cut + 1], short)
    assert full[1].shape[-1] == 5


def recognizer(world, name, alphabet):
    logits = torch.full((256,), -12.0)
    logits[list(alphabet)] = 2.0

    def forecast(raw):
        return logits.expand(*raw.shape, -1)

    world.bank.skills[name].forward = forecast


@pytest.mark.parametrize(("name", "pattern"), (
    ("rule_A", b")P}\x91"),
    ("markdown_prose", b"## Notes\n"),
))
def test_resident_rule_and_markdown_do_not_give_false_birth(name, pattern):
    world = make_bank()
    recognizer(world, name, set(pattern))
    raw = torch.tensor([list(pattern * 20)])
    assert not world.detect_novelty(raw, window=16).item()
    assert world.observe("skill_novel", raw, window=16, steps=1) is None
    assert len(world.bank.skills) == 4


def test_observation_trigger_needs_a_completed_window_and_runs_once():
    torch.manual_seed(1720)
    world = make_bank()
    raw = torch.full((1, 12), 255)
    for name in tuple(world.bank.skill_names)[1:]:
        recognizer(world, name, {0})
    with torch.no_grad():
        base = world.bank.base
        base.core.pos_emb.weight.zero_()
        hidden = base.core.forward_neural(raw[:, :1])[1][0, 0]
        base.readout.weight.zero_()
        base.readout.weight[0] = 16 * hidden
    assert world.observe("skill_novel", raw[:, :8], window=8, steps=1) is None
    assert world.detect_novelty(raw, window=8).item()
    born = world.observe("skill_novel", raw, window=8, steps=1)
    assert born is world.bank.skills["skill_novel"]
    assert world.observe("skill_novel", raw, window=8, steps=1) is None
    assert len(world.bank.skills) == 5


def test_expanded_bank_cold_restore_uses_existing_format(tmp_path):
    torch.manual_seed(1721)
    world = make_bank()
    world.spawn_and_adapt_skill("skill_novel", torch.randint(256, (1, 24)), steps=3)
    raw = torch.randint(256, (2, 16))
    path = tmp_path / "expanded.pt"
    world.save_bank(path)
    expected = world(raw, include_next=True)
    for restored in (OpenWorldSkillBank.load_bank(path).bank, load_bank(path)):
        assert restored.skill_names == world.bank.skill_names
        assert state_hash(restored) == state_hash(world.bank)
        for a, b in zip(expected, restored.forward_hmm_bma(raw, include_next=True), strict=True):
            assert torch.equal(a, b)


def test_failed_birth_leaves_bank_intact():
    world = make_bank()
    original = state_hash(world.bank)
    with pytest.raises(ValueError, match="at least two"):
        world.spawn_and_adapt_skill("skill_novel", torch.tensor([[10]]))
    with pytest.raises(ValueError, match="unique"):
        world.spawn_and_adapt_skill("rule_A", torch.tensor([[10, 11]]))
    assert len(world.bank.skills) == 4
    assert state_hash(world.bank) == original
