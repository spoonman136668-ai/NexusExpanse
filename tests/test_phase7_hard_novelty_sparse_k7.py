"""Causality, call accounting, cold routing and calibrated novelty checks."""
import json
import subprocess
import sys

import pytest
import torch

from nexus_expanse.checkpoint_transport import ImportedSkill
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.jepa import state_hash
from nexus_expanse.multi_skill_router import MultiSkillBank, load_bank, save_bank
from nexus_expanse.skill_birth import OpenWorldSkillBank


RESIDENT = ("base_python", "rule_A", "rule_B", "markdown_prose")
BORN = ("rule_C", "rule_D", "rule_E")


@pytest.fixture(scope="module", autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def bank_for(names):
    return MultiSkillBank(ImportedSkill(), {name: ImportedSkill() for name in names},
                          resident_names=RESIDENT)


@pytest.mark.parametrize("seed", range(50))
def test_sparse_hierarchical_k7_has_exact_prefix_causality(seed):
    torch.manual_seed(seed)
    bank = bank_for((*RESIDENT[1:], *BORN))
    length = 70 if seed == 49 else 12 + seed % 7
    raw = torch.randint(0, 256, (2, length), dtype=torch.uint8)
    cut = length - 2 if seed == 49 else 1 + seed % (length - 1)
    changed = raw.clone()
    changed[:, cut:] = (changed[:, cut:].long() + seed + 1) % 256
    route = bank.forward_hmm_sparse_hierarchical
    first, second = route(raw), route(changed)
    short = route(raw[:, :cut], include_next=True)
    for a, b, c in zip(first[:2], second[:2], short[:2], strict=True):
        assert torch.equal(a[:, :cut + 1], b[:, :cut + 1]), seed
        assert torch.equal(a[:, :cut + 1], c), seed


def alphabet_predictor(model, alphabet):
    logits = torch.full((256,), -6.0)
    logits[list(alphabet)] = 6.0
    model.forward = lambda raw: logits.expand(*raw.shape, -1)


def test_sparse_uses_actual_top1_calls_between_k7_probes():
    bank = bank_for((*RESIDENT[1:], *BORN))
    for skill in bank.skills.values():
        alphabet_predictor(skill, {42})
    calls = [0] * len(bank.skills)

    def counter(slot):
        def count(_module, inputs):
            calls[slot] += inputs[0].size(0)
        return count

    handles = [skill.register_forward_pre_hook(counter(i)) for i, skill in enumerate(bank.skills.values())]
    try:
        logp, weights, average = bank.forward_hmm_sparse_hierarchical(torch.full((2, 80), 42))
    finally:
        for handle in handles:
            handle.remove()
    assert (logp[:, 1:].argmax(-1) == 42).all()
    assert (weights[:, 0, :4] > weights[:, 0, 4:].max()).all()
    # The uniform first forecast probes at 1. Periodic probes occur at 24, 48 and 72.
    assert sum(calls) == 2 * (79 + 4 * 6)
    assert average == sum(calls) / 160 < 2.8


def test_cold_bank_and_sparse_cli_are_exact(tmp_path):
    torch.manual_seed(2000)
    bank = bank_for((*RESIDENT[1:], *BORN))
    path = tmp_path / "k7.pt"
    save_bank(bank, path)
    raw = torch.tensor([[255, 0, 253, 252, 128, 17]])
    with forbid_donor_forward() as guard:
        cold = load_bank(path)
        expected = bank.forward_hmm_sparse_hierarchical(raw, include_next=True)
        actual = cold.forward_hmm_sparse_hierarchical(raw, include_next=True)
    assert cold.resident_names == RESIDENT
    assert state_hash(cold) == state_hash(bank)
    assert all(torch.equal(a, b) for a, b in zip(expected[:2], actual[:2], strict=True))
    assert expected[2] == actual[2]
    assert guard.donor_forward_calls == 0

    command = [sys.executable, "-m", "nexus_expanse.cli", "route-stream", "--bank", str(path),
               "--router", "hmm_sparse_hierarchical", "--input-hex", "ff00fd", "--input-hex", "fc8011"]
    report = json.loads(subprocess.check_output(command, text=True, timeout=60))
    assert report["router"] == "hmm_sparse_hierarchical"
    assert report["prefix_max_abs_diff"] == 0.0 and report["donor_forward_calls"] == 0
    assert report["avg_active_skills_per_byte"] == expected[2]
    for chunk, end in zip(report["chunks"], (3, 6), strict=True):
        assert chunk["end_offset"] == end
        assert chunk["next_byte"] == int(expected[0][0, end].argmax())
        assert chunk["routing_weights"] == pytest.approx(dict(zip(bank.skill_names, expected[1][0, end].tolist())))


def rule_predictor(model, offset):
    alphabet = [*range(40, 48), 80, 125, *range(97, 105)]
    background = torch.full((256,), -6.0)
    background[alphabet] = 6.0
    answer = torch.full((8, 256), -6.0)
    for cue in range(8):
        answer[cue, 97 + (cue + offset) % 8] = 6.0

    def predict(raw):
        logits = background.expand(*raw.shape, -1).clone()
        if raw.size(1) >= 3:
            key = (raw[:, :-2] - 40).clamp(0, 7)
            logits[:, 2:] = torch.where((raw[:, 2:] == 125).unsqueeze(-1), answer[key], logits[:, 2:])
        return logits

    model.forward = predict


def rule_stream(offset, start):
    return bytes(value for step in range(32) for key in ((step + start) % 8,)
                 for value in (40 + key, 80, 125, 97 + (key + offset) % 8))


def test_tail_limits_fit_only_id_and_detect_same_alphabet_rule():
    world = OpenWorldSkillBank(bank_for(RESIDENT[1:]))
    py = b"def fn " * 20
    md = b"# Notes\n" * 16
    alphabet_predictor(world.bank.base, set(py))
    alphabet_predictor(world.bank.skills["markdown_prose"], set(md))
    rule_predictor(world.bank.skills["rule_A"], 0)
    rule_predictor(world.bank.skills["rule_B"], 1)

    def tensor(raw):
        return torch.tensor([list(raw)])

    validation = dict(base_python=tensor(py), rule_A=tensor(rule_stream(0, 0)),
                      rule_B=tensor(rule_stream(1, 0)), markdown_prose=tensor(md))
    calibration = world.calibrate_tail_surprise(validation)
    negatives = (py[4:], rule_stream(0, 3), rule_stream(1, 4), md[3:] + b"#")
    for stream in negatives:
        assert world.observe("candidate", tensor(stream), tail=calibration) is None
    assert len(world.bank.skills) == 4
    novel = rule_stream(2, 6)
    assert set(novel) == set(rule_stream(0, 3)) == set(rule_stream(1, 4))
    assert not world.detect_novelty(tensor(novel)).item()
    assert world.detect_novelty(tensor(novel), tail=calibration).item()
    assert not world.detect_novelty(tensor(novel[:64]), tail=calibration).item()
