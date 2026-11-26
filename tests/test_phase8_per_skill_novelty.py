"""Calibration, exact prefixes, call budgets and the cold CLI contract."""
import json
import subprocess
import sys

import pytest
import torch

from nexus_expanse.checkpoint_transport import ImportedSkill
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.jepa import state_hash
from nexus_expanse.multi_skill_router import MultiSkillBank, load_bank, save_bank
from nexus_expanse.skill_birth import OpenWorldSkillBank, PerSkillNoveltyDetector


RESIDENT = ("base_python", "rule_A", "rule_B", "markdown_prose")
BORN = ("rule_C", "rule_D", "rule_E")


@pytest.fixture(scope="module", autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def bank_for(born=True, limits=True):
    names = (*RESIDENT, *BORN) if born else RESIDENT
    return MultiSkillBank(
        ImportedSkill(), {name: ImportedSkill() for name in names[1:]}, resident_names=RESIDENT,
        byte_nll_q99={name: 12.0 for name in names} if limits else None,
    )


def alphabet(model, allowed, peak=6.0):
    logits = torch.full((256,), -peak)
    logits[list(allowed)] = peak
    model.forward = lambda raw: logits.expand(*raw.shape, -1)


@pytest.mark.parametrize("seed", range(50))
def test_conf_sparse_k7_has_exact_prefix_causality(seed):
    torch.manual_seed(seed)
    bank = bank_for()
    # Some seeds use a dominant expert so the check also runs sparse stretches.
    if seed % 5 == 0:
        for i, expert in enumerate(bank.skills.values()):
            alphabet(expert, {42 + i}, 60)
    length = 70 if seed == 49 else 12 + seed % 7
    raw = (torch.full((2, length), 42, dtype=torch.uint8) if seed % 5 == 0 else
           torch.randint(0, 256, (2, length), dtype=torch.uint8))
    cut = length - 2 if seed == 49 else 1 + seed % (length - 1)
    changed = raw.clone()
    changed[:, cut:] = (changed[:, cut:].long() + seed + 1) % 256
    route = bank.forward_hmm_conf_sparse_hierarchical
    first, second = route(raw), route(changed)
    short = route(raw[:, :cut], include_next=True)
    for a, b, c in zip(first[:2], second[:2], short[:2], strict=True):
        assert torch.equal(a[:, :cut + 1], b[:, :cut + 1]), seed
        assert torch.equal(a[:, :cut + 1], c), seed


def test_measured_steady_calls_and_next_step_anomaly_probe():
    bank = bank_for()
    for i, expert in enumerate(bank.skills.values()):
        alphabet(expert, {42 + i}, 60)
    calls = [0] * 80

    def note(_model, inputs):
        calls[inputs[0].size(1)] += inputs[0].size(0)

    hooks = [skill.register_forward_pre_hook(note) for skill in bank.skills.values()]
    try:
        raw = torch.full((2, 80), 42)
        raw[:, 11] = 43
        _, weights, actual = bank.forward_hmm_conf_sparse_hierarchical(raw)
    finally:
        for handle in hooks:
            handle.remove()
    # Calls in the first context window have their forecast index as prefix width.
    assert calls[1] == calls[12] == 14
    assert calls[2] == calls[10] == calls[11] == 2
    assert weights[:, 12].argmax(-1).eq(1).all()
    assert actual == sum(calls) / raw.numel() < 2.8


def test_low_confidence_mixture_and_missing_calibration_match_dense():
    torch.manual_seed(2100)
    raw = torch.randint(0, 256, (2, 16))
    for limits in (True, False):
        bank = bank_for(limits=limits)
        logp, weights = bank.forward_hmm_bma_hierarchical(raw)
        actual = bank.forward_hmm_conf_sparse_hierarchical(raw, conf_threshold=1.0)
        assert torch.equal(logp, actual[0])
        assert torch.equal(weights, actual[1])
        assert actual[2] == 7 * 15 / 16


def test_q99_cold_snapshot_and_cli(tmp_path):
    torch.manual_seed(2101)
    bank = bank_for(limits=False)
    val = {name: torch.randint(0, 256, (3, 21)) for name in bank.skill_names}
    limits = bank.calibrate_byte_nll_q99(val)
    for index, name in enumerate(bank.skill_names):
        raw = val[name]
        forecast = bank.component_log_probs(raw, selected=raw.new_full(raw.shape, index))
        nll = -forecast.gather(-1, raw.unsqueeze(-1))[:, 1:].flatten()
        assert limits[name] == float(nll.max())

    path = tmp_path / "calibrated.pt"
    save_bank(bank, path)
    raw = torch.tensor([[255, 0, 253, 252, 128, 17]])
    with forbid_donor_forward() as guard:
        cold = load_bank(path)
        expected = bank.forward_hmm_conf_sparse_hierarchical(raw, include_next=True)
        actual = cold.forward_hmm_conf_sparse_hierarchical(raw, include_next=True)
    assert cold.byte_nll_q99 == limits
    assert cold.resident_names == RESIDENT and state_hash(cold) == state_hash(bank)
    assert all(torch.equal(a, b) for a, b in zip(expected[:2], actual[:2], strict=True))
    assert expected[2] == actual[2]
    assert guard.donor_forward_calls == 0

    command = [sys.executable, "-m", "nexus_expanse.cli", "route-stream", "--bank", str(path),
               "--router", "hmm_conf_sparse_hierarchical", "--input-hex", "ff00fd", "--input-hex", "fc8011"]
    report = json.loads(subprocess.check_output(command, text=True, timeout=60))
    assert report["prefix_max_abs_diff"] == 0.0 and report["donor_forward_calls"] == 0
    assert report["router"] == "hmm_conf_sparse_hierarchical"
    assert report["avg_active_skills_per_byte"] == expected[2]
    for chunk, end in zip(report["chunks"], (3, 6), strict=True):
        assert chunk["next_byte"] == int(expected[0][0, end].argmax())
        assert chunk["routing_weights"] == pytest.approx(dict(zip(bank.skill_names, expected[1][0, end].tolist())))


def rule_fixture(model, offset):
    pattern = [*range(40, 48), 80, 125, *range(97, 105)]
    generic = torch.full((256,), -6.0)
    generic[pattern] = 6.0
    answers = torch.full((8, 256), -6.0)
    for cue in range(8):
        answers[cue, 97 + (cue + offset) % 8] = 6.0

    def predict(raw):
        logits = generic.expand(*raw.shape, -1).clone()
        if raw.size(1) >= 3:
            key = (raw[:, :-2] - 40).clamp(0, 7)
            logits[:, 2:] = torch.where((raw[:, 2:] == 125).unsqueeze(-1), answers[key], logits[:, 2:])
        return logits

    model.forward = predict


def ascii_rule(offset, start):
    return bytes(value for step in range(32) for key in ((step + start) % 8,)
                 for value in (40 + key, 80, 125, 97 + (key + offset) % 8))


def test_per_skill_detector_uses_own_ceilings_with_zero_id_flags():
    world = OpenWorldSkillBank(bank_for(born=False))
    py, md = b"def fn " * 20, b"# Notes\n" * 16
    alphabet(world.bank.base, set(py))
    alphabet(world.bank.skills["markdown_prose"], set(md))
    rule_fixture(world.bank.skills["rule_A"], 0)
    rule_fixture(world.bank.skills["rule_B"], 1)

    def tensor(raw):
        return torch.tensor([list(raw)])

    val = dict(base_python=tensor(py), rule_A=tensor(ascii_rule(0, 0)), rule_B=tensor(ascii_rule(1, 0)),
               markdown_prose=tensor(md))
    detector = PerSkillNoveltyDetector.calibrate(world.bank, val)
    assert len(set(detector.ceilings_bpb)) > 1
    negatives = (py[4:], ascii_rule(0, 3), ascii_rule(1, 4), md[3:] + b"#")
    for stream in negatives:
        assert world.observe("candidate", tensor(stream), per_skill=detector) is None
    assert world.bank.skill_names == RESIDENT
    novel = tensor(ascii_rule(2, 6))
    assert not world.detect_novelty(novel).item()
    flag, bpb, routed, ceiling = detector.assess(world.bank, novel)
    assert routed.item() in (1, 2)
    assert flag.item() and bpb.item() > ceiling.item()
    assert world.detect_novelty(novel, per_skill=detector).item()
    assert not world.detect_novelty(novel[:, :64], per_skill=detector).item()
