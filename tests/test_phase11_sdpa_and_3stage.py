"""Opt-in SDPA parity and exact raw-byte assimilation, edit, birth and routing."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.nn import functional as F

from benchmarks.phase10b_shard_stream_and_edit import write_mock_shards
from benchmarks.phase11_sdpa_and_3stage import bank_prefix_diff, core_prefix_diff, make_sdpa_model
from nexus_expanse.checkpoint_transport import ImportedSkill
from nexus_expanse.jepa import state_hash
from nexus_expanse.multi_skill_router import MultiSkillBank, load_bank, save_bank
from nexus_expanse.shard_stream_assimilation import assimilate_zero_exec
from nexus_expanse.skill_birth import EPISODIC_BIRTH, OpenWorldSkillBank
from nexus_expanse.weight_editing import apply_surgical_edit, restore_skill, verified_surgical_edit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("context", [32, 128, 512])
def test_same_weights_sdpa_matches_legacy_train_and_eval(context, one_thread):
    torch.manual_seed(11011)
    legacy = make_sdpa_model(context, use_sdpa=False, dim=32)
    sdpa = make_sdpa_model(context, use_sdpa=True, dim=32)
    sdpa.load_state_dict(legacy.state_dict(), strict=True)
    raw = torch.randint(256, (1, context))
    a, b = legacy.forward_neural(raw)[:2], sdpa.forward_neural(raw)[:2]
    assert max((left - right).abs().max().item() for left, right in zip(a, b, strict=True)) <= 1e-5
    legacy.eval()
    sdpa.eval()
    with torch.inference_mode():
        a, b = legacy.forward_neural(raw)[:2], sdpa.forward_neural(raw)[:2]
        assert max((left - right).abs().max().item() for left, right in zip(a, b, strict=True)) <= 1e-5


def test_sdpa_exact_prefix_across_50_seeds(one_thread):
    for seed in range(50):
        torch.manual_seed(seed)
        geometry = (32, 128, 512)[seed % 3]
        model = make_sdpa_model(geometry, use_sdpa=True, dim=32).eval()
        raw = torch.randint(256, (2, geometry))
        assert core_prefix_diff(model, raw, 1 + seed % (geometry - 1)) == 0.0


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_train_archive_state_dict_is_interchangeable(tmp_path, one_thread, use_sdpa):
    model = make_sdpa_model(32, use_sdpa=use_sdpa, dim=32)
    raw = torch.arange(32).unsqueeze(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    outputs = model.forward_neural(raw)[0]
    F.cross_entropy(outputs.flatten(0, 1), (raw + 1).flatten()).backward()
    optimizer.step()
    path = tmp_path / "state.pt"
    torch.save(model.state_dict(), path)
    opposite = make_sdpa_model(32, use_sdpa=not use_sdpa, dim=32)
    opposite.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    assert state_hash(opposite) == state_hash(model)
    assert set(model.state_dict()) == set(opposite.state_dict())
    assert torch.equal(model.forward_neural(raw)[0], opposite.forward_neural(raw)[0])


def test_sdpa_skips_core_mask_and_honors_explicit_layer_mask(monkeypatch, one_thread):
    model = make_sdpa_model(32, use_sdpa=True, dim=32)
    hidden, mask = model._backbone_inputs(torch.arange(32).unsqueeze(0))
    assert mask is None
    legacy = make_sdpa_model(32, use_sdpa=False, dim=32)
    legacy.load_state_dict(model.state_dict())
    real = F.scaled_dot_product_attention

    def attention(q, k, v, *args, **kwargs):
        assert not args and kwargs["is_causal"] is True
        assert "attn_mask" not in kwargs
        return real(q, k, v, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(F, "scaled_dot_product_attention", attention)
        model.layers[0].attention_residual(hidden, None)
    custom_mask = torch.ones(32, 32, dtype=torch.bool).tril(-1)
    assert torch.equal(
        model.layers[0].attention_residual(hidden, custom_mask),
        legacy.layers[0].attention_residual(hidden, custom_mask),
    )


@pytest.fixture
def donor_shards(tmp_path):
    rng = torch.Generator().manual_seed(2500)

    def weight(*shape):
        return 0.1 * torch.randn(shape, generator=rng)

    shards = [{"model.embed_tokens.weight": weight(320, 64)}, {"lm_head.weight": weight(320, 64)}, {}]
    for i in range(2):
        root = f"model.layers.{i}."
        shards[i + 1].update({
            root + "self_attn.v_proj.weight": weight(64, 64), root + "self_attn.o_proj.weight": weight(64, 64),
            root + "mlp.up_proj.weight": weight(96, 64), root + "mlp.down_proj.weight": weight(64, 96),
        })
    vocab = [bytes([byte]) for byte in range(256)] + [bytes([42, byte]) for byte in range(64)]
    path = tmp_path / "donor"
    write_mock_shards(path, shards, vocab)
    return path


def small_residents(base):
    return MultiSkillBank(base, {name: ImportedSkill() for name in ("rule_A", "rule_B", "markdown_prose")})


def test_exact_three_stage_bank_prefix_across_50_seeds(donor_shards, tmp_path, one_thread):
    path = tmp_path / "expanded.pt"
    for seed in range(50):
        torch.manual_seed(seed)
        archive, assimilation = assimilate_zero_exec(donor_shards, mock_pt=True, seed=seed)
        assert assimilation["donor_forward_calls"] == 0
        assert assimilation["peak_loaded_shards"] == 1
        base = restore_skill(archive)
        v = torch.eye(32)[seed % 32]
        layer = base.core.layers[1]
        before = float(layer.attn.out_proj.bias[seed % 32])

        def edit(skill):
            return apply_surgical_edit(skill, 1, "attn.out_proj", "bias_shift", v, 0.1)

        def fault(skill):
            assert not torch.is_grad_enabled()
            return 5 + before - float(skill.core.layers[1].attn.out_proj.bias[seed % 32])

        accepted, repaired, receipt = verified_surgical_edit(base, edit, fault, lambda _: 5.0)
        assert accepted and receipt["exact_round_trip_state"]
        assert receipt["round_trip_max_abs_diff"] == receipt["prefix_max_abs_diff"] == 0.0
        world = OpenWorldSkillBank(small_residents(repaired))
        resident = state_hash(world.bank.base)
        observed = torch.tensor([[176, 192, 208, 224 + seed % 8] * 3])
        # A permissive threshold isolates composition in this synthetic plumbing test.
        assert world.observe("rule_C", observed, mode=EPISODIC_BIRTH, window=4,
                             min_bpb_threshold=1.0, steps=1) is not None
        world.save_bank(path)
        cold = OpenWorldSkillBank.load_bank(path).bank
        assert len(cold.skills) == 5 and state_hash(cold.base) == resident
        raw = torch.cat((observed[:, :6], torch.randint(256, (1, 3))), dim=1)
        for before_restore, after_restore in zip(world.bank.forward_hmm_bma_hierarchical(raw),
                                                  cold.forward_hmm_bma_hierarchical(raw), strict=True):
            assert torch.equal(before_restore, after_restore)
        assert bank_prefix_diff(cold, raw, 1 + seed % (raw.shape[1] - 1)) == 0.0


def invoke(*args):
    done = subprocess.run([sys.executable, "-m", "nexus_expanse.cli", *map(str, args)], cwd=ROOT,
                          text=True, capture_output=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")]


def test_fresh_process_assimilate_edit_route(donor_shards, tmp_path, one_thread):
    assimilated = tmp_path / "assimilation.pt"
    edited = tmp_path / "edited.pt"
    (initial,) = invoke("assimilate-zero-exec", "--donor-dir", donor_shards, "--mock-pt", "--out", assimilated)
    assert initial["donor_forward_calls"] == 0 and initial["peak_loaded_shards"] == 1
    (repair,) = invoke("weight-edit", "--skill", assimilated, "--op", "subspace_ablate",
                       "--layer", 1, "--alpha", 0.5, "--out", edited)
    assert repair["exact_round_trip_state"] and repair["prefix_max_abs_diff"] == 0.0
    skill = restore_skill(edited)
    bank = small_residents(skill)
    path = tmp_path / "bank.pt"
    save_bank(bank, path)
    restored = load_bank(path)
    assert restored.base.prior_alpha == 0.25 and restored.base.core.fixed_eval_geometry
    assert state_hash(restored.base) == state_hash(skill)
    (routed,) = invoke("route-stream", "--bank", path, "--router", "hmm_bma_hierarchical",
                       "--input-hex", "00010203", "--input-hex", "fdfeff")
    assert routed["prefix_max_abs_diff"] == 0.0 and routed["donor_forward_calls"] == 0
    assert set(routed["chunks"][-1]["routing_weights"]) == set(bank.skill_names)


def test_cli_sdpa_long_context_training(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i, name in enumerate(("math", "code", "prose")):
        for j, split in enumerate(("train", "eval")):
            (corpus / f"{name}-{split}.bin").write_bytes(bytes((n + i * 3 + j) % 256 for n in range(560)))
    run = tmp_path / "train"
    archive = tmp_path / "deployment.pt"
    report = invoke("train", "--directory", run, "--corpus-dir", corpus, "--device", "cpu",
                    "--use-sdpa", "--context", 520, "--dim", 32, "--epochs", 1, "--steps-per-epoch", 1,
                    "--batch-per-domain", 1, "--eval-blocks", 1, "--save-snapshot", archive)
    assert all(value == 0.0 for value in report[-1]["parity"].values())
    receipt = json.loads((run / "gate17.json").read_bytes())
    assert receipt["identity"]["use_sdpa"] is True and receipt["identity"]["geometry"]["context"] == 520
    assert receipt["cold_equal"] and archive.is_file()
