"""Offline checks for scaled native skills and static DeepSeek Engram extraction."""

from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from benchmarks.phase12_scaled_zero_exec import ARMS, SEEDS, STAGES, scaling_verdict
from nexus_expanse import shard_stream_assimilation as streaming
from nexus_expanse.deepseek_engram import _compressed_tokens, engram_rows
from nexus_expanse.donor_cache_io import GPT2_BYTES
from nexus_expanse.multi_skill_router import MultiSkillBank, load_bank, save_bank
from nexus_expanse.weight_editing import prefix_diff, restore_skill, verify_skill


ROOT = Path(__file__).resolve().parents[1]
BLOCK = 8
DONOR_WIDTH = 128


@pytest.fixture
def one_thread():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prior)


@pytest.fixture
def donor(tmp_path):
    root = tmp_path / "donor"
    root.mkdir()
    generator = torch.Generator().manual_seed(12)

    def random(*shape):
        return torch.randn(shape, generator=generator) * 0.2

    def quantized(state, name, rows, cols, *, rowwise=False):
        weights = random(rows, cols)
        shape = (rows if rowwise else (rows + BLOCK - 1) // BLOCK, (cols + BLOCK - 1) // BLOCK)
        state[name + ".weight"] = weights.to(torch.float8_e4m3fn)
        state[name + ".scale"] = torch.exp2(torch.randint(-2, 3, shape, generator=generator).float()).to(
            torch.float8_e8m0fnu,
        )

    states = {
        "embedding.pt": {"embed.weight": random(4200, DONOR_WIDTH).bfloat16()},
        "head.pt": {"head.weight": random(256, DONOR_WIDTH).bfloat16(), "norm.weight": torch.ones(DONOR_WIDTH)},
    }
    for index in range(4):
        state = {}
        prefix = f"layers.{index}."
        for name, rows, cols in (
            ("attn.wkv", 16, DONOR_WIDTH), ("attn.wq_a", 16, DONOR_WIDTH),
            ("attn.wq_b", 32, 16), ("attn.wo_a", 16, 32), ("attn.wo_b", DONOR_WIDTH, 16),
            ("ffn.shared_experts.w3", 24, DONOR_WIDTH), ("ffn.shared_experts.w2", DONOR_WIDTH, 24),
        ):
            quantized(state, prefix + name, rows, cols)
        states[f"layer-{index}.pt"] = state
    for index, rows in ((1, 18), (14, 30)):
        state = {}
        quantized(state, f"layers.{index}.engram.embed", rows, 16, rowwise=True)
        quantized(state, f"layers.{index}.engram.wkv", DONOR_WIDTH * 2, 32)
        states[f"engram-{index}.pt"] = state

    index = {}
    for name, state in states.items():
        torch.save(state, root / name)
        index.update(dict.fromkeys(state, name))
    tokenizer = json.dumps({
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": GPT2_BYTES},
    }).encode()
    _, compressed_size = _compressed_tokens(tokenizer)
    (root / "tokenizer.json").write_bytes(tokenizer)
    (root / "model.pt.index.json").write_text(json.dumps({"weight_map": index}))
    config = {
        "head_dim": 16, "hc_mult": 1, "engram_max_ngram_size": 2, "engram_n_heads": 2,
        "engram_head_dim": 16, "engram_vocab_size": 7, "engram_layer_ids": [1, 14],
        "engram_pad_token_id": 0, "engram_num_embeddings": [18, 30],
        "engram_compressed_vocab_size": compressed_size,
    }
    (root / "config.json").write_text(json.dumps({
        "text_config": config, "quantization_config": {"weight_block_size": [BLOCK, BLOCK]},
    }))
    return root


@pytest.mark.parametrize(("dim", "layers"), [(64, 4), (128, 2)])
@pytest.mark.parametrize("enabled", [False, True], ids=["no-engram", "engram"])
def test_scaled_fp8_skill_is_causal_across_50_probe_seeds(donor, one_thread, dim, layers, enabled):
    archive, receipt = streaming.assimilate_zero_exec(
        donor, mock_pt=True, student_dim=dim, student_layers=layers, use_engram=enabled,
    )
    assert receipt["student_width"] == dim and receipt["selected_layers"] == layers
    assert receipt["engram_layers"] == ([1, 14] if enabled else [])
    assert receipt["uses_static_engram"] is enabled
    assert receipt["loaded_shard_count"] == 2 + layers + 2 * enabled
    assert receipt["peak_loaded_shards"] == 1 and receipt["donor_forward_calls"] == 0
    assert receipt["prefix_max_abs_diff"] == receipt["round_trip_max_abs_diff"] == 0.0
    assert receipt["exact_round_trip_state"]
    student = restore_skill(archive)
    assert student.readout.in_features == dim and len(student.core.layers) == layers
    assert max(prefix_diff(student, seed=seed, probes=1) for seed in range(50)) == 0.0


def test_opt_in_streams_separate_engram_shards_and_changes_only_static_prior(donor, one_thread, monkeypatch):
    index = json.loads((donor / "model.pt.index.json").read_bytes())["weight_map"]
    visits = []

    def download(name, directory):
        out = directory / name
        if name == streaming.HF_INDEX:
            out.write_text(json.dumps({"weight_map": {
                key: Path(value).with_suffix(".safetensors").name for key, value in index.items()
            }}))
        elif name.endswith(".safetensors"):
            assert all(not previous.exists() for previous in visits)
            shutil.copyfile(donor / Path(name).with_suffix(".pt"), out)
            visits.append(out)
        else:
            shutil.copyfile(donor / name, out)
        return out

    @contextmanager
    def safe_open(path, **kwargs):
        weights = torch.load(BytesIO(Path(path).read_bytes()), weights_only=True)
        yield SimpleNamespace(get_tensor=weights.__getitem__)

    monkeypatch.setattr(streaming, "_hf_source", lambda _: (download, "a" * 40))
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=safe_open))
    plain, plain_receipt = streaming.assimilate_zero_exec(hf_repo="offline/fixture", student_dim=64)
    enabled, receipt = streaming.assimilate_zero_exec(hf_repo="offline/fixture", student_dim=64, use_engram=True)
    assert plain_receipt["loaded_shard_count"] == 4 and receipt["loaded_shard_count"] == 6
    assert receipt["peak_disk_shards"] == receipt["peak_loaded_shards"] == 1
    assert all(not path.exists() for path in visits)
    first, second = restore_skill(plain).state_dict(), restore_skill(enabled).state_dict()
    assert not torch.equal(first.pop("static_byte_prior"), second.pop("static_byte_prior"))
    assert all(torch.equal(value, second[key]) for key, value in first.items())


def test_scaled_cli_cold_restore(donor, tmp_path, one_thread):
    archive = tmp_path / "scaled.pt"
    output = subprocess.check_output([
        sys.executable, "-m", "nexus_expanse.cli", "assimilate-zero-exec", "--donor-dir", str(donor),
        "--mock-pt", "--student-dim", "64", "--student-layers", "4", "--use-engram", "--out", str(archive),
    ], cwd=ROOT, text=True)
    receipt = json.loads(output)
    model = restore_skill(archive)
    assert receipt["status"] == "PASS" and receipt["engram_layers"] == [1, 14]
    assert model.readout.in_features == 64 and len(model.core.layers) == 4
    assert prefix_diff(model) == 0.0


def test_largest_native_shape_cold_restores(one_thread):
    student = streaming.make_student(29, student_dim=256, student_layers=6).eval()
    archive, receipt = verify_skill(student)
    model = restore_skill(archive)
    assert model.readout.in_features == 256 and len(model.core.layers) == 6
    assert receipt["exact_round_trip_state"] and receipt["prefix_max_abs_diff"] == 0.0


def test_bank_archive_preserves_scaled_depth(tmp_path, one_thread):
    skill = streaming.make_student(1, student_dim=64, student_layers=4).eval()
    path = tmp_path / "bank.pt"
    save_bank(MultiSkillBank(skill, {}, base_name="scaled"), path)
    restored = load_bank(path).skills["scaled"]
    assert restored.readout.in_features == 64 and len(restored.core.layers) == 4
    assert prefix_diff(restored) == 0.0


def test_engram_prime_ranges_and_padding(donor):
    source = (donor / "tokenizer.json").read_bytes()
    config = json.loads((donor / "config.json").read_bytes())["text_config"]
    rows = engram_rows(source, config, torch.arange(256))
    for layer, (first, total) in {1: (7, 18), 14: (13, 30)}.items():
        ids, shape = rows[layer]
        assert shape == (total, 16) and ids.shape == (256, 2)
        assert (ids[:, 0] < first).all()
        assert ((ids[:, 1] >= first) & (ids[:, 1] < total)).all()
    # ASCII case folding sends equivalent singleton bytes to the same buckets.
    assert torch.equal(rows[1][0][ord("A")], rows[1][0][ord("a")])


@pytest.mark.parametrize(("control_wins", "baseline_wins", "expected"), [
    (5, 4, "PASS"), (4, 3, "MIXED"), (4, 4, "FAIL"), (3, 5, "FAIL"),
])
def test_frozen_rule_requires_the_same_arm_to_clear_both_stages(control_wins, baseline_wins, expected):
    rows = []
    for index, seed in enumerate(SEEDS):
        scores = dict.fromkeys(ARMS, (4.0, 4.0))
        scores["dim64_4L_engram"] = (3.0 if index < control_wins else 4.0, 3.0)
        scores["dim32_2L"] = (4.0, 3.5 if index < baseline_wins else 2.5)
        arms = {name: {"cold": {stage: {"bpb": {"combined": value}}
                                for stage, value in zip(STAGES, values, strict=True)}}
                for name, values in scores.items()}
        rows.append(dict(seed=seed, arms=arms))
    assert scaling_verdict(rows)["status"] == expected
