"""Offline checks for scheduled training, scaled snapshots, and byte sampling."""

import hashlib
import json
import math
import random
import sys

import numpy as np
import pytest
import torch

from benchmarks import gate17_gpu_scaling_harness as scaling
from benchmarks.phase13_warmstart_train import cli
from benchmarks.phase15_cosine_lr_and_sampling import call_cli, space_snapshot
from nexus_expanse.cli import main, read_snapshot, repetition_adjusted_logs
from nexus_expanse.jepa import state_hash
from nexus_expanse.patching import CompactBigram
from nexus_expanse.unified import NexusExpanseUnified


@pytest.fixture(autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    rng = random.Random(15)
    for domain in scaling.DOMAINS:
        for split in ("train", "eval"):
            (root / f"{domain}-{split}.bin").write_bytes(rng.randbytes(96))
    return root


def test_cosine_lr_warmup_midpoint_and_endpoint():
    base = 0.004
    rates = [scaling.cosine_lr(step, 100, base) for step in range(1, 101)]
    assert rates[:5] == [base * fraction for fraction in (.2, .4, .6, .8, 1.)]
    assert rates[-1] == 0.1 * base
    assert all(a > b for a, b in zip(rates[4:], rates[5:]))
    assert scaling.cosine_lr(3, 5, base) == pytest.approx(.55 * base)
    assert scaling.cosine_lr(1, 1, base) == 0.1 * base


def test_cli_cosine_cold_resume_rejects_changed_lr_and_schedule(tmp_path, corpus):
    resumed = tmp_path / "resumed"
    direct = tmp_path / "direct"
    common = ("train", "--corpus-dir", corpus, "--device", "cpu", "--dim", 32, "--context", 16,
              "--epochs", 2, "--steps-per-epoch", 20, "--batch-per-domain", 1, "--eval-blocks", 1,
              "--lr", 0.004, "--lr-schedule", "cosine")
    cli(*common, "--directory", resumed, "--epoch-limit", 1)
    first_path = resumed / "gate17-epoch-01.json"
    committed = first_path.read_bytes()
    first = json.loads(committed)
    assert first["identity"]["lr"] == 0.004 and first["identity"]["lr_schedule"] == "cosine"
    for override in (("--lr", 0.005), ("--lr-schedule", "constant")):
        with pytest.raises(RuntimeError, match="epoch identity or hash mismatch"):
            cli(*common, "--directory", resumed, *override)
        assert not (resumed / "gate17-epoch-02.intent.json").exists()
        assert first_path.read_bytes() == committed
    cli(*common, "--directory", resumed, "--epoch-limit", 1)
    cli(*common, "--directory", direct, "--epoch-limit", 2)
    a, b = (json.loads((root / "gate17.json").read_bytes()) for root in (resumed, direct))
    assert a["cold_equal"] and b["cold_equal"] and a["final_hash"] == b["final_hash"]
    assert a["donor_forward_calls"] == b["donor_forward_calls"] == 0
    assert set(a["fresh_evaluation"]["parity"].values()) == {0.0}
    assert len({e["telemetry"]["worker_pid"] for e in a["epochs"]}) == 2
    assert a["epochs"][1]["telemetry"]["optimizer_steps_at_start"] == 20
    assert [e["sha256"] for e in a["epochs"]] == [e["sha256"] for e in b["epochs"]]
    rates = [rate for e in a["epochs"] for rate in e["telemetry"]["learning_rates"]]
    assert rates == [scaling.cosine_lr(step, 40, 0.004) for step in range(1, 41)]


def test_scaled_training_geometry_and_512x8_snapshot(tmp_path, corpus, monkeypatch):
    observed = []

    def collect(spec, args, geometry, backbone, *_args, **_kwargs):
        assert args.lr_schedule == "constant" and args.lr == 0.003
        assert "lr" not in spec.identity and "lr_schedule" not in spec.identity
        observed.append((geometry, backbone))

    monkeypatch.setattr(scaling, "git", lambda *args: b"" if args[0] == "status" else b"phase15-test")
    monkeypatch.setattr(scaling, "run_slice", collect)
    main(["train", "--corpus-dir", str(corpus), "--directory", str(tmp_path / "scaled"), "--device", "cpu",
          "--dim", "512", "--layers", "8", "--context", "4096", "--use-sdpa", "--skill-backbone",
          "--steps-per-epoch", "256", "--batch-per-domain", "16"])
    (geometry, kind), = observed
    assert (geometry.dim, geometry.layers, geometry.steps_per_epoch, geometry.batch_per_domain) == (512, 8, 256, 16)
    model = scaling.model_for(geometry, kind, torch.device("cpu"))
    with torch.inference_mode():
        logits, hidden = model.forward_neural(torch.arange(8)[None])[:2]
    assert logits.shape == (1, 8, 256) and hidden.shape == (1, 8, 512)
    assert torch.isfinite(logits).all()
    counts = CompactBigram.from_dense(np.zeros((256, 256), dtype="<u4"))
    unified = NexusExpanseUnified(model, counts)
    model_hash = state_hash(model)
    raw = unified.snapshot()
    assert 32_000_000 < len(raw) <= NexusExpanseUnified.SNAPSHOT_LIMIT
    path = tmp_path / "scaled.pt"
    path.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    del raw, unified, model
    restored = NexusExpanseUnified.restore(read_snapshot(path, NexusExpanseUnified.SNAPSHOT_LIMIT, expected))
    assert restored.max_seq_len == 4096 and len(restored.backbone.layers) == 8
    assert state_hash(restored.backbone) == model_hash


@pytest.mark.parametrize("flag,value", [("--dim", "513"), ("--dim", "48"), ("--layers", "10"),
                                        ("--steps-per-epoch", "0"), ("--steps-per-epoch", "1025"),
                                        ("--batch-per-domain", "33"), ("--lr", "0"), ("--lr", "nan")])
def test_train_rejects_invalid_geometry_and_rate(flag, value, tmp_path, corpus):
    with pytest.raises(SystemExit) as exc:
        main(["train", "--corpus-dir", str(corpus), "--directory", str(tmp_path / "invalid"), flag, value])
    assert exc.value.code == 2 and not (tmp_path / "invalid").exists()


def test_snapshot_reader_retains_a_hard_size_limit(tmp_path):
    path = tmp_path / "oversized.pt"
    with path.open("wb") as output:
        output.truncate(NexusExpanseUnified.SNAPSHOT_LIMIT + 1)
    with pytest.raises(ValueError, match="size limit"):
        read_snapshot(path, NexusExpanseUnified.SNAPSHOT_LIMIT)


def test_repetition_log_scaling_reduces_recent_mass_and_handles_full_window():
    probabilities = torch.tensor([.28, .42, .30], dtype=torch.float64)
    adjusted = repetition_adjusted_logs(probabilities, [1, 1], 2.0)
    assert adjusted[1] == 2 * math.log(.42)
    assert adjusted.softmax(0)[1] < probabilities[1] and int(adjusted.argmax()) == 2
    extreme = repetition_adjusted_logs(probabilities, range(3), sys.float_info.max)
    assert torch.equal(extreme.softmax(0), torch.tensor([0., 1., 0.], dtype=torch.float64))


def test_generate_penalty_one_matches_stream_and_breaks_space_collapse(tmp_path):
    path = tmp_path / "space.pt"
    path.write_bytes(space_snapshot(2901))
    prompt = (b"." + b" " * 8).hex()
    args = ("generate", "--snapshot", path, "--prompt-hex", prompt, "--max-bytes", 16, "--seed", 2901)
    disabled = call_cli(*args, "--repetition-penalty", 1.0)
    streamed = call_cli("stream", "--snapshot", path, "--input-hex", prompt, "--no-adapt", "--continuation", 16)
    assert disabled["generated_hex"] == streamed["continuation_hex"] == (b" " * 16).hex()
    assert disabled["repeat_window"] == 64 and disabled["repetition_penalty"] == 1.0
    enabled = (*args, "--repetition-penalty", 2.0, "--repeat-window", 1)
    broken = call_cli(*enabled)
    assert broken["generated_hex"] == (b". " * 8).hex()
    assert broken["repeat_window"] == 1 and broken["repetition_penalty"] == 2.0
    stochastic = (*enabled, "--temperature", 0.7, "--top-p", 0.9)
    a, b = call_cli(*stochastic), call_cli(*stochastic)
    assert a["generated_hex"] == b["generated_hex"] != disabled["generated_hex"]
    assert all(row["donor_forward_calls"] == 0 for row in (disabled, broken, a, b))


@pytest.mark.parametrize("flag,value", [("--repetition-penalty", "0.9"), ("--repetition-penalty", "inf"),
                                        ("--repeat-window", "0"), ("--repeat-window", "513")])
def test_generate_rejects_invalid_repetition_settings(flag, value, tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["generate", "--snapshot", str(tmp_path / "absent.pt"), "--prompt-text", "x", flag, value])
    assert exc.value.code == 2
