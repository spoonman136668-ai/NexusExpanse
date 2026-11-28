"""Offline data selection, custody, and export checks for training modes."""

from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
from io import StringIO
import json
import random
from types import SimpleNamespace

import pytest
import torch

from benchmarks import gate17_gpu_scaling_harness as scaling
from benchmarks.phase13_warmstart_train import cli
from benchmarks.phase15_cosine_lr_and_sampling import call_cli
from nexus_expanse.shard_stream_assimilation import make_student
from nexus_expanse.weight_editing import skill_snapshot
from scripts import prepare_real_corpus as packer
from scripts.prepare_gpu_corpus import assert_disjoint


def binary_bytes(seed, size):
    return random.Random(seed).randbytes(size)


@pytest.mark.parametrize("focus", scaling.DOMAINS)
def test_focused_batches_keep_the_legacy_width_and_use_only_the_selected_domain(focus):
    geometry = replace(scaling.SMOKE, batch_per_domain=2, context=8)
    train = {d: bytes([i]) * 65 for i, d in enumerate(scaling.DOMAINS)}
    mixed = scaling.training_rows(train, geometry, 1)
    assert [data for data, _ in mixed] == [train[d] for d in scaling.DOMAINS for _ in range(2)]
    focused = scaling.training_rows(train, geometry, 1, focus_domain=focus)
    assert [data for data, _ in focused] == [train[focus]] * 6
    assert [index for _, index in focused] == [scaling.shuffled_index(i, 8, geometry.seed + scaling.DOMAINS.index(focus))
                                                for i in range(6, 12)]
    stream = scaling.training_rows({"stream": train[focus]}, geometry, 1)
    assert len(stream) == 2


@pytest.mark.parametrize("kind,diagnostic", [
    ("same", "distinct files"), ("identical", "identical full files"), ("short", "at least two raw bytes"),
    ("missing", "supplied together"), ("mixed", "used alone"),
])
def test_loader_rejects_invalid_single_stream_inputs(tmp_path, kind, diagnostic):
    first, second = tmp_path / "fit", tmp_path / "check"
    first.write_bytes(b"\x00\xff\x01")
    second.write_bytes(first.read_bytes() if kind == "identical" else b"\xff" if kind == "short" else b"\xff\x00")
    args = SimpleNamespace(smoke=False, train_file=first,
                           eval_file=first if kind == "same" else None if kind == "missing" else second,
                           focus_domain="prose" if kind == "mixed" else "all")
    with pytest.raises(ValueError, match=diagnostic):
        scaling.load_corpora(args)


def run_packer(*args):
    with redirect_stdout(StringIO()) as buffer:
        packer.main(list(map(str, args)))
    return json.loads(buffer.getvalue())


def check_packed(out, manifest):
    assert manifest == json.loads((out / "corpus_manifest.json").read_bytes())
    assert manifest["overlap_32b"] == {"stream": 0} and manifest["global_overlap_32b"] == 0
    assert set(manifest["files"]) == {"train_stream", "eval_stream"}
    train, held = ((out / f"stream-{s}.bin").read_bytes() for s in ("train", "eval"))
    for split, data in (("train", train), ("eval", held)):
        path = manifest["files"][f"{split}_stream"]
        assert path["bytes"] == len(data) >= packer.MIN_BYTES
        assert path["sha256"] == hashlib.sha256(data).hexdigest()
    assert len(train) + len(held) == manifest["total_bytes"] <= manifest["size_limit_bytes"]
    assert_disjoint((train,), {"stream": held})
    return train, held


def test_single_file_packer_preserves_binary_bytes_and_disjoint_windows(tmp_path):
    source = tmp_path / "capture.bin"
    data = binary_bytes(17, 4 * 65536)
    source.write_bytes(data)
    a, b = tmp_path / "a", tmp_path / "b"
    args = ("--single-dir", source, "--size-mb", "0.26")
    manifest = run_packer(*args, "--out-dir", a)
    assert manifest == run_packer(*args, "--out-dir", b)
    train, held = check_packed(a, manifest)
    spans = {data[i:i + 65536] for i in range(0, len(data), 65536)}
    assert set(train) == set(held) == set(range(256))
    assert all(payload[i:i + 65536] in spans for payload in (train, held) for i in range(0, len(payload), 65536))


def test_single_hub_cli_uses_mock_records_and_rejects_mixed_sources(tmp_path, monkeypatch):
    def records(spec, field):
        assert (spec, field) == ("offline/stories:plain", "story")
        rng = random.Random(42)
        for index in range(390):
            stem = (b"Once upon a time, there was a little house. " if index < 330
                    else b"The sun came up on the far edge of a lake. ")
            yield stem + bytes(rng.choices(b"abcdefghijklmnopqrstuvwxyz0123456789", k=1200))

    monkeypatch.setattr(packer, "hub_documents", records)
    args = ("--single-hf", "offline/stories:plain", "--hf-text-field", "story", "--size-mb", "0.4")
    target = tmp_path / "packed"
    manifest = run_packer(*args, "--out-dir", target)
    assert manifest["sources"] == {"stream": "hf"}
    check_packed(target, manifest)
    with pytest.raises(SystemExit) as error:
        run_packer(*args, "--prose-dir", tmp_path, "--out-dir", tmp_path / "invalid")
    assert error.value.code == 2 and not (tmp_path / "invalid").exists()


def test_single_packer_refuses_to_publish_overlapping_splits(tmp_path):
    directory = tmp_path / "documents"
    directory.mkdir()
    for i in range(2):
        (directory / str(i)).write_bytes(b"continuous shared header across entire split\n" + binary_bytes(i, 60000))
    with pytest.raises(ValueError, match="could not form 2 leakage-free files"):
        packer.prepare_single(tmp_path / "out", local=directory, limit=140_000)
    assert not (tmp_path / "out").exists()


def qualification(path, exported):
    verified = call_cli("verify", "--snapshot", path, "--snapshot-sha256", exported["snapshot_sha256"])
    generated = call_cli("generate", "--snapshot", path, "--prompt-hex", "00ff417f", "--max-bytes", 8, "--always-wake")
    for item in (exported, verified):
        assert item["snapshot_quant"] == "int8" and item["exact_state"]
        assert item["round_trip_probabilities"] == 0.0
        assert set(item["parity"].values()) == {0.0} and len(item["parity"]) == 4
    assert all(item["donor_forward_calls"] == 0 for item in (exported, verified, generated))
    assert generated["always_wake"] and generated["generated_bytes"] == 8


@pytest.mark.parametrize("mode", ["focus", "stream"])
def test_cli_cold_resume_rejects_identity_drift_and_exports_verified_int8(tmp_path, mode):
    if mode == "focus":
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        for i, label in enumerate(scaling.DOMAINS):
            (corpus / f"{label}-train.bin").write_bytes(binary_bytes(i, 385))
            (corpus / f"{label}-eval.bin").write_bytes(binary_bytes(90 + i, 65))
        source = ("--corpus-dir", corpus, "--focus-domain", "prose", "--skill-backbone")
        drift = ("--focus-domain", "code")
    else:
        train, held, changed = (tmp_path / name for name in ("fit.bin", "check.bin", "changed.bin"))
        train.write_bytes(bytes(range(256)) + binary_bytes(71, 129))
        held.write_bytes(binary_bytes(72, 65))
        changed.write_bytes(bytes(range(256)) + binary_bytes(73, 129))
        assert all({0, 255} <= set(p.read_bytes()) for p in (train, changed))
        init = tmp_path / "init.pt"
        init.write_bytes(skill_snapshot(make_student(17).eval()))
        source = ("--train-file", train, "--eval-file", held, "--init-skill", init, "--dtype", "bf16", "--use-sdpa")
        drift = ("--train-file", changed)
    common = ("train", *source, "--device", "cpu", "--dim", 32, "--context", 16, "--epochs", 2,
              "--steps-per-epoch", 2, "--batch-per-domain", 1, "--eval-blocks", 1, "--lr-schedule", "cosine")
    resumed, direct = tmp_path / "resumed", tmp_path / "direct"
    cli(*common, "--directory", resumed, "--epoch-limit", 1)
    first = resumed / "gate17-epoch-01.json"
    immutable = first.read_bytes()
    with pytest.raises(RuntimeError, match="epoch identity or hash mismatch"):
        cli(*common, "--directory", resumed, *drift)
    assert first.read_bytes() == immutable and not (resumed / "gate17-epoch-02.intent.json").exists()
    target, direct_target = tmp_path / "resumed.pt", tmp_path / "direct.pt"
    exported = cli(*common, "--directory", resumed, "--epoch-limit", 1, "--until-complete",
                   "--save-snapshot", target, "--snapshot-quant", "int8")
    reference = cli(*common, "--directory", direct, "--epoch-limit", 2, "--save-snapshot", direct_target,
                    "--snapshot-quant", "int8")
    a, b = (json.loads((p / "gate17.json").read_bytes()) for p in (resumed, direct))
    assert a["cold_equal"] and b["cold_equal"] and a["final_hash"] == b["final_hash"]
    assert a["donor_forward_calls"] == b["donor_forward_calls"] == 0
    assert [item["sha256"] for item in a["epochs"]] == [item["sha256"] for item in b["epochs"]]
    assert exported["snapshot_sha256"] == reference["snapshot_sha256"]
    assert len({item["telemetry"]["worker_pid"] for item in a["epochs"]}) == 2
    assert set(a["fresh_evaluation"]["parity"].values()) == {0.0}
    if mode == "focus":
        assert a["identity"]["focus_domain"] == "prose"
        assert "prose" in a["fresh_evaluation"]["validation"]
        assert all(item["telemetry"]["target_bytes"] == 2 * 3 * 16 for item in a["epochs"])
    else:
        assert set(a["fresh_evaluation"]["validation"]) == {"stream"}
        for split, file in (("train", train), ("eval", held)):
            assert a["identity"]["corpora"][split]["stream"]["sha256"] == hashlib.sha256(file.read_bytes()).hexdigest()
        assert all(item["telemetry"]["target_bytes"] == 2 * 16 for item in a["epochs"])
    qualification(target, exported)
