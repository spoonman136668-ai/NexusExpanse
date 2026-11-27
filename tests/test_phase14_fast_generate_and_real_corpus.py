"""Offline exactness, device selection, and whole-document corpus checks."""

import errno
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter

import numpy as np
import pytest
import torch

from benchmarks.gate17_gpu_scaling_harness import Geometry, initialize_from_skill, model_for
from nexus_expanse.cli import main
from nexus_expanse.patching import CompactBigram
from nexus_expanse.shard_stream_assimilation import make_student
from nexus_expanse.unified import NexusExpanseUnified
from scripts import prepare_real_corpus
from scripts.prepare_real_corpus import MIN_BYTES, prepare

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def make_snapshot(path, *, warm):
    geometry = Geometry(71, 64 if warm else 32, 32, 1, 1, 1, 1, 4 if warm else 2)
    core = model_for(geometry, "causal_attention_skill" if warm else "causal_attention_monarch",
                     torch.device("cpu"))
    if warm:
        initialize_from_skill(core, make_student(79, student_dim=64, student_layers=4))
    counts = np.zeros((256, 256), dtype="<u4")
    counts[97, 97] = 1024
    model = NexusExpanseUnified(core, CompactBigram.from_dense(counts))
    # Exercise an in-flight snapshot ending in a folded byte.
    model.prefill(b"fitted cursor aa", adapt=True)
    path.write_bytes(model.snapshot())
    return path


@pytest.mark.parametrize("warm", [False, True], ids=["scratch", "warm-start"])
@pytest.mark.parametrize("length", [7, 97])
@pytest.mark.parametrize("adapt", [False, True], ids=["frozen", "adapt"])
def test_fast_generation_preserves_legacy_steps(tmp_path, capsys, warm, length, adapt):
    path = make_snapshot(tmp_path / "unified.pt", warm=warm)
    saved = path.read_bytes()
    prompt = (b"aa\x00\xff byte-level\x13 " * length)[:length]
    baseline = NexusExpanseUnified.restore(saved)
    fast = NexusExpanseUnified.restore(saved, fast_decode=True)
    for byte in prompt:
        expected = baseline.step(byte, adapt=adapt)
    actual = fast.prefill(prompt, adapt=adapt)
    assert torch.equal(expected.mixed, actual.mixed)
    selected = bytearray()
    for _ in range(20):
        value = int(expected.mixed.argmax())
        selected.append(value)
        expected = baseline.step(value, adapt=adapt)
        actual = fast.step(value, adapt=adapt)
        assert torch.equal(expected.mixed, actual.mixed)
    assert fast.arbitrator.updates == baseline.arbitrator.updates

    args = ["generate", "--snapshot", str(path), "--device", "cpu", "--prompt-hex", prompt.hex(),
            "--max-bytes", "20"]
    main([*args, "--adapt"] if adapt else args)
    result = json.loads(capsys.readouterr().out)
    assert result["generated_hex"] == selected.hex()
    assert result["device"] == "cpu" and result["prefill_ms"] >= 0 and result["ms_per_byte"] >= 0
    assert result["donor_forward_calls"] == 0 and result["snapshot_sha256"] == hashlib.sha256(saved).hexdigest()
    if not adapt:
        main(["stream", "--snapshot", str(path), "--input-hex", prompt.hex(), "--continuation", "20", "--no-adapt"])
        assert json.loads(capsys.readouterr().out)["continuation_hex"] == selected.hex()
    assert path.read_bytes() == saved


def test_generate_device_dispatch_and_unavailable_cuda(tmp_path, capsys, monkeypatch):
    path = make_snapshot(tmp_path / "unified.pt", warm=False)
    args = ["generate", "--snapshot", str(path), "--prompt-text", "hello", "--max-bytes", "2"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    main([*args, "--device", "auto"])
    assert json.loads(capsys.readouterr().out)["device"] == "cpu"
    with pytest.raises(SystemExit) as error:
        main([*args, "--device", "cuda"])
    assert error.value.code == 2 and "CUDA is unavailable" in capsys.readouterr().err
    original = NexusExpanseUnified.restore

    def mock_restore(data, *, fast_decode, decode_device):
        assert decode_device.type == "cuda" and fast_decode
        return original(data, fast_decode=True)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(NexusExpanseUnified, "restore", staticmethod(mock_restore))
    main([*args, "--device", "auto"])
    assert json.loads(capsys.readouterr().out)["device"] == "cuda"


@pytest.fixture
def sources(tmp_path):
    result = {}
    duplicate = b"shared source paragraph, repeated verbatim across multiple input files.\n" * 5
    for domain, suffix in (("prose", ".md"), ("code", ".py")):
        directory = tmp_path / domain
        directory.mkdir()
        result[domain] = directory
        for index in range(16):
            rng = random.Random(f"{domain}:{index}")
            raw = bytes(rng.choices(b"abcdefghijklmnopqrstuvwxyz0123456789", k=12_500))
            (directory / f"record-{index}{suffix}").write_bytes((duplicate if index < 4 else b"") + raw)
        (directory / f"duplicate{suffix}").write_bytes((directory / f"record-0{suffix}").read_bytes())
    return result


def check_pack(out, manifest, inputs):
    assert manifest == json.loads((out / "corpus_manifest.json").read_bytes())
    assert manifest["overlap_32b"] == {"math": 0, "code": 0, "prose": 0} and manifest["global_overlap_32b"] == 0
    assert manifest["total_bytes"] <= manifest["size_limit_bytes"]
    seen = {"train": set(), "eval": set()}
    for key, entry in manifest["files"].items():
        split, domain = key.split("_")
        raw = (out / entry["file"]).read_bytes()
        assert len(raw) == entry["bytes"] >= MIN_BYTES
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        seen[split].update(raw[i:i + 32] for i in range(len(raw) - 31))
        if domain in inputs:
            docs = {p.read_bytes() + b"\n" for p in inputs[domain].iterdir()}
            selected = [doc for doc in docs if doc in raw]
            assert len(selected) == entry["documents"] and sum(map(len, selected)) == len(raw)
    assert not seen["train"] & seen["eval"]


def test_whole_documents_and_duplicate_paragraphs_pack_deterministically(tmp_path, sources, monkeypatch):
    def fuse(*args, **kwargs):
        raise PermissionError(errno.EPERM, "FUSE hard links unavailable")

    monkeypatch.setattr(os, "link", fuse)
    a, b = tmp_path / "pack-a", tmp_path / "pack-b"
    math_path = tmp_path / "short-math.txt"
    math_path.write_text("This file has too few bytes for two complete splits.")
    settings = dict(local={**sources, "math": math_path}, hub={}, limit=600_000, math_fallback=True)
    manifest = prepare(a, **settings)
    assert manifest == prepare(b, **settings)
    assert manifest["sources"]["math"] == "synthetic"
    check_pack(a, manifest, sources)
    assert not list(a.glob("*.partial"))
    with pytest.raises(FileExistsError):
        prepare(a, **settings)
    assert manifest == json.loads((a / "corpus_manifest.json").read_bytes())


def test_hub_like_holdout_with_shared_prefixes_finishes_under_five_seconds(tmp_path, monkeypatch):
    code = tmp_path / "code"
    code.mkdir()
    rng = random.Random(14)
    alphabet = b"abcdefghijklmnopqrstuvwxyz0123456789"
    # Ranking mixes the first two import groups and leaves too little training
    # code. The bounded repair can reserve the middle group for training.
    for group, count, size in (("a", 3, 12_500), ("b", 4, 14_000), ("c", 5, 6_000)):
        imports = f"from {group * 32} import prepare_many_cases\n".encode("ascii")
        for index in range(count):
            (code / f"{group}-{index:02}.py").write_bytes(imports + bytes(rng.choices(alphabet, k=size)))

    streamed = 0

    def hub_records(spec, field):
        nonlocal streamed
        assert (spec, field) == ("offline/story-fixture:raw", "text")
        rng = random.Random(14)
        for index in range(5_000):
            streamed += 1
            # Mixing the two prefix groups in eval blocks every train story.
            start = (b"Once upon a time, there was a little " if index < 4_400
                     else b"At sunrise the village shutters opened on ")
            yield start + bytes(rng.choices(alphabet, k=112 - len(start)))

    original_graph = prepare_real_corpus.conflict_graph
    graph_sizes = []

    def check_graph(documents):
        assert set(documents) == {"code"}, "Hub records must stay out of the conflict graph"
        graph_sizes.append(len(documents["code"]))
        return original_graph(documents)

    monkeypatch.setattr(prepare_real_corpus, "hub_documents", hub_records)
    monkeypatch.setattr(prepare_real_corpus, "conflict_graph", check_graph)
    out = tmp_path / "packed"
    start = perf_counter()
    manifest = prepare(out, local={"code": code}, hub={"prose": "offline/story-fixture:raw"},
                       limit=900_000, math_fallback=True)
    elapsed = perf_counter() - start
    assert elapsed < 5, f"5,000-document corpus took {elapsed:.3f}s"
    assert streamed == 5_000
    assert graph_sizes == [12]
    assert manifest["sources"] == {"math": "synthetic", "code": "local", "prose": "hf"}
    check_pack(out, manifest, {"code": code})


def test_real_corpus_cli_feeds_training(tmp_path, sources):
    out = tmp_path / "pack"
    pack = subprocess.run(
        [sys.executable, "scripts/prepare_real_corpus.py", "--prose-dir", str(sources["prose"]),
         "--code-dir", str(sources["code"]), "--math-fallback-synthetic", "--size-mb", "0.6",
         "--out-dir", str(out)], cwd=ROOT, capture_output=True, text=True, check=True,
    )
    check_pack(out, json.loads(pack.stdout), sources)
    training = subprocess.run(
        [sys.executable, "-m", "nexus_expanse.cli", "train", "--corpus-dir", str(out),
         "--directory", str(tmp_path / "run"), "--device", "cpu", "--dim", "32", "--context", "16",
         "--epochs", "1", "--steps-per-epoch", "1", "--batch-per-domain", "1", "--eval-blocks", "1",
         "--save-snapshot", str(tmp_path / "unified.pt")],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    summary = [json.loads(line) for line in training.stdout.splitlines() if line.startswith("{")][-1]
    assert summary["exact_state"] and set(summary["parity"].values()) == {0.0}
