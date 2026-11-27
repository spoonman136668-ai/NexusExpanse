"""Verify warm-start custody and raw-byte generation with local archives."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from benchmarks import gate17_gpu_scaling_harness as scaling
from nexus_expanse.cli import default_model, main
from nexus_expanse.checkpoint_transport import CorrectedLinear
from nexus_expanse.shard_stream_assimilation import make_student
from nexus_expanse.unified import NexusExpanseUnified
from nexus_expanse.weight_editing import skill_snapshot


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def single_thread():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prior)


def invoke(*args, expected=0):
    run = subprocess.run([sys.executable, "-m", "nexus_expanse.cli", *map(str, args)],
                         cwd=ROOT, text=True, capture_output=True, timeout=120, check=False)
    assert run.returncode == expected, run.stdout + run.stderr
    return run if expected else [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]


def test_warm_start_copies_untied_readout_and_folds_affine_residual():
    skill = make_student(19, student_dim=64, student_layers=4).eval()
    fc1 = skill.core.layers[0].ffn.fc1
    skill.core.layers[0].ffn.fc1 = CorrectedLinear(fc1, torch.eye(128, 64), "spectral4")
    geometry = scaling.Geometry(42, 64, 80, 1, 1, 1, 1, 4)
    target = scaling.model_for(geometry, "causal_attention_skill", torch.device("cpu"))
    tail = target.pos_emb.weight[64:].detach().clone()
    scaling.initialize_from_skill(target, skill)
    assert torch.equal(target.readout.weight, skill.readout.weight)
    assert torch.equal(target.emb.weight, skill.core.emb.weight)
    assert torch.equal(target.pos_emb.weight[:64], skill.core.pos_emb.weight)
    assert torch.equal(target.pos_emb.weight[64:], tail)
    probe = torch.eye(64)
    torch.testing.assert_close(target.layers[0].ffn.fc1(probe), skill.core.layers[0].ffn.fc1(probe))


def test_warm_start_cli_exact_resume_and_snapshot(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for domain_index, domain in enumerate(scaling.DOMAINS):
        for split_index, split in enumerate(("train", "eval")):
            raw = bytes((i * (domain_index + 1) + 7 * split_index) % 256 for i in range(96))
            (corpus / f"{domain}-{split}.bin").write_bytes(raw)
    skill = make_student(13, student_dim=64, student_layers=4).eval()
    skill.static_byte_prior = torch.zeros((256, 256)).log_softmax(-1)
    skill.prior_alpha = 0.25
    initial = tmp_path / "initial.pt"
    initial.write_bytes(skill_snapshot(skill))
    args = ("train", "--init-skill", initial, "--corpus-dir", corpus, "--device", "cpu", "--dim", 64,
            "--context", 16, "--epochs", 2, "--steps-per-epoch", 2, "--batch-per-domain", 1,
            "--eval-blocks", 1, "--use-sdpa")
    run, reference = tmp_path / "run", tmp_path / "uninterrupted"
    snapshot = tmp_path / "unified.pt"
    invoke(*args, "--directory", run)
    receipt = json.loads((run / "gate17-epoch-01.json").read_bytes())
    expected = hashlib.sha256(initial.read_bytes()).hexdigest()
    assert receipt["identity"]["init_skill_sha256"] == expected
    assert receipt["identity"]["geometry"]["layers"] == 4

    changed = tmp_path / "changed.pt"
    changed.write_bytes(skill_snapshot(make_student(14, student_dim=64, student_layers=4).eval()))
    rejected = invoke(*args, "--init-skill", changed, "--directory", run, expected=2)
    assert "epoch identity or hash mismatch" in rejected.stderr
    assert not (run / "gate17-epoch-02.intent.json").exists()
    assert "width or depth" in invoke(*args, "--layers", 2, "--directory", run, expected=2).stderr

    exported = invoke(*args, "--directory", run, "--save-snapshot", snapshot)[-1]
    invoke(*args, "--directory", reference, "--epoch-limit", 2)
    completed = json.loads((run / "gate17.json").read_bytes())
    whole = json.loads((reference / "gate17.json").read_bytes())
    assert completed["final_hash"] == whole["final_hash"] and completed["cold_equal"]
    resumed = torch.load(run / "gate17-epoch-02.pt", weights_only=True)
    direct = torch.load(reference / "gate17-epoch-02.pt", weights_only=True)
    torch.testing.assert_close(resumed, direct, rtol=0, atol=0)
    first, second = (item["telemetry"] for item in completed["epochs"])
    assert first["end_model_hash"] == second["start_model_hash"]
    assert first["prior_alpha_schedule"] == [0.25, 0.0] and second["prior_alpha_schedule"] == [0.0, 0.0]
    assert second["optimizer_steps_at_start"] == 2
    verified, = invoke("verify", "--snapshot", snapshot, "--snapshot-sha256", exported["snapshot_sha256"])
    assert verified["exact_state"] and verified["round_trip_probabilities"] == 0.0
    assert set(verified["parity"].values()) == {0.0} and len(verified["parity"]) == 4
    assert exported["donor_forward_calls"] == completed["donor_forward_calls"] == 0
    assert completed["fresh_evaluation"]["donor_forward_calls"] == 0
    assert not any("prior" in key for key in NexusExpanseUnified.restore(snapshot.read_bytes()).backbone.state_dict())


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "unified.pt"
    path.write_bytes(default_model().snapshot())
    return path


def test_generate_exact_greedy_bytes_match_steps_and_stream(snapshot, capsys):
    prompt = "café"
    data = snapshot.read_bytes()
    model = NexusExpanseUnified.restore(data)
    for byte in prompt.encode("utf-8"):
        forecast = model.step(byte, adapt=False)
    expected = []
    for _ in range(16):
        value = int(forecast.mixed.argmax())
        expected.append(value)
        forecast = model.step(value, adapt=False)
    main(["generate", "--snapshot", str(snapshot), "--prompt-text", prompt, "--max-bytes", "16"])
    report = json.loads(capsys.readouterr().out)
    main(["stream", "--snapshot", str(snapshot), "--input-hex", prompt.encode().hex(),
          "--continuation", "16", "--no-adapt"])
    streamed = json.loads(capsys.readouterr().out)
    assert report["generated_hex"] == bytes(expected).hex() == streamed["continuation_hex"]
    assert report["generated_text"] == bytes(expected).decode("utf-8", errors="replace")
    assert report["prompt_bytes"] == 5 and report["generated_bytes"] == 16 and report["ms_per_byte"] >= 0
    assert report["donor_forward_calls"] == 0 and report["snapshot_sha256"] == hashlib.sha256(data).hexdigest()
    assert snapshot.read_bytes() == data


def test_generate_seeded_nucleus_is_repeatable_with_adaptation(snapshot, capsys):
    args = ["generate", "--snapshot", str(snapshot), "--prompt-hex", "ffcc00", "--max-bytes", "24",
            "--temperature", "0.7", "--top-p", "0.9", "--seed", "19", "--adapt"]
    reports = []
    for _ in range(2):
        main(args)
        reports.append(json.loads(capsys.readouterr().out))
    assert reports[0]["generated_hex"] == reports[1]["generated_hex"]
    assert all(report["donor_forward_calls"] == 0 and report["generated_bytes"] == 24 for report in reports)
