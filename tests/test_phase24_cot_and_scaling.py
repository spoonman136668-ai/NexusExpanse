"""Phase 24 verifier and training regressions using local byte fixtures."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from benchmarks import gate17_gpu_scaling_harness as scaling
from benchmarks.phase24_cot_and_scaling import large_geometry
from nexus_expanse import cli, core
from nexus_expanse.structured_verifiers import final_answer
from nexus_expanse.unified import NexusExpanseUnified
from nexus_expanse.verified_search import VerifiedByteSearch, make_verifier
from scripts import prepare_reasoning_curriculum as curriculum
from scripts.prepare_gpu_corpus import assert_disjoint

CPU = torch.device("cpu")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def torch_settings():
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


@pytest.mark.parametrize("kind,prompt,tests,spec,answer", [
    ("arithmetic", b"3*7+2=", None, None, b"23\n"),
    ("python_fn", b"def f(x):\n    ", [{"args":[2], "expected":3}], None, b"return x+1\n"),
    ("logic_csp", b"Pick 4.", None, {"variables":{"A":[4]}, "constraints":[{"type":"equals","left":"A","value":4}]}, b"A=4\n"),
    ("table_format", b"Return CSV.", None, {"format":"csv","columns":["id"],"rows":[[1]]}, b"id\n1\n"),
    ("instruction_constraints", b"Return go.", None, {"exact_lines":1,"must_start_with":"go"}, b"go\n"),
])
def test_all_verifiers_strip_only_closed_leading_think(kind, prompt, tests, spec, answer):
    direct = make_verifier(kind, prompt, tests, spec=spec)
    verify = make_verifier(kind, prompt, tests, spec=spec, allow_think=True)
    baseline = direct(answer)
    assert baseline.passed and verify(answer) == baseline
    wrapped = b"<think>scratch\n\n" + answer + b"</think> \r\n\t" + answer
    assert verify(wrapped) == baseline and not direct(wrapped).passed
    bad = verify(b"<think>\n" + answer)
    assert not bad.passed and bad.passed_tests == 0 and bad.total_tests == baseline.total_tests
    assert bad.error == "unterminated think block"
    assert not verify(b"<think>\n" + answer + b"</think>").passed
    assert verify(b" <think>" + answer) == direct(b" <think>" + answer)
    assert verify(b"<think>ignored</think>\nwrong") == direct(b"wrong")


def test_cli_search_and_jsonl_eval_wait_for_post_scratchpad_answer(tmp_path, monkeypatch, capsys):
    candidate = b"<think>4+2\n\n equals 6</think>\n\n6\n"

    def forecasts(self, prompt):
        for byte in candidate:
            distribution = torch.zeros(256, dtype=torch.float64)
            distribution[byte] = 1.0
            yield distribution

    monkeypatch.setattr(VerifiedByteSearch, "_forecasts", forecasts)
    snapshot = tmp_path / "model.pt"
    snapshot.write_bytes(cli.default_model().snapshot())
    common = ("--snapshot", str(snapshot), "--max-bytes", str(len(candidate)), "--device", "cpu",
              "--always-wake", "--no-episodic", "--candidates", "1")
    argv = ("search-verified", "--verifier", "arithmetic", "--prompt-text", "4+2=", *common)
    cli.main([*argv])
    plain = json.loads(capsys.readouterr().out)
    assert not plain["verified"] and bytes.fromhex(plain["selected_hex"]) == b"<think>4+2\n"
    cli.main([*argv, "--allow-think"])
    report = json.loads(capsys.readouterr().out)
    assert report["verified"] and report["allow_think"] and report["donor_forward_calls"] == 0
    assert bytes.fromhex(report["selected_hex"]) == candidate
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps(dict(id="math", category="math", verifier="arithmetic", prompt="4+2=")))
    cli.main(["eval-livebench", "--tasks-jsonl", str(tasks), *common, "--allow-think"])
    result = json.loads(capsys.readouterr().out)
    assert result["counts"]["math"]["passed"] == 1
    assert result["allow_think"] and result["donor_forward_calls"] == 0


def test_curriculum_cli_includes_verified_deterministic_work_and_zero_overlap(tmp_path):
    def prepare(path):
        completed = subprocess.run([sys.executable, str(ROOT / "scripts/prepare_reasoning_curriculum.py"),
                                    "--out-dir", str(path), "--size-mb", "0.048", "--seed", "3801",
                                    "--include-think", "--builtin-dev-suite"],
                                   check=True, capture_output=True, text=True)
        return json.loads(completed.stdout)

    first, second = tmp_path / "first", tmp_path / "second"
    report = prepare(first)
    assert report == prepare(second) and report["include_think"]
    assert report["verifier_passed"] == sum(f["examples"] for f in report["files"].values())
    fit, held = ((first / f"stream-{split}.bin").read_bytes() for split in ("train", "eval"))
    assert not any(report["shared_32_byte_windows"].values())
    assert_disjoint((fit,), {"held":held})
    assert_disjoint((fit, held), curriculum.builtin_references())
    assert all(b"<think>" in stream and b"</think>" in stream for stream in (fit, held))
    assert final_answer(b"<think>example</think>\n4") == b"4"


def test_grad_accum_weights_partial_microbatches_by_target_byte_count():
    g = scaling.Geometry(13, 32, 8, 1, 1, 1, 1, n_heads=8, grad_accum=2)
    model, reference = (scaling.model_for(g, "causal_attention_monarch", CPU) for _ in range(2))
    batches = [[(b"abcdefghi", 0)], [(b"xz", 0)]]
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    combined = torch.optim.SGD(reference.parameters(), lr=.01)
    value, targets = scaling.accumulated_update(model, optimizer, batches, g, "fp32", CPU)
    batch, keep, _ = scaling.make_batch(batches[0]+batches[1], g.context, CPU)
    expected = scaling.update(reference, combined, scaling.loss_for(reference, batch, keep, "fp32", CPU))
    assert targets == 9 and value == pytest.approx(expected, abs=1e-6)
    for actual, control in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, control, rtol=1e-6, atol=1e-7)


def test_checkpoint_recomputes_each_attention_layer_only_in_training(monkeypatch):
    geometry = scaling.Geometry(24, 32, 8, 1, 1, 1, 1, 2, 8, 2, True)
    model = scaling.model_for(geometry, "causal_attention_skill", CPU)
    recorded = []
    real = core.checkpoint

    def wrap(layer, *args, **kwargs):
        assert kwargs["use_reentrant"] is False
        recorded.append(layer)
        return real(layer, *args, **kwargs)

    monkeypatch.setattr(core, "checkpoint", wrap)
    tokens = torch.tensor([[229, 30, 78]])
    model.forward_neural(tokens)[0].sum().backward()
    assert recorded == list(model.layers)
    model.eval()
    with torch.inference_mode():
        model.forward_neural(tokens)
    assert recorded == list(model.layers)


def test_cold_resume_and_int8_export_with_accum_checkpoint_and_eight_heads(tmp_path, monkeypatch, capsys):
    fit, check = tmp_path / "fit.bin", tmp_path / "check.bin"
    fit.write_bytes(b"disjoint raw offline fixture data for training\n")
    check.write_bytes(b"cpu parity on held bytes for phase 24\n")
    git = scaling.git
    monkeypatch.setattr(scaling, "git", lambda *args: b"" if args[0] == "status" else git(*args))
    base = ("train", "--train-file", fit, "--eval-file", check, "--device", "cpu", "--dim", 32,
            "--layers", 2, "--context", 8, "--use-sdpa", "--epochs", 2, "--steps-per-epoch", 2,
            "--batch-per-domain", 1, "--eval-blocks", 1, "--lr-schedule", "cosine")

    def run(*args):
        cli.main(list(map(str, args)))
        reports = [line for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
        return json.loads(reports[-1]) if reports else None

    direct, cold = tmp_path / "direct", tmp_path / "cold"
    a, b = tmp_path / "direct.pt", tmp_path / "cold.pt"
    flags = ("--n-heads", 8, "--grad-accum", 2, "--grad-checkpoint")
    run(*base, *flags, "--directory", cold)
    intent = cold / "gate17-epoch-01.intent.json"
    saved = intent.read_bytes()
    for changed in (("--n-heads", 4, "--grad-accum", 2, "--grad-checkpoint"),
                    ("--n-heads", 8, "--grad-accum", 1, "--grad-checkpoint"),
                    ("--n-heads", 8, "--grad-accum", 2)):
        with pytest.raises(SystemExit):
            run(*base, *changed, "--directory", cold)
        assert "epoch identity or hash mismatch" in capsys.readouterr().err
        assert intent.read_bytes() == saved and not (cold / "gate17-epoch-02.intent.json").exists()
    x = run(*base, *flags, "--directory", cold, "--save-snapshot", b, "--snapshot-quant", "int8")
    y = run(*base, *flags, "--directory", direct, "--epoch-limit", 2, "--save-snapshot", a, "--snapshot-quant", "int8")
    ar, br = (json.loads((p / "gate17.json").read_bytes()) for p in (direct,cold))
    assert ar["cold_equal"] and br["cold_equal"] and ar["final_hash"] == br["final_hash"]
    assert ar["checkpoint_sha256"] == br["checkpoint_sha256"] and ar["epochs"][1]["telemetry"]["optimizer_steps_at_start"] == 2
    assert a.read_bytes() == b.read_bytes() and x["snapshot_sha256"] == y["snapshot_sha256"]
    assert set(x["parity"].values()) == {0.0} and x["round_trip_probabilities"] == 0.0 and x["exact_state"]
    assert x["donor_forward_calls"] == br["fresh_evaluation"]["donor_forward_calls"] == 0
    geometry = br["identity"]["geometry"]
    assert geometry["n_heads"] == 8 and geometry["grad_accum"] == 2 and geometry["grad_checkpoint"]
    model = NexusExpanseUnified.restore(b.read_bytes())
    assert model.backbone.layers[0].attn.num_heads == 8 and not model.backbone.grad_checkpoint
    with pytest.raises(ValueError, match="head count"):
        scaling.snapshot_geometry(model, SimpleNamespace(dim=None, layers=None, n_heads=4))


@pytest.mark.parametrize("flag,value", [("--dim", 1056), ("--layers", 10), ("--n-heads", 2), ("--grad-accum", 33)])
def test_training_parser_bounds(flag, value):
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["train", "--train-file", "fit.bin", "--directory", "run", flag, str(value)])


def test_largest_dense_geometry_short_cpu_forward_and_int8_snapshot_size():
    result = large_geometry()
    assert result["config"]["dim"] == 1024 and result["config"]["n_layers"] == result["config"]["n_heads"] == 16
    assert result["parameters"] > 100_000_000 and result["finite"] and result["logits_shape"] == [1, 3, 256]
    assert 0 < result["int8_snapshot_bytes"] < result["fp32_checkpoint_bytes"] < result["snapshot_limit"]
    assert result["donor_forward_calls"] == 0
