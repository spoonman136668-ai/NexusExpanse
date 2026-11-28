"""Offline arithmetic, restricted Python, and causal byte search checks."""
from dataclasses import replace
import hashlib
import json
import math

import pytest
import torch

from benchmarks.phase15_cosine_lr_and_sampling import call_cli, space_snapshot
from nexus_expanse.ssd_engram import SSDEngramStore
from nexus_expanse.unified import NexusExpanseUnified
from nexus_expanse.verified_search import ByteCandidate, SearchResult, Verification, VerifiedByteSearch, make_verifier


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


def test_arithmetic_checks_an_exact_first_line():
    verify = make_verifier("arithmetic", b"Arithmetic:\n (123 + 45) // 2 - 8 % 3 = ")
    assert verify(b" +82 \r\nanything") == (True, 1, 1, None)
    assert verify(b"81") == (False, 0, 1, None)
    for raw in (b"82.0", b"82 words", b"\n82", b"82\xff"):
        assert not verify(raw).passed
    for prompt in (b"2**1000000000=", b"1//0=", b"True=", b"2147483647+1=", b"2/1=", b"1+2"):
        with pytest.raises(ValueError):
            make_verifier("arithmetic", prompt)


def test_function_and_expression_accept_only_exact_bounded_results():
    prompt = b"Add the arguments, then take the absolute value.\ndef f(a, b):\n    "
    rows = [{"args": [-4, 2], "expected": 2}, {"args": [1, 2], "expected": 3}]
    function = make_verifier("python_fn", prompt, rows)
    expression = make_verifier("python_expr", prompt, rows)
    assert function(b"return abs(a+b)\n\n") == (True, 2, 2, None)
    assert function(b"def f(a, b):\n    return a + b if a+b >= 0 else -(a+b)\n")[0]
    assert function(b"return a + b") == (False, 1, 2, None)
    assert expression(b"abs(a+b)")[0]
    assert not function(b"def g(a,b):\n    return abs(a+b)")[0]
    assert not function(b"def f(a,b=2):\n    return abs(a+b)")[0]
    assert not function(b"return 1//(a-1)").passed
    assert function(b"return 3 if a==1 else 1//0").passed_tests == 1
    with pytest.raises(ValueError, match="duplicate"):
        make_verifier("python_fn", b"def f(a,a):", rows)
    with pytest.raises(ValueError, match="test cases"):
        make_verifier("python_fn", prompt, rows * 17)
    assert not make_verifier("python_expr", b"def f(x):", [{"args": [0], "expected": 1}])(b"True").passed


def test_unsafe_candidates_are_rejected_before_any_case(tmp_path):
    marker = tmp_path / "must-not-exist"
    verify = make_verifier("python_fn", b"def f(x):\n    ", [{"args": [0], "expected": 0}])
    payloads = (
        b"return __import__('os')", f"return open({str(marker)!r}, 'w')".encode(),
        b"return x.__class__", b"return 0 if x==0 else x.__class__", b"while True:\n    pass",
        b"return 2**999999999", b"import os\nreturn 0", b"return [i for i in [0]]",
        b"return eval('0')", b"return 0; import os", b"return (lambda: 0)()",
        b"@abs\ndef f(x):\n    return x", b"return " + b"+" * 60 + b"x",
    )
    for source in payloads:
        result = verify(source)
        assert (result.passed, result.passed_tests, result.total_tests) == (False, 0, 1)
        assert result.error
    assert not marker.exists()


def test_verified_selection_prefers_highest_probability_in_eligible_set():
    rejected = Verification(False, 1, 2, None)
    accepted = Verification(True, 2, 2, None)
    raw = tuple(ByteCandidate(bytes([128+i]), score, check, 0.0) for i, (score, check) in enumerate(
        ((-1, rejected), (-4, accepted), (-2, accepted), (-2, accepted))))
    assert SearchResult(raw, 0).selected is raw[2]
    assert SearchResult(raw[:1], 0).selected.verification == rejected


def test_seeded_rollouts_replay_same_scores_and_match_existing_greedy(tmp_path):
    archive = space_snapshot(42)
    snapshot = tmp_path / "snapshot.pt"
    snapshot.write_bytes(archive)
    prompt = b"123+45="
    options = dict(candidates=5, max_bytes=6, seed=3190, stop_bytes=b"\x00\xff")
    engine = VerifiedByteSearch(archive, always_wake=True)
    verify = make_verifier("arithmetic", prompt)
    one, two = (engine.run(prompt, verify, **options) for _ in range(2))
    assert [replace(row, seconds=0) for row in one.candidates] == [replace(row, seconds=0) for row in two.candidates]
    assert one.donor_forward_calls == two.donor_forward_calls == 0
    legacy = call_cli("generate", "--snapshot", snapshot, "--prompt-text", prompt.decode(), "--max-bytes", 6,
                      "--always-wake")
    assert one.candidates[0].raw.hex() == legacy["generated_hex"]
    for item in one.candidates:
        predicted = engine.forecast_trace(prompt, item.raw)
        assert item.logprob == sum(math.log(float(row[byte])) for row, byte in zip(predicted[:-1], item.raw, strict=True))
    prefix = bytes([0, 255])
    a, b = (engine.forecast_trace(prompt, prefix + suffix) for suffix in (b"hello", b"world"))
    assert torch.equal(a[:3], b[:3]) and torch.equal(a[:3], engine.forecast_trace(prompt, prefix))
    with pytest.raises(ValueError, match="SHA-256"):
        VerifiedByteSearch(archive, expected_sha256="0" * 64)


def test_cli_verified_int8_engram_stops_at_delimiter(tmp_path):
    prompt = b"123+45="
    src, index, snapshot = (tmp_path / k for k in ("fit.bin", "index.db", "snapshot.pt"))
    src.write_bytes((prompt + b"168\n") * 80)
    receipt = SSDEngramStore.build(src, index, max_order=len(prompt))
    raw = NexusExpanseUnified.restore(space_snapshot(20)).snapshot(snapshot_quant="int8")
    snapshot.write_bytes(raw)
    common = ("search-verified", "--snapshot", snapshot, "--snapshot-sha256", hashlib.sha256(raw).hexdigest(),
              "--prompt-text", prompt.decode(), "--always-wake", "--engram-index", index, "--engram-alpha", "0.9",
              "--engram-index-sha256", receipt["index_sha256"], "--device", "cpu", "--verifier", "arithmetic",
              "--candidates", "3", "--max-bytes", "16")
    received = [call_cli(*common) for _ in range(2)]
    for row in received:
        assert row.pop("wall_ms") >= 0
        assert row["verified"] and row["selected_hex"] == b"168\n".hex()
        assert row["candidates_tried"] == 3 and 1 <= row["verified_count"] <= 3
        assert row["passed_tests"] == row["total_tests"] == 1 and row["donor_forward_calls"] == 0
    assert received[0] == received[1]


def test_cli_python_cases_reads_file_and_reports_unverified(tmp_path):
    snapshot, cases = tmp_path / "snapshot.pt", tmp_path / "cases.json"
    snapshot.write_bytes(space_snapshot(2))
    rows = [{"args": [1], "expected": 2}, {"args": [2], "expected": 3}]
    cases.write_text(json.dumps(rows))
    options = ("search-verified", "--snapshot", snapshot, "--prompt-text", "def f(x):\n    ",
               "--verifier", "python_fn", "--max-bytes", "1", "--candidates", "1", "--tests-json")
    report = call_cli(*options, cases)
    direct = call_cli(*options, json.dumps(rows))
    assert not report["verified"] and report["passed_tests"] == 0 and report["total_tests"] == 2
    assert report["verified_count"] == 0 and math.isfinite(report["selected_logprob"])
    assert {k:v for k,v in report.items() if k != "wall_ms"} == {k:v for k,v in direct.items() if k != "wall_ms"}
