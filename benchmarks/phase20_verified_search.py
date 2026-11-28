"""Frozen CPU DEV comparison of verified byte search budgets on held-out tasks."""
import argparse
import json
from pathlib import Path
import operator
import platform
import random
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.dev_artifacts import atomic, atomic_bytes, digest
from benchmarks.gate_custody import peak_rss
from benchmarks.phase13_warmstart_train import cli
from benchmarks.phase3_multi_skill_routing import git
from nexus_expanse.ssd_engram import SSDEngramStore
from nexus_expanse.verified_search import SearchResult, VerifiedByteSearch, make_verifier

BASE = "fa5fbec48ef711753c4cef45022d804b11efbbc6"
PREREG = "docs/PHASE20_VERIFIED_TEST_TIME_SEARCH_PREREG.md"
SEEDS = tuple(range(3401, 3406))
BUDGETS = (1, 8, 16)
STEPS, CONTEXT, BATCH = 192, 64, 4
ARITHMETIC = (("+", operator.add), ("-", operator.sub), ("*", operator.mul),
              ("%", operator.mod), ("//", operator.floordiv))
PROGRAMS = (("Add", "+", operator.add), ("Multiply by", "*", operator.mul),
            ("Take the remainder modulo", "%", operator.mod))
TEST_INPUTS = (-13, -3, 0, 2, 5, 17)


def data_for(seed):
    rng = random.Random(seed)
    fit, calibration, check = [], [], []
    for symbol, op in ARITHMETIC:
        pairs = [(a, b) for a in range(1, 13) for b in range(1, 13) if symbol not in ("+", "*") or a <= b]
        rng.shuffle(pairs)
        for i, (a, b) in enumerate(pairs):
            prompt = f"{a}{symbol}{b}=".encode()
            answer = f"{op(a, b)}\n".encode()
            task = dict(kind="arithmetic", prompt=prompt, answer=answer, tests=None, family=symbol)
            (check if i < 2 else calibration if i == 2 else fit).append(task)
    for name, symbol, op in PROGRAMS:
        keys = list(range(1, 13))
        rng.shuffle(keys)
        for i, k in enumerate(keys):
            prompt = f"# {name} {k}.\ndef f(x):\n    ".encode()
            answer = f"return x{symbol}{k}\n\n".encode()
            task = dict(kind="python_fn", prompt=prompt, answer=answer,
                        tests=[dict(args=[x], expected=op(x, k)) for x in TEST_INPUTS], family=name)
            (check if i < 2 else calibration if i == 2 else fit).append(task)
    ids = [{t["prompt"] for t in split} for split in (fit, calibration, check)]
    if any(len(keys) != len(split) for keys, split in zip(ids, (fit, calibration, check), strict=True)) or (
            sum(map(len, ids)) != len(set.union(*ids))):
        raise ValueError("task prompts overlap")
    training = [t["prompt"] + t["answer"] for t in fit for _ in range(5 if t["kind"] == "python_fn" else 1)]
    rng.shuffle(training)
    validation = [t["prompt"] + t["answer"] for t in calibration]
    fit_bytes = b"".join(training)
    padding = (1 - len(fit_bytes)) % CONTEXT
    return fit_bytes + b"\n" * padding, b"".join(validation), check, dict(
        fit_tasks=len(fit), calibration_tasks=len(calibration), held_tasks=len(check), shared_task_prompts=0,
        fit_padding_bytes=padding)


def train(root, seed, training, validation):
    fit, val, archive, run = (root / name for name in ("fit.bin", "val.bin", "unified.pt", "train"))
    atomic_bytes(fit, training)
    atomic_bytes(val, validation)
    report = cli("train", "--train-file", fit, "--eval-file", val, "--directory", run, "--device", "cpu",
                 "--dtype", "fp32", "--dim", 64, "--layers", 2, "--context", CONTEXT, "--seed", seed,
                 "--epochs", 1, "--steps-per-epoch", STEPS, "--batch-per-domain", BATCH, "--eval-blocks", 4,
                 "--skill-backbone", "--save-snapshot", archive, "--snapshot-quant", "int8")
    receipt = json.loads((run / "gate17.json").read_bytes())
    raw = archive.read_bytes()
    return fit, raw, dict(report=report, targets=receipt["epochs"][0]["telemetry"]["target_bytes"], updates=receipt["steps"],
                          cold_equal=receipt["cold_equal"], donor_forward_calls=receipt["donor_forward_calls"] +
                          receipt["fresh_evaluation"]["donor_forward_calls"] + report["donor_forward_calls"],
                          archive_bytes=len(raw), fit_sha256=digest(fit), validation_sha256=digest(val))


def run_task(engine, task, seed):
    prompt, kind = task["prompt"], task["kind"]
    check = make_verifier(kind, prompt, task["tests"])
    if not check(task["answer"]).passed:
        raise ValueError("held fixture answer failed verification")
    return engine.run(prompt, check, candidates=BUDGETS[-1], max_bytes=8 if kind == "arithmetic" else 24,
                      stop_bytes=b"\n" if kind == "arithmetic" else b"\n\n", seed=seed,
                      temperature=0.7, top_p=0.9)


def result_records(result):
    return [dict(hex=c.raw.hex(), logprob=c.logprob, **c.verification._asdict(), seconds=c.seconds)
            for c in result.candidates]


def run_arm(engine, tasks, seed):
    results = [run_task(engine, task, seed * 1000 + i * 64) for i, task in enumerate(tasks)]
    receipts, curves = [], {}
    for task, result in zip(tasks, results, strict=True):
        receipts.append(dict(kind=task["kind"], family=task["family"], prompt_hex=task["prompt"].hex(),
                             reference_hex=task["answer"].hex(), tests=task["tests"], candidates=result_records(result)))
    for k in BUDGETS:
        proposals = [r.candidates[:k] for r in results]
        chosen = [SearchResult(candidates, 0).selected for candidates in proposals]
        verified = [item.verification.passed for item in chosen]
        curves[k] = dict(solves=sum(verified), tasks=len(tasks), solve_rate=sum(verified) / len(tasks),
                         arithmetic_solves=sum(v for t, v in zip(tasks, verified, strict=True) if t["kind"] == "arithmetic"),
                         python_solves=sum(v for t, v in zip(tasks, verified, strict=True) if t["kind"] == "python_fn"),
                         decoded_bytes=sum(len(item.raw) for row in proposals for item in row),
                         candidate_seconds=sum(item.seconds for row in proposals for item in row),
                         logprob_only_solves=sum(max(row, key=lambda c: c.logprob).verification.passed for row in proposals))
    rerun = run_task(engine, tasks[0], seed * 1000)
    def stable(result):
        return tuple((c.raw, c.logprob, c.verification) for c in result.candidates)

    return dict(curves=curves, tasks=receipts, exact_repeat=stable(results[0]) == stable(rerun),
                donor_forward_calls=rerun.donor_forward_calls + sum(r.donor_forward_calls for r in results))


def engineering_probe(root, archive):
    fit, path, marker = (root / name for name in ("engineering.bin", "engineering.db", "unsafe-write"))
    prompt = b"123+45="
    atomic_bytes(fit, (prompt + b"168\n") * 80)
    index = SSDEngramStore.build(fit, path, max_order=len(prompt))
    with SSDEngramStore(path, expected_sha256=index["index_sha256"]) as store:
        gain, _ = store.lookup(prompt, max_alpha=0.9)
        engine = VerifiedByteSearch(archive, always_wake=True, engram=store, engram_alpha=0.9)
        selected = engine.run(prompt, make_verifier("arithmetic", prompt), candidates=1, max_bytes=8)
        failures, maximum = [], 0.0
        for seed in range(100):
            rng = random.Random(seed)
            prefix = b"1" + rng.randbytes(2)
            first, second = (engine.forecast_trace(prompt, prefix + rng.randbytes(4)) for _ in range(2))
            cutoff = len(prefix) + 1
            short = engine.forecast_trace(prompt, prefix)
            delta = max(float((first[:cutoff] - second[:cutoff]).abs().max()),
                        float((first[:cutoff] - short).abs().max()))
            if delta:
                failures.append(seed)
            maximum = max(maximum, delta)
    check = make_verifier("python_fn", b"def f(x):", [{"args": [0], "expected": 0}])
    payloads = {"import_call": b"return __import__('os')", "file_write": f"return open({str(marker)!r}, 'w')".encode(),
                "attribute": b"return 0 if x == 0 else x.__class__", "loop": b"while True:\n    pass",
                "power": b"return 2**999999999"}
    blocked = {k: check(value)._asdict() for k, value in payloads.items()}
    return dict(prefix_seeds=100, prefix_failures=failures, prefix_max_diff=maximum, initial_engram_gain=gain,
                baseline_operational=selected.selected.verification.passed and selected.selected.raw == b"168\n",
                donor_forward_calls=selected.donor_forward_calls, blocked=blocked, safety_marker_exists=marker.exists())


def seed_run(root, seed):
    training, validation, tasks, custody = data_for(seed)
    source, archive, trained = train(root, seed, training, validation)
    index_path = root / "fit.sqlite"
    index = SSDEngramStore.build(source, index_path)
    if index["source_sha256"] != trained["fit_sha256"]:
        raise ValueError("Engram index did not bind exact FIT bytes")
    sha = trained["report"]["snapshot_sha256"]
    arms = {"neural": run_arm(VerifiedByteSearch(archive, expected_sha256=sha, always_wake=True), tasks, seed)}
    with SSDEngramStore(index_path, expected_sha256=index["index_sha256"]) as store:
        arms["engram"] = run_arm(VerifiedByteSearch(archive, expected_sha256=sha, always_wake=True, engram=store), tasks, seed)
    probe = engineering_probe(root, archive) if seed == SEEDS[0] else None
    return dict(seed=seed, training=trained, custody=custody, index=index, arms=arms, probe=probe)


def verdict(rows):
    probe = rows[0]["probe"]
    wins = {name: sum(r["arms"][name]["curves"][16]["solves"] > r["arms"][name]["curves"][1]["solves"] for r in rows)
            for name in ("neural", "engram")}
    calls = probe["donor_forward_calls"] + sum(r["training"]["donor_forward_calls"] +
                                             sum(a["donor_forward_calls"] for a in r["arms"].values()) for r in rows)
    exact = all(a["exact_repeat"] and a["curves"][1]["solves"] <= a["curves"][8]["solves"] <= a["curves"][16]["solves"]
                for r in rows for a in r["arms"].values())
    trained = all(t["cold_equal"] and t["targets"] == STEPS * CONTEXT * BATCH and t["updates"] == STEPS
                  and t["report"]["snapshot_quant"] == "int8" and t["report"]["exact_state"]
                  and len(t["report"]["parity"]) == 4 and not any(t["report"]["parity"].values())
                  and t["report"]["round_trip_probabilities"] == 0.0 for t in (r["training"] for r in rows))
    blocked = (not probe["safety_marker_exists"] and all(not row["passed"] and row["error"] for row in probe["blocked"].values()))
    engineering = (not calls and exact and trained and blocked and not probe["prefix_failures"]
                   and probe["prefix_max_diff"] == 0.0 and probe["baseline_operational"] and probe["initial_engram_gain"] > 0)
    scientific = ("INVALID" if not engineering else "PASS" if min(wins.values()) == len(SEEDS) else
                  "MIXED" if max(wins.values()) == len(SEEDS) else "FAIL")
    return dict(engineering="PASS" if engineering else "FAIL", scientific=scientific, strict_win_seeds=wins,
                donor_forward_calls=calls, exact_seeded_repeats=exact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out.resolve()
    intent = out.with_suffix(".intent.json")
    head = git("rev-parse", "HEAD").decode().strip()
    if (out.is_relative_to(ROOT) or out.exists() or intent.exists() or git("status", "--porcelain")
            or git("rev-parse", "HEAD^").decode().strip() != BASE or not git("ls-tree", head, PREREG)):
        parser.error("run from the clean Phase 20 preregistration commit with a fresh external output")
    binding = dict(base=BASE, source_commit=head, source_tree=git("rev-parse", "HEAD^{tree}").decode().strip(),
                   prereg_sha256=digest(ROOT / PREREG))
    atomic(intent, binding)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix="phase20-verified-search-") as scratch:
        for seed in SEEDS:
            root = Path(scratch) / str(seed)
            root.mkdir()
            rows.append(seed_run(root, seed))
            print(f"Phase 20 seed {seed} complete", flush=True)
    if git("rev-parse", "HEAD").decode().strip() != head or git("status", "--porcelain"):
        parser.error("the committed source changed during scoring")
    result = dict(schema="nexus.phase20-verified-search.v1", **binding, offline=True, torch=str(torch.__version__),
                  platform=platform.platform(), seconds=time.perf_counter() - started, peak_rss_bytes=peak_rss(),
                  seeds=rows, summary=verdict(rows))
    atomic(out, result)
    print(json.dumps(result["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
