"""Frozen five-seed CPU benchmark for the local five-category scoreboard."""

import argparse
import hashlib
import json
from pathlib import Path
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
from benchmarks.phase20_verified_search import BATCH, CONTEXT, STEPS, train
from benchmarks.phase3_multi_skill_routing import git
from nexus_expanse.livebench_eval import CATEGORIES, dev_records, evaluate_budgets, load_tasks_jsonl, suite_from_rows
from nexus_expanse.verified_search import VerifiedByteSearch

BASE = "61d238f84334c9716c2d1c4a56388f83cf1f3e20"
PREREG = "docs/PHASE21_LIVEBENCH_VERIFIERS_PREREG.md"
SEEDS = tuple(range(3501, 3506))
BUDGETS = (1, 16)


def fit_corpus(fit, val, seed):
    records = [row["turns"][0].encode() + expected for row, expected in fit
               for _ in range(1 if row["category"] == "math" else 5)]
    random.Random(seed + 100_000).shuffle(records)
    raw = b"".join(records)
    padding = (1 - len(raw)) % CONTEXT
    validation = b"".join(row["turns"][0].encode() + expected for row, expected in val)
    return raw + b"\n" * padding, validation, padding


def prefix_probe(engine, task):
    failures, max_diff = [], 0.0
    for seed in range(100):
        rng = random.Random(seed)
        prefix = rng.randbytes(4)
        a, b = (engine.forecast_trace(task.prompt, prefix + rng.randbytes(5)) for _ in range(2))
        short = engine.forecast_trace(task.prompt, prefix)
        n = len(prefix) + 1
        delta = max(float((a[:n] - b[:n]).abs().max()), float((a[:n] - short).abs().max()))
        if delta:
            failures.append(seed)
        max_diff = max(max_diff, delta)
    return dict(seeds=100, failures=failures, max_diff=max_diff)


def canonical(board):
    return {key: value for key, value in board.items() if key != "candidate_seconds"}


def seed_run(root, seed):
    fit, val, check = dev_records(seed)
    prompts = [{record["turns"][0] for record, _ in rows} for rows in (fit, val, check)]
    if (sum(map(len, prompts)) != len(set.union(*prompts))
            or any(len(values) != len(rows) for values, rows in zip(prompts, (fit, val, check), strict=True))):
        raise ValueError("task prompt split leak")
    data, validation, padding = fit_corpus(fit, val, seed)
    _, archive, training = train(root, seed, data, validation)
    suite = suite_from_rows(row for row, _ in check)
    references = {row["question_id"]: label for row, label in check}
    if suite.unsupported or len(suite.tasks) != 30 or any(not t.verify(references[t.id]).passed for t in suite.tasks):
        raise ValueError("held-out fixture does not match local verifiers")
    rows = b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row, _ in check)
    path = root / "check.jsonl"
    atomic_bytes(path, rows)
    engine = VerifiedByteSearch(archive, expected_sha256=training["report"]["snapshot_sha256"], always_wake=True)
    seed_start = seed * 1000
    scoreboards, wall = evaluate_budgets(engine, suite, BUDGETS, seed=seed_start)
    replay, replay_wall = evaluate_budgets(engine, load_tasks_jsonl(path), BUDGETS, seed=seed_start)
    exact = all(canonical(scoreboards[k]) == canonical(replay[k]) for k in BUDGETS)
    prefix = prefix_probe(engine, suite.tasks[0]) if seed == SEEDS[0] else None
    return dict(seed=seed, training=training, custody=dict(fit_tasks=len(fit), calibration_tasks=len(val),
                check_tasks=len(check), shared_task_prompts=0, fit_padding_bytes=padding,
                check_sha256=hashlib.sha256(rows).hexdigest()), scoreboards=scoreboards, prefix=prefix,
                file_replay_exact=exact, wall_seconds=wall, replay_wall_seconds=replay_wall,
                replay_donor_forward_calls=replay[16]["donor_forward_calls"])


def verdict(rows):
    prefix = rows[0]["prefix"]
    wins = sum(r["scoreboards"][16]["overall_score"] > r["scoreboards"][1]["overall_score"] for r in rows)
    forwards = sum(row["training"]["donor_forward_calls"] + row["scoreboards"][16]["donor_forward_calls"] +
                   row["replay_donor_forward_calls"] for row in rows)
    archives = all(r["training"]["cold_equal"] and r["training"]["targets"] == CONTEXT*STEPS*BATCH
                   and r["training"]["updates"] == STEPS and r["training"]["report"]["exact_state"]
                   and r["training"]["report"]["snapshot_quant"] == "int8"
                   and r["training"]["report"]["round_trip_probabilities"] == 0.0
                   and len(r["training"]["report"]["parity"]) == 4
                   and not any(r["training"]["report"]["parity"].values()) for r in rows)
    exact = all(row["file_replay_exact"] and row["custody"]["shared_task_prompts"] == 0
                and all(board["counts"][c]["total"] == 6 and not board["unsupported"]
                        for board in row["scoreboards"].values() for c in CATEGORIES) for row in rows)
    engineering = not forwards and archives and exact and not prefix["failures"] and prefix["max_diff"] == 0.0
    science = "INVALID" if not engineering else "PASS" if wins == len(SEEDS) else "MIXED" if wins else "FAIL"
    return dict(engineering="PASS" if engineering else "FAIL", scientific=science, strict_win_seeds=wins,
                donor_forward_calls=forwards, replay_exact=exact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out.resolve()
    intent = out.with_suffix(".intent.json")
    head = git("rev-parse", "HEAD").decode().strip()
    if (out.is_relative_to(ROOT) or out.exists() or intent.exists() or git("status", "--porcelain")
            or git("rev-parse", "HEAD^").decode().strip() != BASE or not git("ls-tree", head, PREREG)):
        parser.error("run from the clean Phase 21 preregistration commit with a fresh external output")
    binding = dict(base=BASE, source_commit=head, source_tree=git("rev-parse", "HEAD^{tree}").decode().strip(),
                   prereg_sha256=digest(ROOT / PREREG))
    atomic(intent, binding)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix="phase21-livebench-scoreboard-") as temp:
        for seed in SEEDS:
            root = Path(temp) / str(seed)
            root.mkdir()
            rows.append(seed_run(root, seed))
            print(f"Phase 21 seed {seed} complete", flush=True)
    if git("rev-parse", "HEAD").decode().strip() != head or git("status", "--porcelain"):
        parser.error("the scored source was changed")
    result = dict(schema="nexus.phase21-livebench-scoreboard.v1", **binding, offline=True,
                  platform=platform.platform(), torch=str(torch.__version__), seconds=time.perf_counter()-start,
                  peak_rss_bytes=peak_rss(), seeds=rows, summary=verdict(rows))
    atomic(out, result)
    print(json.dumps(result["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
