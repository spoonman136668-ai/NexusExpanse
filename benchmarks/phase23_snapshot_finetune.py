"""Frozen Phase 23 CPU DEV comparison of pretrained and curriculum-adapted snapshots."""

import argparse
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.dev_artifacts import atomic, digest
from benchmarks.gate_custody import peak_rss
from benchmarks.phase13_warmstart_train import DOMAINS, cli
from benchmarks.phase17_domain_focus import match_corpus
from benchmarks.phase20_verified_search import BATCH, CONTEXT, STEPS, train
from benchmarks.phase21_livebench_scoreboard import prefix_probe
from benchmarks.phase3_multi_skill_routing import SOURCE, git
from nexus_expanse.livebench_eval import CATEGORIES, builtin_dev_suite, evaluate_budgets
from nexus_expanse.verified_search import VerifiedByteSearch
from scripts.prepare_gpu_corpus import assert_disjoint
from scripts.prepare_reasoning_curriculum import builtin_references, prepare

BASE = "6749dc8b25082eaf66eeb7a6eac1dc073f6748b2"
PREREG = "docs/PHASE23_SNAPSHOT_FINETUNE_PREREG.md"
SEEDS = tuple(range(3701, 3706))
BUDGETS = (1, 16)
CURRICULUM_CAP = 48_000
EPOCHS, EPOCH_STEPS = 2, 64


def base_inputs(root, seed):
    corpus, info = match_corpus(root, seed)
    fit, val = (b"".join((corpus / f"{c}-{suffix}.bin").read_bytes() for c in DOMAINS)
                for suffix in ("train", "eval"))
    fit = fit[:((len(fit)-1)//CONTEXT)*CONTEXT+1]
    assert_disjoint((fit,), {"pretraining calibration":val})
    assert_disjoint((fit, val), builtin_references())
    return fit, val, info


def fine_tune(root, base_path, corpus, *, seed, resumed):
    run = root / "training"
    archive = root / "unified.pt"
    args = ("train", "--train-file", corpus / "stream-train.bin", "--eval-file", corpus / "stream-eval.bin",
            "--init-snapshot", base_path, "--directory", run, "--device", "cpu", "--dtype", "fp32", "--seed", seed,
            "--epochs", EPOCHS, "--steps-per-epoch", EPOCH_STEPS, "--batch-per-domain", BATCH, "--eval-blocks", 4,
            "--epoch-limit", 1 if resumed else EPOCHS, "--lr", 0.001, "--lr-schedule", "cosine")
    if resumed:
        cli(*args)
    export = cli(*args, "--save-snapshot", archive, "--snapshot-quant", "int8")
    receipt = json.loads((run / "gate17.json").read_bytes())
    receipt["epoch_init_digests"] = [json.loads((run / f"gate17-epoch-{i:02}.intent.json").read_bytes())
                                     ["identity"]["init_snapshot_sha256"] for i in range(1,EPOCHS+1)]
    return archive.read_bytes(), dict(export=export, receipt=receipt)


def exact_snapshot(report):
    return (report["snapshot_quant"] == "int8" and report["exact_state"] and report["round_trip_probabilities"] == 0.0
            and len(report["parity"]) == 4 and not any(report["parity"].values()))


def resume_checks(a, b, init_sha):
    left, right = (arm["receipt"] for arm in (a,b))

    def projection(row):
        stats = row["telemetry"]
        return (row["sha256"], row["losses"], stats["start_model_hash"], stats["end_model_hash"],
                stats["target_bytes"], stats["learning_rates"])

    epochs = [projection(x) == projection(y) for x,y in zip(left["epochs"],right["epochs"],strict=True)]
    valid = all(r["cold_equal"] and r["identity"]["init_snapshot_sha256"] == init_sha
                and r["epoch_init_digests"] == [init_sha] * EPOCHS and r["steps"] == EPOCHS*EPOCH_STEPS
                and not any(r["fresh_evaluation"]["parity"].values()) for r in (left,right))
    distinct = len({r["telemetry"]["worker_pid"] for r in right["epochs"]}) == EPOCHS
    return bool(valid and distinct and all(epochs) and left["initial_hash"] == right["initial_hash"]
                and left["final_hash"] == right["final_hash"] and left["checkpoint_sha256"] == right["checkpoint_sha256"]
                and all(arm["export"]["init_snapshot_sha256"] == init_sha and exact_snapshot(arm["export"]) for arm in (a,b))
                and a["export"]["snapshot_sha256"] == b["export"]["snapshot_sha256"])


def seed_run(root, seed):
    source, calibration, provenance = base_inputs(root, seed)
    pretrain = root / "pretrain"
    pretrain.mkdir()
    _, initial, base_training = train(pretrain, seed, source, calibration)
    path = pretrain / "unified.pt"
    sha = base_training["report"]["snapshot_sha256"]
    curriculum = root / "curriculum"
    manifest = prepare(curriculum, size_bytes=CURRICULUM_CAP, seed=seed)
    fit, held = ((curriculum / f"stream-{name}.bin").read_bytes() for name in ("train","eval"))
    assert_disjoint((fit,), {"curriculum eval":held})
    assert_disjoint((fit,held),builtin_references())
    for name, entry in manifest["files"].items():
        if entry["sha256"] != digest(curriculum / name):
            raise ValueError("curriculum digest mismatch")
    one, direct = fine_tune(root / "direct", path, curriculum, seed=seed, resumed=False)
    two, resumed = fine_tune(root / "resumed", path, curriculum, seed=seed, resumed=True)
    exact = one == two and resume_checks(direct, resumed, sha)
    suite = builtin_dev_suite()
    base = VerifiedByteSearch(initial, expected_sha256=sha, always_wake=True, no_episodic=True)
    tuned = VerifiedByteSearch(two, expected_sha256=resumed["export"]["snapshot_sha256"], always_wake=True, no_episodic=True)
    results, seconds = {}, {}
    for name, model in (("base",base),("fine_tuned",tuned)):
        results[name], seconds[name] = evaluate_budgets(model, suite, BUDGETS, seed=seed*1000)
    prefix = prefix_probe(tuned, suite.tasks[0]) if seed == SEEDS[0] else None
    return dict(seed=seed, base_training=base_training, base_corpus=provenance, curriculum=manifest,
                training=dict(direct=direct,resumed=resumed), exact_resume=exact,
                scoreboards=results, evaluation_wall_seconds=seconds, prefix=prefix)


def verdict(rows):
    prefix = rows[0]["prefix"]
    wins = {k:sum(r["scoreboards"]["fine_tuned"][k]["overall_score"] > r["scoreboards"]["base"][k]["overall_score"]
                  for r in rows) for k in BUDGETS}
    gains = []
    for category in CATEGORIES[1:]:
        before, after = (sum(r["scoreboards"][arm][16]["counts"][category]["passed"] for r in rows)
                         for arm in ("base", "fine_tuned"))
        if after > before:
            gains.append(category)
    calls = sum(r["base_training"]["donor_forward_calls"] + sum(arm[16]["donor_forward_calls"]
                for arm in r["scoreboards"].values()) + sum(a["export"]["donor_forward_calls"] +
                a["receipt"]["fresh_evaluation"]["donor_forward_calls"] for a in r["training"].values()) for r in rows)
    exact = all(r["exact_resume"] and r["base_training"]["cold_equal"] and exact_snapshot(r["base_training"]["report"])
                and r["base_training"]["updates"] == STEPS and r["base_training"]["targets"] == CONTEXT*STEPS*BATCH
                and not any(r["curriculum"]["shared_32_byte_windows"].values()) and r["curriculum"]["verifier_passed"] ==
                sum(e["examples"] for e in r["curriculum"]["files"].values()) for r in rows)
    engineering = exact and not calls and not prefix["failures"] and prefix["max_diff"] == 0.0
    science = ("INVALID" if not engineering else "PASS" if min(wins.values()) == len(SEEDS) and len(gains) >= 2
               else "MIXED" if any(wins.values()) or gains else "FAIL")
    return dict(engineering="PASS" if engineering else "FAIL", scientific=science, strict_win_seeds=wins,
                gained_nonmath_categories=gains, exact_resume_seeds=sum(r["exact_resume"] for r in rows), donor_forward_calls=calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out.resolve()
    intent = out.with_suffix(".intent.json")
    head = git("rev-parse", "HEAD").decode().strip()
    if (out.is_relative_to(ROOT) or out.exists() or intent.exists() or git("status", "--porcelain")
            or git("rev-parse", "HEAD^").decode().strip() != BASE or not git("ls-tree", head, PREREG)):
        parser.error("run from the clean Phase 23 preregistration commit with a fresh external output")
    binding = dict(base=BASE, source_commit=head, source_tree=git("rev-parse", "HEAD^{tree}").decode().strip(),
                   prereg_sha256=digest(ROOT / PREREG), base_corpus_source_sha=SOURCE)
    atomic(intent, binding)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix="phase23-snapshot-finetune-") as tmp:
        for seed in SEEDS:
            root = Path(tmp) / str(seed)
            root.mkdir()
            rows.append(seed_run(root, seed))
            print(f"Phase 23 seed {seed} complete", flush=True)
    if git("rev-parse", "HEAD").decode().strip() != head or git("status", "--porcelain"):
        parser.error("the scored source changed")
    result = dict(schema="nexus.phase23-snapshot-finetune.v1", **binding, offline=True,
                  platform=platform.platform(), torch=str(torch.__version__), seconds=time.perf_counter()-started,
                  peak_rss_bytes=peak_rss(), seeds=rows, summary=verdict(rows))
    atomic(out, result)
    print(json.dumps(result["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
