"""Match training updates and raw target bytes for focused and mixed prose BPB."""

import argparse
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import gate17_gpu_scaling_harness as scaling
from benchmarks.dev_artifacts import atomic, atomic_bytes, digest
from benchmarks.gate_custody import peak_rss
from benchmarks.phase13_warmstart_train import build_corpus, cli
from benchmarks.phase15_cosine_lr_and_sampling import call_cli
from benchmarks.phase3_multi_skill_routing import SOURCE, git
from scripts.prepare_gpu_corpus import CorpusOverlapError, assert_disjoint
from scripts.prepare_real_corpus import prepare_single

BASE = "a4b1b390bcbf2a1fd82cdad33888d5b5b2b15af8"
PREREG = "docs/PHASE17_DOMAIN_FOCUS_PREREG.md"
SEEDS = tuple(range(3101, 3106))
EPOCHS, STEPS, CONTEXT = 2, 32, 32


def match_corpus(root, seed):
    corpus = root / "corpus"
    rejected = []
    for attempt in range(8):
        corpus_seed = seed + 10_000 * attempt
        original = root / f"untrimmed-{corpus_seed}"
        try:
            provenance = build_corpus(original, corpus_seed)
        except CorpusOverlapError as exc:
            rejected.append(dict(seed=corpus_seed, reason=str(exc)))
        else:
            break
    else:
        raise ValueError("no leakage-free fixture within the preregistered corpus candidates")
    corpus.mkdir()
    train, held = {}, {}
    for path in sorted(original.iterdir()):
        raw = path.read_bytes()
        split = train if path.name.endswith("-train.bin") else held
        if split is train:
            # Every selectable block contains the same number of valid targets.
            raw = raw[:((len(raw) - 1) // CONTEXT) * CONTEXT + 1]
        split[path.name] = raw
        atomic_bytes(corpus / path.name, raw)
    assert_disjoint(train.values(), held)
    return corpus, dict(source=provenance, corpus_seed=corpus_seed, rejected=rejected,
                        sha256={p.name: digest(p) for p in sorted(corpus.iterdir())})


def binary_corpus(root, seed):
    raw = root / "capture.bin"
    atomic_bytes(raw, random.Random(seed + 100_000).randbytes(4 * 65536))
    directory = root / "packed"
    manifest = prepare_single(directory, local=raw, seed=seed, limit=4 * 65536)
    train, held = directory / "stream-train.bin", directory / "stream-eval.bin"
    assert_disjoint((train.read_bytes(),), {"stream": held.read_bytes()})
    return train, held, manifest


def train_run(directory, seed, inputs, *, resume=False, export=False):
    args = ["train", *inputs, "--directory", directory, "--device", "cpu", "--dtype", "fp32",
            "--skill-backbone", "--seed", seed, "--dim", 32, "--layers", 2, "--context", CONTEXT,
            "--epochs", EPOCHS, "--steps-per-epoch", STEPS, "--batch-per-domain", 1, "--eval-blocks", 48,
            "--lr", 0.003, "--lr-schedule", "cosine", "--epoch-limit", 1 if resume else EPOCHS]
    snapshot = directory / "unified.pt"
    if resume:
        for _ in range(EPOCHS - 1):
            cli(*args)
    exported = cli(*args, "--save-snapshot", snapshot, "--snapshot-quant", "int8") if export else cli(*args)
    result = json.loads((directory / "gate17.json").read_bytes())
    return dict(receipt=result, export=exported if export else None), snapshot


def exact_snapshot(item):
    return (item["snapshot_quant"] == "int8" and item["exact_state"] and item["round_trip_probabilities"] == 0.0
            and len(item["parity"]) == 4 and set(item["parity"].values()) == {0.0})


def resume_check(root, seed, inputs):
    direct, _ = train_run(root / "direct", seed, inputs, export=True)
    resumed, snapshot = train_run(root / "resumed", seed, inputs, resume=True, export=True)
    a, b = direct["receipt"], resumed["receipt"]
    epochs_a, epochs_b = a["epochs"], b["epochs"]
    exact_epochs = [x["sha256"] == y["sha256"] and x["losses"] == y["losses"]
                    for x, y in zip(epochs_a, epochs_b, strict=True)]
    exact = (len({row["telemetry"]["worker_pid"] for row in epochs_a}) == 1
             and len({row["telemetry"]["worker_pid"] for row in epochs_b}) == EPOCHS
             and all(exact_epochs) and a["cold_equal"] and b["cold_equal"] and a["final_hash"] == b["final_hash"]
             and direct["export"]["snapshot_sha256"] == resumed["export"]["snapshot_sha256"])
    verify = call_cli("verify", "--snapshot", snapshot, "--snapshot-sha256", resumed["export"]["snapshot_sha256"])
    generate = call_cli("generate", "--snapshot", snapshot, "--prompt-hex", "ff00417f", "--max-bytes", 8, "--always-wake")
    exact = exact and all(exact_snapshot(item) for item in (direct["export"], resumed["export"], verify))
    exact = exact and generate["always_wake"] and generate["generated_bytes"] == 8
    calls = sum(item["donor_forward_calls"] for item in (a, b, a["fresh_evaluation"], b["fresh_evaluation"],
                                                         direct["export"], resumed["export"], verify, generate))
    return dict(direct=direct, resumed=resumed, exact_epochs=exact_epochs, exact_resume_and_export=exact,
                verify=verify, generate=generate, donor_forward_calls=calls)


def seed_run(root, seed):
    corpus, provenance = match_corpus(root, seed)
    train, held, packed = binary_corpus(root, seed)
    mixed, _ = train_run(root / "mixed", seed, ("--corpus-dir", corpus, "--focus-domain", "all"))
    focus = resume_check(root / "focus", seed, ("--corpus-dir", corpus, "--focus-domain", "prose"))
    single = resume_check(root / "stream", seed, ("--train-file", train, "--eval-file", held))
    a, b = mixed["receipt"], focus["direct"]["receipt"]
    expected = EPOCHS * STEPS * len(scaling.DOMAINS) * CONTEXT
    target_a, target_b = (sum(e["telemetry"]["target_bytes"] for e in item["epochs"]) for item in (a, b))
    cold = a["fresh_evaluation"]
    calls = a["donor_forward_calls"] + cold["donor_forward_calls"] + sum(
        r["donor_forward_calls"] for r in (focus, single))
    exact = (a["cold_equal"] and set(cold["parity"].values()) == {0.0} and a["initial_hash"] == b["initial_hash"]
             and a["steps"] == b["steps"] == EPOCHS * STEPS and target_a == target_b == expected
             and "focus_domain" not in a["identity"] and b["identity"]["focus_domain"] == "prose"
             and packed["global_overlap_32b"] == 0)
    return dict(seed=seed, corpus=provenance, single_manifest=packed, mixed=mixed, focus=focus, single=single,
                target_bytes=dict(mixed=target_a, focus=target_b), matched_training=exact,
                prose_bpb={"mixed": cold["validation"]["prose"]["bpb"],
                           "focus": b["fresh_evaluation"]["validation"]["prose"]["bpb"]}, donor_forward_calls=calls)


def verdict(rows):
    deltas = [row["prose_bpb"]["focus"] - row["prose_bpb"]["mixed"] for row in rows]
    wins = sum(delta < 0 for delta in deltas)
    exact = sum(row["matched_training"] and row["focus"]["exact_resume_and_export"]
                and row["single"]["exact_resume_and_export"] for row in rows)
    calls = sum(row["donor_forward_calls"] for row in rows)
    engineering = exact == len(SEEDS) and calls == 0
    scientific = "INVALID" if not engineering else "PASS" if wins == len(SEEDS) else "MIXED" if wins >= 3 else "FAIL"
    return dict(scored_engineering="PASS" if engineering else "FAIL", scientific=scientific, focused_wins=wins,
                mean_focused_minus_mixed_prose_bpb=statistics.mean(deltas), exact_seeds=exact, donor_forward_calls=calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out.resolve()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    head = git("rev-parse", "HEAD").decode().strip()
    tree = git("rev-parse", "HEAD^{tree}").decode().strip()
    intent = out.with_suffix(".intent.json")
    if (out.is_relative_to(ROOT) or out.exists() or intent.exists() or git("status", "--porcelain")
            or git("rev-parse", "HEAD^").decode().strip() != BASE or not git("ls-tree", head, PREREG)):
        parser.error("run once from the clean Phase 17 preregistration commit with a new external output")
    identity = dict(base=BASE, source_commit=head, source_tree=tree, prereg_sha256=digest(ROOT / PREREG))
    atomic(intent, identity)
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix="nexus-phase17-") as scratch:
        for seed in SEEDS:
            root = Path(scratch) / str(seed)
            root.mkdir()
            rows.append(seed_run(root, seed))
            print(f"Phase 17 seed {seed} complete", flush=True)
    if git("rev-parse", "HEAD").decode().strip() != head or git("status", "--porcelain"):
        parser.error("the committed source changed during scoring")
    report = dict(schema="nexus.phase17-domain-focus.v1", **identity, corpus_source_sha=SOURCE,
                  platform=platform.platform(), torch=str(torch.__version__), device="cpu", threads=1,
                  external_downloads=0, wall_seconds=time.perf_counter() - started,
                  parent_peak_rss_bytes=peak_rss(), seeds=rows, summary=verdict(rows))
    atomic(out, report)
    print(json.dumps(report["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
