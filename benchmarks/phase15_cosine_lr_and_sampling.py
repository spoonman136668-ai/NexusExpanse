"""Compare matched constant and cosine training with exact resume on CPU."""

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import gate17_gpu_scaling_harness as scaling
from benchmarks.dev_artifacts import atomic, atomic_bytes, digest
from benchmarks.gate_custody import peak_rss, update
from benchmarks.phase13_warmstart_train import build_corpus, cli
from benchmarks.phase3_multi_skill_routing import SOURCE, git
from nexus_expanse.cli import main as cli_main
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.patching import CompactBigram
from nexus_expanse.shard_stream_assimilation import make_student
from nexus_expanse.unified import NexusExpanseUnified
from nexus_expanse.weight_editing import skill_snapshot

BASE = "e124207417be94c3ea242a8c5042067860ac23f5"
PREREG = "docs/PHASE15_COSINE_LR_AND_SAMPLING_PREREG.md"
SEEDS = range(2901, 2906)
EPOCHS, STEPS, CONTEXT = 4, 32, 32
SCHEDULES = ("constant", "cosine")
DECODE_BYTES = 16


def fit_skill(corpus, seed, path):
    cpu = torch.device("cpu")
    geometry = scaling.Geometry(seed, 32, CONTEXT, 1, STEPS, 1, 1)
    train = {domain: (corpus / f"{domain}-train.bin").read_bytes() for domain in scaling.DOMAINS}
    with forbid_donor_forward() as guard:
        skill = make_student(seed + 50_000).train()
        optimizer = torch.optim.AdamW(skill.parameters(), lr=scaling.DEFAULT_LR, weight_decay=0.01)
        losses = []
        for step in range(STEPS):
            rows = scaling.training_rows(train, geometry, step)
            batch, keep, _ = scaling.make_batch(rows, CONTEXT, cpu)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(skill(batch[:, :-1])[keep], batch[:, 1:][keep])
            losses.append(update(skill, optimizer, loss))
        atomic_bytes(path, skill_snapshot(skill.eval()))
    return dict(sha256=digest(path), updates=STEPS, losses=losses, donor_forward_calls=guard.donor_forward_calls)


def space_snapshot(seed):
    """A native fitted byte table with a higher score for space than period."""
    geometry = scaling.Geometry(seed, 32, CONTEXT, 1, 1, 1, 1)
    model = scaling.model_for(geometry, "causal_attention_monarch", torch.device("cpu"))
    counts = np.zeros((256, 256), dtype="<u4")
    counts[:, ord(" ")] = 90 + seed % 5
    counts[:, ord(".")] = 80
    return NexusExpanseUnified(model, CompactBigram.from_dense(counts)).snapshot()


def call_cli(*args):
    with redirect_stdout(StringIO()) as output:
        cli_main(list(map(str, args)))
    return json.loads(output.getvalue())


def train_arm(root, corpus, seed, skill, schedule, *, snapshot=None):
    common = ["train", "--corpus-dir", corpus, "--init-skill", skill, "--device", "cpu", "--dtype", "fp32",
              "--seed", seed, "--dim", 32, "--layers", 2, "--context", CONTEXT, "--epochs", EPOCHS,
              "--steps-per-epoch", STEPS, "--batch-per-domain", 1, "--eval-blocks", 48,
              "--lr", scaling.DEFAULT_LR, "--lr-schedule", schedule]
    direct, resumed = root / "direct", root / "resumed"
    export_options = ["--save-snapshot", snapshot] if snapshot else []
    exported = cli(*common, "--directory", direct, "--epoch-limit", EPOCHS, *export_options)
    for _ in range(EPOCHS):
        cli(*common, "--directory", resumed, "--epoch-limit", 1)
    reference, restarted = (json.loads((path / "gate17.json").read_bytes()) for path in (direct, resumed))
    direct_epochs, resumed_epochs = reference["epochs"], restarted["epochs"]
    workers = [epoch["telemetry"]["worker_pid"] for epoch in resumed_epochs]
    epoch_equal = [a["sha256"] == b["sha256"] and a["losses"] == b["losses"]
                   for a, b in zip(direct_epochs, resumed_epochs, strict=True)]
    rates = [[rate for epoch in report["epochs"] for rate in epoch["telemetry"].get("learning_rates", [])]
             for report in (reference, restarted)]
    expected = ([scaling.cosine_lr(step, EPOCHS * STEPS, scaling.DEFAULT_LR)
                 for step in range(1, EPOCHS * STEPS + 1)] if schedule == "cosine" else [])
    exact = (len(set(workers)) == EPOCHS and len({e["telemetry"]["worker_pid"] for e in direct_epochs}) == 1
             and all(epoch_equal) and rates[0] == rates[1] == expected
             and reference["final_hash"] == restarted["final_hash"])
    validation = reference["fresh_evaluation"]["validation"]
    total_bytes = sum(v["target_bytes"] for v in validation.values())
    pooled = sum(v["bpb"] * v["target_bytes"] for v in validation.values()) / total_bytes
    verified = call_cli("verify", "--snapshot", snapshot, "--snapshot-sha256", exported["snapshot_sha256"]) if snapshot else None
    return dict(direct=reference, resumed=restarted, exact_resume=exact, epoch_state_and_losses_equal=epoch_equal,
                pooled_bpb=pooled, export=exported if snapshot else None, verify=verified)


def generation(seed, root, snapshot):
    case = root / "space.pt"
    atomic_bytes(case, space_snapshot(seed))
    prompt = (b"." + b" " * 8).hex()
    options = ("--prompt-hex", prompt, "--max-bytes", DECODE_BYTES, "--seed", seed, "--repeat-window", 1)
    plain = call_cli("generate", "--snapshot", case, *options, "--repetition-penalty", 1.0)
    broken = call_cli("generate", "--snapshot", case, *options, "--repetition-penalty", 2.0)
    draws = [call_cli("generate", "--snapshot", case, *options, "--repetition-penalty", 2.0,
                      "--temperature", 0.7, "--top-p", 0.9) for _ in range(2)]
    generated = call_cli("generate", "--snapshot", snapshot, *options, "--repetition-penalty", 1.0)
    streamed = call_cli("stream", "--snapshot", snapshot, "--input-hex", prompt, "--no-adapt",
                        "--continuation", DECODE_BYTES)
    collapse_broken = plain["generated_hex"] == (b" " * DECODE_BYTES).hex() and (
        broken["generated_hex"] == (b". " * (DECODE_BYTES // 2)).hex())
    return dict(plain=plain, penalized=broken, stochastic=draws, disabled=generated,
                stream_hex=streamed["continuation_hex"], collapse_broken=collapse_broken,
                seeded_equal=draws[0]["generated_hex"] == draws[1]["generated_hex"],
                disabled_equal=generated["generated_hex"] == streamed["continuation_hex"],
                donor_forward_calls=sum(report["donor_forward_calls"] for report in (plain, broken, *draws, generated)))


def seed_run(root, seed):
    corpus = root / "corpus"
    provenance = build_corpus(corpus, seed)
    skill = root / "fit-skill.pt"
    initializer = fit_skill(corpus, seed, skill)
    snapshot = root / "cosine.pt"
    arms = {name: train_arm(root / name, corpus, seed, skill, name, snapshot=snapshot if name == "cosine" else None)
            for name in SCHEDULES}
    sample = generation(seed, root, snapshot)
    reports = [arm[key] for arm in arms.values() for key in ("direct", "resumed")]
    cold = [report["fresh_evaluation"] for report in reports]
    verified = arms["cosine"]["verify"]
    exact = (all(arm["exact_resume"] for arm in arms.values()) and all(report["cold_equal"] for report in reports)
             and all(set(report["parity"].values()) == {0.0} for report in cold)
             and verified["exact_state"] and set(verified["parity"].values()) == {0.0}
             and verified["round_trip_probabilities"] == 0.0)
    calls = initializer["donor_forward_calls"] + sample["donor_forward_calls"]
    calls += sum(report["donor_forward_calls"] for report in reports + cold)
    return dict(seed=seed, corpus_provenance=provenance, fit_skill=initializer, arms=arms, generation=sample,
                exact_training=exact, donor_forward_calls=calls)


def verdict(rows):
    changes = [row["arms"]["cosine"]["pooled_bpb"] - row["arms"]["constant"]["pooled_bpb"] for row in rows]
    wins = sum(change < 0 for change in changes)
    average = statistics.mean(changes)
    exact = sum(row["exact_training"] for row in rows)
    calls = sum(row["donor_forward_calls"] for row in rows)
    science = "MIXED"
    if exact != len(SEEDS) or calls:
        science = "INVALID"
    elif wins == len(SEEDS) and average <= -0.01:
        science = "PASS"
    elif wins == 0 and average >= 0.01:
        science = "FAIL"
    disabled = sum(row["generation"]["disabled_equal"] for row in rows)
    collapsed = sum(row["generation"]["collapse_broken"] and row["generation"]["seeded_equal"] for row in rows)
    return dict(scientific=science, cosine_wins=wins, mean_pooled_delta_bpb=average, exact_training_seeds=exact,
                disabled_parity_seeds=disabled, collapse_broken_and_seeded_seeds=collapsed, donor_forward_calls=calls,
                scored_engineering="PASS" if exact == disabled == collapsed == len(SEEDS) and not calls else "FAIL")


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
        parser.error("run once from the clean Phase 15 preregistration commit with a new external output")
    identity = dict(base=BASE, source_commit=head, source_tree=tree, prereg_sha256=digest(ROOT / PREREG))
    atomic(intent, identity)
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix="nexus-phase15-") as scratch:
        for seed in SEEDS:
            root = Path(scratch) / str(seed)
            root.mkdir()
            rows.append(seed_run(root, seed))
            print(f"Phase 15 seed {seed} complete", flush=True)
    if git("rev-parse", "HEAD").decode().strip() != head or git("status", "--porcelain"):
        parser.error("the committed source changed during scoring")
    result = dict(schema="nexus.phase15-cosine-lr.v1", **identity, corpus_source_sha=SOURCE,
                  platform=platform.platform(), torch=str(torch.__version__), device="cpu", threads=1,
                  external_downloads=0, wall_seconds=time.perf_counter() - started,
                  parent_peak_rss_bytes=peak_rss(), seeds=rows, summary=verdict(rows))
    atomic(out, result)
    print(json.dumps(result["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
