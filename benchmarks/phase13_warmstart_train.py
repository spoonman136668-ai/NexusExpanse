"""Preregistered Phase 13 warm-start comparison through the native training CLI."""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.dev_artifacts import atomic, atomic_bytes, digest
from benchmarks.gate_custody import peak_rss
from benchmarks.phase3_multi_skill_routing import SOURCE, git
from benchmarks.phase9_weight_edit_and_assimilate import frozen_data
from benchmarks.phase10b_shard_stream_and_edit import produce_cross_width_donor
from nexus_expanse.shard_stream_assimilation import assimilate_zero_exec
from scripts.prepare_gpu_corpus import assert_disjoint, corpus_bytes

BASE = "bbdc3c3ae4f43e4677d111ba38bb7bccf4b3de9b"
PREREG = "docs/PHASE13_WARMSTART_AND_GENERATE_PREREG.md"
SEEDS = tuple(range(2701, 2706))
ARMS = ("warmstart_train", "scratch_train", "shuffled_init_train")
DOMAINS = ("math", "code", "prose")


def build_corpus(root, seed):
    root.mkdir()
    splits, provenance = frozen_data(seed)
    for split, source, windows in (("train", "fit", 96), ("eval", "check", 24)):
        for domain, name in zip(DOMAINS, (None, "base_python", "markdown_prose"), strict=True):
            raw = (corpus_bytes(seed, split, domain, 64 * windows) if name is None
                   else splits[source][name].to(torch.uint8).numpy().tobytes())
            atomic_bytes(root / f"{domain}-{split}.bin", raw)
    assert_disjoint(
        [(root / f"{domain}-train.bin").read_bytes() for domain in DOMAINS],
        {domain: (root / f"{domain}-eval.bin").read_bytes() for domain in DOMAINS},
    )
    return provenance


def subprocess_json(*args):
    completed = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    records = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    return records[-1] if records else None


def cli(*args):
    return subprocess_json("-m", "nexus_expanse.cli", *args)


def train_arm(root, corpus, seed, skill):
    snapshot = root / "unified.pt"
    args = [
        "train", "--corpus-dir", corpus, "--directory", root, "--device", "cpu", "--dtype", "fp32",
        "--seed", seed, "--dim", 64, "--layers", 4, "--context", 64, "--epochs", 2,
        "--steps-per-epoch", 32, "--batch-per-domain", 1, "--eval-blocks", 24, "--epoch-limit", 1,
    ]
    args.extend(("--init-skill", skill) if skill else ("--skill-backbone",))
    cli(*args)
    trained = cli(*args, "--save-snapshot", snapshot)
    verified = cli("verify", "--snapshot", snapshot, "--snapshot-sha256", trained["snapshot_sha256"])
    receipt = json.loads((root / "gate17.json").read_bytes())
    first, second = (epoch["telemetry"] for epoch in receipt["epochs"])
    exact_resume = (first["worker_pid"] != second["worker_pid"]
                    and first["end_model_hash"] == second["start_model_hash"]
                    and second["optimizer_steps_at_start"] == 32)
    cold = receipt["fresh_evaluation"]
    total = sum(domain["target_bytes"] for domain in cold["validation"].values())
    pooled = sum(domain["bpb"] * domain["target_bytes"] for domain in cold["validation"].values()) / total
    return dict(receipt=receipt, export=trained, verify=verified, exact_resume=exact_resume,
                eval_pooled_bpb=pooled)


def seed_run(root, seed):
    corpus = root / "corpus"
    provenance = build_corpus(corpus, seed)
    donor = root / "donor"
    producer = subprocess_json(Path(__file__).resolve(), "--produce", corpus, "--donor", donor, "--seed", seed)
    imports = {}
    skills = {}
    for arm, shuffle in (("warmstart_train", None), ("shuffled_init_train", seed + 50000)):
        archive, receipt = assimilate_zero_exec(donor, mock_pt=True, student_dim=64, student_layers=4,
                                               seed=seed, shuffle_seed=shuffle, use_engram=False)
        path = root / f"{arm}.pt"
        atomic_bytes(path, archive)
        imports[arm] = dict(receipt, skill_sha256=digest(path))
        skills[arm] = path
    arms = {arm: train_arm(root / arm, corpus, seed, skills.get(arm)) for arm in ARMS}
    snapshot = root / "warmstart_train" / "unified.pt"
    prompt = "Phase 13".encode().hex()
    generated = cli("generate", "--snapshot", snapshot, "--prompt-hex", prompt, "--max-bytes", 16)
    stream = cli("stream", "--snapshot", snapshot, "--input-hex", prompt, "--continuation", 16, "--no-adapt")
    return dict(seed=seed, fixture=provenance, producer=producer, imports=imports, arms=arms,
                greedy=dict(generate=generated, stream_hex=stream["continuation_hex"],
                            exact=generated["generated_hex"] == stream["continuation_hex"]))


def verdict(rows):
    wins = sum(row["arms"][ARMS[0]]["eval_pooled_bpb"] <
               min(row["arms"][control]["eval_pooled_bpb"] for control in ARMS[1:]) for row in rows)
    exact = donor_calls = 0
    for row in rows:
        exact += row["greedy"]["exact"] and all(
            arm["exact_resume"] and arm["receipt"]["cold_equal"]
            and not any(arm["verify"]["parity"].values())
            and arm["verify"]["round_trip_probabilities"] == 0.0 and arm["verify"]["exact_state"]
            for arm in row["arms"].values()
        )
        donor_calls += row["greedy"]["generate"]["donor_forward_calls"]
        donor_calls += sum(item["donor_forward_calls"] for item in row["imports"].values())
        donor_calls += sum(arm["receipt"]["donor_forward_calls"] + arm["receipt"]["fresh_evaluation"]["donor_forward_calls"]
                           for arm in row["arms"].values())
    return dict(scientific="PASS" if wins == 5 else "MIXED" if wins >= 4 else "FAIL",
                pooled_both_controls_wins=wins, exact_cold_resume_restore_and_greedy_seeds=exact,
                donor_forward_calls=donor_calls, scored_engineering="PASS" if exact == 5 and donor_calls == 0 else "FAIL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--out", type=Path)
    modes.add_argument("--produce", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--donor", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.produce:
        rows = [torch.tensor(list((args.produce / f"{domain}-train.bin").read_bytes())).reshape(-1, 64)
                for domain in DOMAINS]
        print(json.dumps(produce_cross_width_donor(args.donor, torch.cat(rows), args.seed, dim=128, layers=4)))
        return

    out = args.out.resolve()
    intent = out.with_suffix(".intent.json")
    head = git("rev-parse", "HEAD").decode().strip()
    tree = git("rev-parse", "HEAD^{tree}").decode().strip()
    if (out.is_relative_to(ROOT) or out.exists() or intent.exists() or git("status", "--porcelain")
            or git("rev-parse", "HEAD^").decode().strip() != BASE or not git("ls-tree", head, PREREG)):
        parser.error("run once from the clean Phase 13 preregistration commit with a new external output")
    identity = dict(base=BASE, source_commit=head, source_tree=tree, prereg_sha256=digest(ROOT / PREREG))
    atomic(intent, identity)
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix="nexus-phase13-") as scratch:
        for seed in SEEDS:
            root = Path(scratch) / str(seed)
            root.mkdir()
            rows.append(seed_run(root, seed))
            print(f"Phase 13 seed {seed} complete", flush=True)
    if git("rev-parse", "HEAD").decode().strip() != head or git("status", "--porcelain"):
        parser.error("the committed source changed during scoring")
    result = dict(schema="nexus.phase13-warmstart-train.v1", **identity, corpus_source_sha=SOURCE,
                  platform=platform.platform(), torch=str(torch.__version__), device="cpu", external_downloads=0,
                  wall_seconds=time.perf_counter() - started, parent_peak_rss_bytes=peak_rss(),
                  seeds=rows, summary=verdict(rows))
    atomic(out, result)
    print(json.dumps(result["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
