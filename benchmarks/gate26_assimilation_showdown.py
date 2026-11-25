"""Preregistered Gate 26 full graft and offline byte-KL CPU DEV benchmark."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import weakref
import xml.etree.ElementTree as ET

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks import gate25_zero_run_assimilation as gate25
from benchmarks.dev_artifacts import atomic, digest, save_pt
from benchmarks.gate4_corpus import BASE as CORPUS_BASE, MANIFEST, git, read_split
from benchmarks.gate_custody import peak_rss, prepare_directory, update, writer
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.donor_assimilation import (
    DenseTokenDonorLM, assimilate_zero_run, canonical_byte_bpe, export_offline_byte_kl_cache,
    forbid_donor_forward, learn_byte_bpe, load_offline_byte_kl_cache, sparse_byte_kl,
)
from nexus_expanse.jepa import state_hash


SEEDS = (42, 43, 44, 45, 46)
ARMS = ("scratch_nexus", "shuffled_full_graft", "zero_run_full_graft", "offline_byte_kl", "hybrid_graft_plus_kl")
KL_ARMS = ARMS[3:]
GRAFT_ARMS = (ARMS[1], ARMS[2], ARMS[4])
ALPHA = 0.5
TOP_K = 16
STUDENT = dict(gate25.STUDENT, enable_skip=False, fixed_eval_geometry=True)
BASE = "eca9408aca0e2fd4f0fced57eb5beeb8106d3730"
BASE_RECEIPT = ROOT / "evidence/gate25-zero-run-assimilation-dev-001.json"
BASE_RECEIPT_SHA = "84c02791529f88f229d09448b7c972920c095147766654d67ab97d1deec43572"
CACHE_NAME = "offline-byte-kl.bin"


def source_identity():
    if git("status", "--porcelain"):
        raise ValueError("Gate 26 requires a clean source tree")
    sha, tree, parent = git("rev-parse", "HEAD", "HEAD^{tree}", "HEAD^").decode().split()
    changed = git("show", "--format=", "--name-only", sha).decode().splitlines()
    prereg = "docs/GATE26_PREREG.md"
    if parent != BASE or prereg not in changed:
        raise ValueError("run Gate 26 from the exact preregistration commit")
    if digest(MANIFEST) != gate25.CORPUS_MANIFEST_SHA or digest(BASE_RECEIPT) != BASE_RECEIPT_SHA:
        raise ValueError("Gate 26 corpus or baseline receipt has changed")
    return dict(source_sha=sha, source_tree=tree, base_sha=parent, prereg_sha256=digest(ROOT / prereg),
                corpus_base=CORPUS_BASE, manifest_sha256=gate25.CORPUS_MANIFEST_SHA,
                gate25_evidence_sha256=BASE_RECEIPT_SHA, python=platform.python_version(),
                torch=str(torch.__version__), numpy=version("numpy"), pytest=version("pytest"),
                platform=platform.platform(), machine=platform.machine(),
                device="cpu", threads=torch.get_num_threads())


def prepare_teacher(seed, directory, raw_train, encoded, vocab, merges, prior, historical):
    donor_metrics = gate25.train_donor(seed, encoded, vocab, directory)
    if historical is not None:
        donor_metrics["gate25_checkpoint_sha256"] = historical
        donor_metrics["gate25_checkpoint_equal"] = donor_metrics["checkpoint_sha256"] == historical
    exported = torch.load(directory / "donor.pt", weights_only=True, map_location="cpu")
    if digest(directory / "donor.pt") != donor_metrics["checkpoint_sha256"]:
        raise ValueError("donor checkpoint hash differs")
    donor = DenseTokenDonorLM().eval().requires_grad_(False)
    donor.load_state_dict(exported)
    start, cpu = time.monotonic(), time.process_time()
    cache = export_offline_byte_kl_cache(donor, directory / CACHE_NAME, raw_train, vocab, merges, prior, top_k=TOP_K)
    reference = weakref.ref(donor)
    del donor
    if reference() is not None or cache["donor_forward_calls_offline"] != 1:
        raise RuntimeError("offline donor must execute once and be deleted")
    cache.update(donor_deleted_before_student=True, wall_seconds=time.monotonic() - start,
                 cpu_seconds=time.process_time() - cpu)
    return donor_metrics, cache


@torch.inference_mode()
def prefix_error(model, raw):
    model.eval()
    original = torch.tensor(list(raw[:32]))[None]
    changed = original.clone()
    changed[:, 13:] = 255 - changed[:, 13:]
    expected = model.forward_neural(original)[0][:, :13]
    short = model.forward_neural(original[:, :13])[0]
    altered = model.forward_neural(changed)[0][:, :13]
    return max(float((expected - short).abs().max()), float((expected - altered).abs().max()))


def train_arm(seed, arm, train, dev, probe, directory, *, static=None, cache=None):
    start, cpu = time.monotonic(), time.process_time()
    with forbid_donor_forward() as guard:
        torch.manual_seed(seed)
        student = NexusExpanseByteCore(**STUDENT)
        initial = state_hash(student)
        assimilation = None
        if static is not None:
            mode = "shuffled_full_graft" if arm == "shuffled_full_graft" else "zero_run_full_graft"
            assimilation = asdict(assimilate_zero_run(student, *static, mode=mode,
                                                      bridge_steps=gate25.BRIDGE_STEPS, shuffle_seed=seed + 2600))
        warm = state_hash(student)
        initial_prefix_error = prefix_error(student, probe)
        optimizer = gate25.optimizer(student)
        steps, curve = 0, {}
        for epoch in range(1, gate25.ADAPT_EPOCHS + 1):
            student.train()
            for index in gate25.order(seed, epoch - 1, len(train)):
                x, y = train[index]
                optimizer.zero_grad(set_to_none=True)
                logits = student.forward_neural(x)[0]
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                if cache is not None:
                    ids, probs = cache[index]
                    loss = loss + ALPHA * sparse_byte_kl(logits, ids, probs)
                update(student, optimizer, loss)
                steps += 1
            if epoch in (1, 2, gate25.ADAPT_EPOCHS):
                curve[str(epoch * 100 // gate25.ADAPT_EPOCHS)] = dict(step=steps, **gate25.score(student, dev))
        final_prefix_error = prefix_error(student, probe)
        calls = guard.donor_forward_calls

    checkpoint = directory / f"{arm}.pt"
    save_pt(checkpoint, student.state_dict())
    return dict(initial_sha256=initial, warm_sha256=warm, state_sha256=state_hash(student),
                checkpoint_sha256=digest(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
                curve=curve, adaptation_steps=steps, adaptation_byte_targets=steps * train[0][1].numel(),
                assimilation=assimilation, prefix_max_abs_diff_before=initial_prefix_error,
                prefix_max_abs_diff=final_prefix_error, donor_forward_calls_during_student=calls,
                wall_seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu)


def cold_restore(directory, dev):
    restored = {}
    for arm in ARMS:
        model = NexusExpanseByteCore(**STUDENT)
        model.load_state_dict(torch.load(directory / f"{arm}.pt", weights_only=True, map_location="cpu"))
        restored[arm] = dict(state_sha256=state_hash(model), **gate25.score(model, dev))
    return restored


def verify_seed(record, seed, identity, directory):
    if record["identity"] != identity or record["seed"] != seed or set(record["arms"]) != set(ARMS):
        raise ValueError("seed receipt identity differs")
    actual = {"donor.pt": record["donor"]["checkpoint_sha256"], "vocab.json": record["donor"]["vocab_sha256"],
              CACHE_NAME: record["cache"]["sha256"]}
    actual.update({f"{arm}.pt": data["checkpoint_sha256"] for arm, data in record["arms"].items()})
    if any(digest(directory / name) != sha for name, sha in actual.items()):
        raise ValueError("seed artifact digest differs")
    return record


def run_seed(seed, directory, identity, raw_train, encoded, vocab, merges, prior, train, dev, historical):
    start, cpu = time.monotonic(), time.process_time()
    seed_dir = directory / f"seed-{seed}"
    receipt = seed_dir / "receipt.json"
    with writer(directory, f"seed-{seed}"):
        if receipt.exists():
            return verify_seed(json.loads(receipt.read_text()), seed, identity, seed_dir)
        if seed_dir.exists():
            raise RuntimeError(f"incomplete seed custody, automatic replay refused: {seed_dir}")
        seed_dir.mkdir()
        atomic(seed_dir / "intent.json", dict(seed=seed, identity=identity))
        donor, cache_metrics = prepare_teacher(seed, seed_dir, raw_train, encoded, vocab, merges, prior, historical)
        with forbid_donor_forward() as guard:
            state = torch.load(seed_dir / "donor.pt", weights_only=True, map_location="cpu")
            raw_vocab = [bytes.fromhex(item) for item in json.loads((seed_dir / "vocab.json").read_text())]
            cached = load_offline_byte_kl_cache(seed_dir / CACHE_NAME, raw_train, cache_metrics["sha256"], top_k=TOP_K)
            # Row zero is cached for completeness. CE trains the other 255 transitions.
            targets = [tuple(t[1:].reshape(-1, gate25.CONTEXT, TOP_K) for t in span) for span in cached]
            results = {}
            for arm in ARMS:
                result = train_arm(seed, arm, train, dev, raw_train[0], seed_dir,
                                   static=(state, raw_vocab) if arm in GRAFT_ARMS else None,
                                   cache=targets if arm in KL_ARMS else None)
                results[arm] = result
                print(f"seed {seed} {arm} BPB {result['curve']['100']['bpb']:.6f}", flush=True)
            calls = guard.donor_forward_calls

        cold_started = time.monotonic()
        cold = json.loads(subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve()), "--cold-restore", str(seed_dir)],
            cwd=ROOT, text=True, timeout=60,
        ))
        for arm, data in results.items():
            data["cold_restore"] = cold[arm]
            data["cold_equal"] = (cold[arm]["state_sha256"] == data["state_sha256"]
                                  and cold[arm]["prediction_sha256"] == data["curve"]["100"]["prediction_sha256"])
        record = dict(seed=seed, identity=identity, donor=donor, cache=cache_metrics, arms=results,
                      donor_forward_calls_during_student=calls, cold_wall_seconds=time.monotonic() - cold_started,
                      wall_seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu,
                      parent_peak_rss_bytes=peak_rss())
        atomic(receipt, record)
        return record


def track(first, second):
    return "PASS" if first and second else "MIXED" if first or second else "FAIL"


def verdicts(runs, checks):
    scores = {arm: [r["arms"][arm]["curve"]["100"]["bpb"] for r in runs] for arm in ARMS}
    scratch, shuffled, full = (scores[arm] for arm in ARMS[:3])
    full_deltas = [a - b for a, b in zip(full, scratch, strict=True)]
    shuffled_deltas = [a - b for a, b in zip(full, shuffled, strict=True)]
    full_margin = sum(delta <= -0.05 for delta in full_deltas)
    full_shuffled = sum(delta < 0 for delta in shuffled_deltas)
    kl_arms = {}
    for arm in KL_ARMS:
        scratch_delta = [a - b for a, b in zip(scores[arm], scratch, strict=True)]
        full_delta = [a - b for a, b in zip(scores[arm], full, strict=True)]
        kl_margin = sum(delta <= -0.10 for delta in scratch_delta)
        kl_full = sum(delta < 0 for delta in full_delta)
        kl_arms[arm] = dict(verdict=track(kl_margin == 5, kl_full >= 4),
                            scratch_margin_wins=kl_margin, full_graft_wins=kl_full,
                            scratch_deltas_bpb=scratch_delta, full_graft_deltas_bpb=full_delta)
    kl_statuses = {value["verdict"] for value in kl_arms.values()}
    kl_verdict = next(status for status in ("PASS", "MIXED", "FAIL") if status in kl_statuses)

    arms = [data for run in runs for data in run["arms"].values()]
    prefixes = sum(arm["prefix_max_abs_diff"] == arm["prefix_max_abs_diff_before"] == 0.0 for arm in arms)
    cold = sum(arm["cold_equal"] for arm in arms)
    zero_calls = all(run["cache"]["donor_forward_calls_offline"] == 1
                     and run["cache"]["donor_deleted_before_student"]
                     and run["cache"]["max_normalization_error"] <= 1e-5
                     and run["donor_forward_calls_during_student"] == 0 for run in runs)
    zero_calls = zero_calls and all(arm["donor_forward_calls_during_student"] == 0
                                    and (arm["assimilation"] is None or arm["assimilation"]["donor_forward_calls"] == 0)
                                    for arm in arms)
    copied = [error for arm in arms if arm["assimilation"] is not None
              for error in arm["assimilation"]["copied_max_abs_error"].values()]
    pytest, prefix = checks["pytest"], checks["prefix"]
    engineering = (
        pytest["exit_code"] == pytest["errors"] == pytest["failures"] == pytest["skipped"] == 0
        and pytest["gate26_tests"] > 0 and pytest["passed"] == 176 + pytest["gate26_tests"]
        and prefix["exit_code"] == 0 and prefix["seeds"] == 100 and not prefix["fails"]
        and len(copied) == 240 and all(error == 0.0 for error in copied)
        and prefixes == cold == 25 and zero_calls
    )
    return dict(engineering="PASS" if engineering else "FAIL", deployment="NOT_RUN",
                prefix_exact_runs=prefixes, cold_restore_exact_runs=cold, zero_donor_forward_verified=zero_calls,
                dense_tensors_copied_exactly=len(copied) if all(error == 0.0 for error in copied) else 0,
                scientific=dict(zero_run=dict(verdict=track(full_margin >= 4, full_shuffled == 5),
                                              scratch_margin_wins=full_margin, shuffled_wins=full_shuffled,
                                              scratch_deltas_bpb=full_deltas, shuffled_deltas_bpb=shuffled_deltas),
                                offline_byte_kl=dict(verdict=kl_verdict, arms=kl_arms)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--seed-limit", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--cold-restore", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    gate25.setup()
    started = time.monotonic()
    identity = source_identity()
    corpus = read_split()
    raw_train = corpus["train"]
    if (len(raw_train), len(corpus["dev"])) != (31, 8):
        raise ValueError("unexpected Gate 26 split")
    dev = gate25.byte_batches(corpus["dev"])
    if args.cold_restore:
        print(json.dumps(cold_restore(args.cold_restore, dev), allow_nan=False))
        return
    if args.directory is None:
        parser.error("--directory is required")
    directory = prepare_directory(args.directory, ROOT)
    train = gate25.byte_batches(raw_train)
    vocab, encoded, merges = learn_byte_bpe(raw_train)
    ranks = {pair: token for token, pair in enumerate(merges, 256)}
    if [list(canonical_byte_bpe(raw, ranks)) for raw in raw_train] != encoded:
        raise ValueError("canonical BPE does not reproduce donor training IDs")
    counts = 1 + torch.bincount(torch.tensor([span[0] for span in encoded]), minlength=len(vocab)).double()
    prior = counts / counts.sum()
    history = {r["seed"]: r["donor"]["checkpoint_sha256"] for r in json.loads(BASE_RECEIPT.read_text())["runs"]}

    runs, new = [], 0
    for seed in SEEDS:
        if not (directory / f"seed-{seed}" / "receipt.json").exists():
            if new == args.seed_limit:
                print(f"Completed {len(runs)} of {len(SEEDS)} seeds. Resume with the same directory.")
                return
            new += 1
        runs.append(run_seed(seed, directory, identity, raw_train, encoded, vocab, merges, prior,
                             train, dev, history.get(seed)))

    checks = gate25.verify_engineering(directory, identity)
    xml = directory / "pytest.xml"
    if digest(xml) != checks["pytest"]["junit_sha256"] or source_identity() != identity:
        raise ValueError("Gate 26 source or JUnit receipt changed")
    checks["pytest"]["gate26_tests"] = sum(
        case.get("classname", "").endswith("test_gate26_assimilation") for case in ET.parse(xml).iter("testcase")
    )
    result = dict(schema="nexus-gate26-assimilation-showdown-dev-v1", qualification="REUSED_MANIFEST_DEV",
                  identity=identity, checks=checks, runs=runs, verdicts=verdicts(runs, checks),
                  config=dict(seeds=SEEDS, arms=ARMS, student=STUDENT, donor_epochs=gate25.DONOR_EPOCHS,
                              bridge_steps=gate25.BRIDGE_STEPS, adaptation_steps=124, context_bytes=gate25.CONTEXT,
                              train_spans=len(train), dev_spans=len(dev), alpha=ALPHA, top_k=TOP_K,
                              marginalization="all final boundaries with one canonical BPE history each"),
                  wall_seconds=time.monotonic() - started, peak_rss_bytes=peak_rss())
    with writer(directory, "result"):
        atomic(directory / "result.json", result)
    print(json.dumps(result["verdicts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
