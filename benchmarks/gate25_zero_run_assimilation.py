"""Source-bound, resumable CPU DEV benchmark for static donor assimilation."""

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import math
import os
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
from benchmarks.dev_artifacts import atomic, digest, save_pt
from benchmarks.gate4_corpus import BASE as CORPUS_BASE, MANIFEST, git, read_split
from benchmarks.gate_custody import peak_rss, prepare_directory, update, writer
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.donor_assimilation import (
    MODES, DenseTokenDonorLM, assimilate_zero_run, fit_byte_bpe, forbid_donor_forward,
)
from nexus_expanse.jepa import state_hash


SEEDS = (42, 43, 44)
DONOR_EPOCHS = 8
ADAPT_EPOCHS = 4
BRIDGE_STEPS = 96
CONTEXT = 51
STUDENT = dict(dim=128, n_layers=2, n_heads=4, ffn_mult=4, enable_skip=True, dropout=0.0, max_seq_len=64)
CORPUS_MANIFEST_SHA = "f5d449a93e1e006e5bda5ca6b6e8b16b50fd6a6098c40f34eff0a82224a3f87d"
SCHEMA = "nexus-gate25-zero-run-assimilation-dev-v1"


def setup():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def source_identity():
    if git("status", "--porcelain"):
        raise ValueError("Gate 25 requires a clean source tree")
    source_sha = git("rev-parse", "HEAD").decode().strip()
    if git("show", "--format=", "--name-only", source_sha).decode().split().count("docs/GATE25_PREREG.md") != 1:
        raise ValueError("run Gate 25 from the exact preregistration commit")
    manifest_sha = digest(MANIFEST)
    if manifest_sha != CORPUS_MANIFEST_SHA:
        raise ValueError("unexpected Gate 25 corpus manifest")
    return dict(source_sha=source_sha, source_tree=git("rev-parse", "HEAD^{tree}").decode().strip(),
                prereg_sha256=digest(ROOT / "docs/GATE25_PREREG.md"),
                corpus_base=CORPUS_BASE, manifest_sha256=manifest_sha,
                python=platform.python_version(), torch=str(torch.__version__),
                platform=platform.platform(), machine=platform.machine(), device="cpu",
                numpy=version("numpy"), pytest=version("pytest"), threads=torch.get_num_threads())


def order(seed, epoch, length):
    return torch.randperm(length, generator=torch.Generator().manual_seed(seed + epoch)).tolist()


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)


def train_donor(seed, token_spans, vocab, directory):
    torch.manual_seed(seed)
    donor = DenseTokenDonorLM()
    optim = optimizer(donor)
    token_rows = [torch.tensor(span)[None] for span in token_spans]
    start, cpu_start = time.monotonic(), time.process_time()
    epoch_losses = []
    for epoch in range(DONOR_EPOCHS):
        weighted_loss = count = 0
        for index in order(seed, epoch, len(token_rows)):
            row = token_rows[index]
            x, y = row[:, :-1], row[:, 1:]
            optim.zero_grad(set_to_none=True)
            loss = update(donor, optim, F.cross_entropy(donor(x).flatten(0, 1), y.flatten()))
            weighted_loss += loss * y.numel()
            count += y.numel()
        epoch_losses.append(weighted_loss / count)
    donor.eval().requires_grad_(False)
    state, exported_vocab = donor.export_static(vocab)
    train_calls = donor.forward_calls
    expected = DONOR_EPOCHS * len(token_rows)
    if train_calls != expected:
        raise RuntimeError("unexpected donor training call count")
    donor_reference = weakref.ref(donor)
    del optim, donor
    if donor_reference() is not None:
        raise RuntimeError("donor model remains reachable")
    state_path, vocab_path = directory / "donor.pt", directory / "vocab.json"
    save_pt(state_path, state)
    atomic(vocab_path, [spelling.hex() for spelling in exported_vocab])
    metrics = dict(training_forward_calls=train_calls, model_deleted=True,
                   training_targets_per_epoch=count, epoch_nll_nats=epoch_losses,
                   wall_seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu_start,
                   checkpoint_bytes=state_path.stat().st_size,
                   tensor_bytes=sum(t.numel() * t.element_size() for t in state.values()),
                   checkpoint_sha256=digest(state_path), vocab_sha256=digest(vocab_path))
    return metrics


def byte_batches(spans):
    batches = []
    for span in spans:
        row = torch.tensor(list(span))
        batches.append((row[:-1].reshape(-1, CONTEXT), row[1:].reshape(-1, CONTEXT)))
    return batches


@torch.inference_mode()
def score(model, dev):
    model.eval()
    prediction_digest = hashlib.sha256()
    nll = 0.0
    targets = 0
    for x, y in dev:
        logits, _, _, _ = model.forward_neural(x)
        prediction_digest.update(logits.contiguous().numpy().tobytes())
        nll += float(F.cross_entropy(logits.flatten(0, 1).double(), y.flatten(), reduction="sum"))
        targets += y.numel()
    return dict(bpb=nll / (math.log(2) * targets), targets=targets,
                prediction_sha256=prediction_digest.hexdigest())


@torch.inference_mode()
def prefix_error(model, raw):
    model.eval()
    row = torch.tensor(list(raw[:32]))[None]
    full = model.forward_neural(row)[0]
    prefix = model.forward_neural(row[:, :13])[0]
    return float((full[:, :13] - prefix).abs().max())


def train_arm(seed, mode, state, vocab, train, dev, probe, directory):
    start, cpu_start = time.monotonic(), time.process_time()
    torch.manual_seed(seed)
    student = NexusExpanseByteCore(**STUDENT)
    initial_sha = state_hash(student)
    graft = assimilate_zero_run(student, state, vocab, mode, bridge_steps=BRIDGE_STEPS, shuffle_seed=seed + 2500)
    max_prefix_error = prefix_error(student, probe)
    assimilation_seconds = time.monotonic() - start

    optim = optimizer(student)
    steps = 0
    curve = {}
    for epoch in range(1, ADAPT_EPOCHS + 1):
        student.train()
        for index in order(seed, epoch - 1, len(train)):
            x, y = train[index]
            optim.zero_grad(set_to_none=True)
            logits, _, _, _ = student.forward_neural(x)
            update(student, optim, F.cross_entropy(logits.flatten(0, 1), y.flatten()))
            steps += 1
        if epoch in (1, 2, ADAPT_EPOCHS):
            curve[str(epoch * 100 // ADAPT_EPOCHS)] = dict(step=steps, **score(student, dev))

    checkpoint = directory / f"{mode}.pt"
    save_pt(checkpoint, student.state_dict())
    return dict(initial_sha256=initial_sha, state_sha256=state_hash(student),
                checkpoint_sha256=digest(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
                assimilation=asdict(graft), prefix_max_abs_diff=max_prefix_error, curve=curve,
                adaptation_steps=steps, adaptation_byte_targets=steps * train[0][1].numel(),
                assimilation_wall_seconds=assimilation_seconds, wall_seconds=time.monotonic() - start,
                cpu_seconds=time.process_time() - cpu_start)


def cold_restore(directory, dev):
    out = {}
    for mode in MODES:
        model = NexusExpanseByteCore(**STUDENT)
        model.load_state_dict(torch.load(directory / f"{mode}.pt", weights_only=True, map_location="cpu"))
        out[mode] = dict(state_sha256=state_hash(model), **score(model, dev))
    return out


def validate_receipt(path, directory, identity, seed):
    record = json.loads(path.read_text())
    if record["identity"] != identity or record["seed"] != seed or set(record["arms"]) != set(MODES):
        raise ValueError(f"seed identity mismatch: {path}")
    expected = {"donor.pt": record["donor"]["checkpoint_sha256"], "vocab.json": record["donor"]["vocab_sha256"]}
    expected.update({f"{mode}.pt": item["checkpoint_sha256"] for mode, item in record["arms"].items()})
    if any(digest(directory / name) != sha for name, sha in expected.items()):
        raise ValueError(f"seed checkpoint custody mismatch: {path}")
    return record


def run_seed(seed, directory, identity, train, dev, encoded, vocab, probe):
    start, cpu_start = time.monotonic(), time.process_time()
    seed_dir = directory / f"seed-{seed}"
    receipt = seed_dir / "receipt.json"
    with writer(directory, f"seed-{seed}"):
        if receipt.exists():
            return validate_receipt(receipt, seed_dir, identity, seed)
        if seed_dir.exists():
            raise RuntimeError(f"incomplete seed custody, automatic replay refused: {seed_dir}")
        seed_dir.mkdir()
        atomic(seed_dir / "intent.json", dict(seed=seed, identity=identity))
        donor = train_donor(seed, encoded, vocab, seed_dir)
        with forbid_donor_forward() as guard:
            state = torch.load(seed_dir / "donor.pt", map_location="cpu", weights_only=True)
            raw_vocab = [bytes.fromhex(item) for item in json.loads((seed_dir / "vocab.json").read_text())]
            cores = {}
            for mode in MODES:
                cores[mode] = train_arm(seed, mode, state, raw_vocab, train, dev, probe, seed_dir)
                print(f"seed {seed} {mode} BPB {cores[mode]['curve']['100']['bpb']:.6f}", flush=True)
            calls = guard.donor_forward_calls

        verify_start = time.monotonic()
        cold = json.loads(subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve()), "--cold-restore", str(seed_dir)],
            cwd=ROOT, text=True, timeout=60,
        ))
        for mode, arm in cores.items():
            arm["cold_restore"] = cold[mode]
            arm["cold_equal"] = (cold[mode]["state_sha256"] == arm["state_sha256"]
                                 and cold[mode]["prediction_sha256"] == arm["curve"]["100"]["prediction_sha256"])
        record = dict(seed=seed, identity=identity, donor=donor, arms=cores,
                      donor_forward_calls=calls, cold_wall_seconds=time.monotonic() - verify_start,
                      wall_seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu_start,
                      parent_peak_rss_bytes=peak_rss())
        atomic(receipt, record)
        return record


def verify_engineering(directory, identity):
    receipt_path = directory / "qualification.json"
    with writer(directory, "qualification"):
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["identity"] != identity:
                raise ValueError("test receipt source or runtime differs")
            return receipt
        report = directory / "pytest.xml"
        if report.exists():
            raise RuntimeError("incomplete test custody, automatic replay refused")
        env = os.environ.copy()
        env["PYTHONPATH"] = f".:{env.get('PYTHONPATH', '')}"
        env.pop("PYTEST_ADDOPTS", None)
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "1"
        start = time.monotonic()
        tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "--junitxml", str(report)], cwd=ROOT, env=env)
        test_seconds = time.monotonic() - start
        suites = ET.parse(report).getroot().findall("testsuite")
        counts = {key: sum(int(suite.attrib[key]) for suite in suites)
                  for key in ("tests", "errors", "failures", "skipped")}
        gate25_count = sum(1 for suite in suites for case in suite.findall("testcase")
                           if case.get("classname", "").endswith("test_gate25_donor_assimilation"))
        start = time.monotonic()
        sweep = subprocess.run([sys.executable, "benchmarks/prefix_causality_dev.py"], cwd=ROOT, env=env,
                               capture_output=True, check=False, text=True)
        prefix = json.loads(sweep.stdout)
        print(sweep.stdout, end="", flush=True)
        if prefix["source_sha"] != identity["source_sha"]:
            raise ValueError("prefix check ran on a different revision")
        receipt = dict(identity=identity, pytest=dict(exit_code=tests.returncode, **counts,
                       passed=counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"],
                       gate25_tests=gate25_count, wall_seconds=test_seconds, junit_sha256=digest(report)),
                       prefix=dict(exit_code=sweep.returncode, **prefix, wall_seconds=time.monotonic() - start))
        atomic(receipt_path, receipt)
        return receipt


def verdicts(runs, checks):
    differences = {}
    for run in runs:
        scores = {mode: arm["curve"]["100"]["bpb"] for mode, arm in run["arms"].items()}
        differences[str(run["seed"])] = {mode: scores["graft_and_bridge"] - scores[mode] for mode in MODES[1:]}
    scratch_wins = sum(row["scratch_nexus"] < 0 for row in differences.values())
    margin_wins = sum(row["scratch_nexus"] <= -0.05 for row in differences.values())
    shuffled_wins = sum(row["shuffled_donor_control"] < 0 for row in differences.values())
    graft_matches = sum(row["graft_only"] <= 0 for row in differences.values())
    if margin_wins == shuffled_wins == 3 and graft_matches >= 2:
        science = "PASS"
    elif scratch_wins >= 2 or (shuffled_wins == 3 and margin_wins < 3):
        science = "MIXED"
    else:
        science = "FAIL"

    test, prefix = checks["pytest"], checks["prefix"]
    cold_seeds = sum(all(arm["cold_equal"] for arm in run["arms"].values()) for run in runs)
    zero_calls = all(run["donor_forward_calls"] == 0 and run["donor"]["model_deleted"] and
                     all(arm["assimilation"]["donor_forward_calls"] == 0 for arm in run["arms"].values())
                     for run in runs)
    engineering = (
        test["passed"] > 0 and test["gate25_tests"] > 0 and
        test["exit_code"] == test["errors"] == test["failures"] == test["skipped"] == 0 and
        prefix["exit_code"] == 0 and prefix["seeds"] == 100 and not prefix["fails"] and
        zero_calls and cold_seeds == 3 and
        all(arm["prefix_max_abs_diff"] == 0.0 for run in runs for arm in run["arms"].values())
    )
    return dict(engineering="PASS" if engineering else "FAIL", scientific=science, deployment="NOT_RUN",
                cold_restore_seeds=cold_seeds, zero_donor_forward_verified=zero_calls,
                differences_bpb=differences, scratch_wins=scratch_wins, scratch_margin_wins=margin_wins,
                shuffled_wins=shuffled_wins, graft_only_wins_or_ties=graft_matches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, help="artifact directory outside the repository")
    parser.add_argument("--seed-limit", type=int, choices=range(1, 4), default=3,
                        help="maximum new seeds in this invocation, resuming from completed seeds")
    parser.add_argument("--cold-restore", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    setup()
    started = time.monotonic()
    identity = source_identity()
    corpus = read_split()
    if (len(corpus["train"]), len(corpus["dev"])) != (31, 8):
        raise ValueError("unexpected Gate 25 split size")
    dev = byte_batches(corpus["dev"])
    if args.cold_restore:
        print(json.dumps(cold_restore(args.cold_restore, dev), allow_nan=False))
        return
    if args.directory is None:
        parser.error("--directory is required")
    directory = prepare_directory(args.directory, ROOT)
    train = byte_batches(corpus["train"])
    vocab, encoded = fit_byte_bpe(corpus["train"])
    if any(b"".join(vocab[index] for index in ids) != raw for raw, ids in zip(corpus["train"], encoded)):
        raise ValueError("donor BPE changed source bytes")
    runs, new = [], 0
    for seed in SEEDS:
        exists = (directory / f"seed-{seed}" / "receipt.json").exists()
        if not exists:
            if new == args.seed_limit:
                print(f"Completed {len(runs)} of {len(SEEDS)} seeds. Resume with the same directory.", flush=True)
                return
            new += 1
        record = run_seed(seed, directory, identity, train, dev, encoded, vocab, corpus["train"][0])
        runs.append(record)

    checks = verify_engineering(directory, identity)
    if source_identity() != identity:
        raise ValueError("source changed during Gate 25")
    result = dict(schema=SCHEMA, qualification="REUSED_MANIFEST_DEV", identity=identity,
                  config=dict(seeds=SEEDS, donor=dict(epochs=DONOR_EPOCHS, vocab_size=len(vocab),
                              dim=128, n_layers=2, n_heads=4, ffn_mult=4, max_seq_len=256,
                              max_spelling_bytes=max(map(len, vocab))), bridge_steps=BRIDGE_STEPS,
                              adaptation_steps=ADAPT_EPOCHS * len(train), student=STUDENT,
                              context_bytes=CONTEXT, train_spans=len(train), dev_spans=len(dev)),
                  checks=checks, runs=runs, verdicts=verdicts(runs, checks),
                  final_invocation_wall_seconds=time.monotonic() - started,
                  final_invocation_peak_rss_bytes=peak_rss())
    with writer(directory, "result"):
        atomic(directory / "result.json", result)
    print(json.dumps(result["verdicts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
