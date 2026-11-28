"""Frozen Phase 19 CPU comparison of neural forecasts and on-disk byte counts."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.dev_artifacts import atomic, atomic_bytes, digest
from benchmarks.gate_custody import peak_rss
from benchmarks.phase13_warmstart_train import DOMAINS, cli
from benchmarks.phase17_domain_focus import match_corpus
from benchmarks.phase3_multi_skill_routing import SOURCE, git
from nexus_expanse.cli import engram_forecast
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.ssd_engram import SSDEngramStore
from nexus_expanse.unified import NexusExpanseUnified
from scripts.prepare_gpu_corpus import assert_disjoint

BASE = "fa8a42a35ca36d010842ed4aed49bae594cdb89a"
PREREG = "docs/PHASE19_SSD_ENGRAM_MEMORY_PREREG.md"
SEEDS = tuple(range(3301, 3306))
CONTEXT, STEPS, ALPHA = 32, 64, 0.35
DISTRACTORS = {"prose": "code", "code": "prose"}
BINARY_PREFIX = b"\x00\xff\x80\x01\x19\xee\x93\x02\xa0\xfe\x16\x11\x00\xd6\xbb\x17"
LOOKUP_WORKER = """
import hashlib, random, sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'nexus_expanse'))
from ssd_engram import SSDEngramStore
path, expected = sys.argv[1:]
raw = random.Random(3319).randbytes(65536)
timer = time.perf_counter()
with SSDEngramStore(path, expected_sha256=expected) as store:
    opened = time.perf_counter()
    digest = hashlib.sha256()
    hits = 0
    for j in range(4096):
        n = 12 + (j * 1871) % (len(raw) - 12)
        prefix = raw[n-12:n]
        if j % 2:
            prefix = prefix[:-4] + bytes(x ^ 0xff for x in prefix[-4:])
        gain, match = store.lookup(prefix)
        hits += bool(gain)
        digest.update(repr((gain, match)).encode())
    elapsed = time.perf_counter() - opened
# Linux process high-water mark after exec, independent of the PyTorch parent.
peak_kib = next(int(line.split()[1]) for line in Path('/proc/self/status').read_text().splitlines()
                if line.startswith('VmHWM:'))
print(json.dumps(dict(digest=digest.hexdigest(), hits=hits, queries=4096, open_ms=(opened-timer)*1000,
                      ms_per_byte=elapsed*1000/4096, peak_rss_bytes=1024*peak_kib)))
"""


def train(root, corpus, seed, focus):
    path = root / f"{focus}.pt"
    run = root / focus
    report = cli("train", "--corpus-dir", corpus, "--focus-domain", focus, "--directory", run, "--device", "cpu",
                 "--dtype", "fp32", "--dim", 32, "--layers", 2, "--context", CONTEXT, "--epochs", 1,
                 "--steps-per-epoch", STEPS, "--batch-per-domain", 1, "--skill-backbone", "--seed", seed,
                 "--eval-blocks", 4, "--save-snapshot", path)
    receipt = json.loads((run / "gate17.json").read_bytes())
    return NexusExpanseUnified.restore(path.read_bytes(), fast_decode=True), dict(
        archive=report, target_bytes=receipt["epochs"][0]["telemetry"]["target_bytes"], updates=receipt["steps"],
        cold_equal=receipt["cold_equal"], donor_forward_calls=report["donor_forward_calls"] +
        receipt["donor_forward_calls"] + receipt["fresh_evaluation"]["donor_forward_calls"])


@torch.inference_mode()
def probabilities(model, raw):
    ids = torch.tensor(list(raw[:-1]), dtype=torch.long)
    prefix = [model.forward_neural(ids[:n][None])[0][0, -1:].softmax(-1) for n in range(1, CONTEXT)]
    full = [model.forward_neural(batch)[0][:, -1].softmax(-1) for batch in ids.unfold(0, CONTEXT, 1).split(32)]
    return torch.cat(prefix + full)


def augment(base, raw, store):
    mixed, hits = [], 0
    for pos, probs in enumerate(base, 1):
        forecast, gain = engram_forecast(probs, raw[max(0, pos - store.max_order):pos], store, ALPHA)
        mixed.append(forecast)
        hits += bool(gain)
    return torch.stack(mixed), hits


def nll(probs, raw):
    observed = torch.tensor(list(raw[1:]), dtype=torch.long)
    return -float(probs[torch.arange(len(observed)), observed].double().log2().sum())


def score(base, raw, path, report):
    with SSDEngramStore(path, expected_sha256=report["index_sha256"]) as store:
        first, hits = augment(base, raw, store)
    with SSDEngramStore(path, expected_sha256=report["index_sha256"]) as store:
        second, reopened_hits = augment(base, raw, store)
    count = len(raw) - 1
    neural, combined = nll(base, raw), nll(first, raw)
    return dict(targets=count, neural_bits=neural, engram_bits=combined, neural_bpb=neural / count,
                engram_bpb=combined / count, delta_bpb=(combined - neural) / count, hits=hits,
                hit_rate=hits / count, cold_max_diff=float((first - second).abs().max()), same_hits=hits == reopened_hits)


def stream_forecasts(model, raw, store):
    model.reset_context()
    model.reader.minimum_probability = 2.0
    predictions, gains = [], []
    for i, observed in enumerate(raw, 1):
        base = model.step(observed, adapt=False).mixed
        forecast, gain = engram_forecast(base, raw[max(0, i - store.max_order):i], store, ALPHA)
        predictions.append(forecast)
        gains.append(gain)
    return torch.stack(predictions), tuple(gains)


def prefix_checks(root, model):
    source, index = root / "binary.bin", root / "binary.sqlite"
    atomic_bytes(source, BINARY_PREFIX * 12)
    report = SSDEngramStore.build(source, index)
    failures, active, maximum = [], 0, 0.0
    with SSDEngramStore(index, expected_sha256=report["index_sha256"]) as store:
        for seed in range(100):
            rng = random.Random(seed)
            size = rng.randrange(8, len(BINARY_PREFIX) + 1)
            prompt = BINARY_PREFIX[:size]
            left, right = (stream_forecasts(model, prompt + rng.randbytes(5), store) for _ in range(2))
            shorter, short_gains = stream_forecasts(model, prompt, store)
            a, ag = left
            b, bg = right
            difference = max(float((a[:size] - b[:size]).abs().max()), float((a[:size] - shorter).abs().max()))
            maximum = max(maximum, difference)
            active += any(ag[:size])
            if difference or ag[:size] != bg[:size] or ag[:size] != short_gains:
                failures.append(seed)
    return dict(seeds=100, active=active, maximum=maximum, failures=failures)


def seed_run(root, seed):
    corpus, provenance = match_corpus(root, seed)
    train_paths = {d: corpus / f"{d}-train.bin" for d in DOMAINS}
    train_bytes = {d: p.read_bytes() for d, p in train_paths.items()}
    held = {d: (corpus / f"{d}-eval.bin").read_bytes() for d in DOMAINS}
    assert_disjoint(train_bytes.values(), held)
    fit = root / "index-fit"
    fit.mkdir()
    for d, raw in train_bytes.items():
        atomic_bytes(fit / f"{d}.bin", raw)
    indexes = {"all": root / "all.sqlite", **{d: root / f"{d}.sqlite" for d in DISTRACTORS}}
    reports = {d: SSDEngramStore.build(fit if d == "all" else train_paths[d], path) for d, path in indexes.items()}
    if any(reports[d]["source_sha256"] != digest(train_paths[d]) for d in DISTRACTORS):
        raise ValueError("single-domain index did not bind the declared FIT source")
    models, training = {}, {}
    for d in indexes:
        models[d], training[d] = train(root, corpus, seed, d)
    domains = {d: score(probabilities(models["all"], raw), raw, indexes["all"], reports["all"])
               for d, raw in held.items()}
    ood = {d: score(probabilities(models[d], held[d]), held[d], indexes[other], reports[other])
           for d, other in DISTRACTORS.items()}
    count = sum(v["targets"] for v in domains.values())
    pooled = {arm: sum(v[f"{arm}_bits"] for v in domains.values()) / count for arm in ("neural", "engram")}
    prefix = prefix_checks(root, models["all"]) if seed == SEEDS[0] else None
    return dict(seed=seed, corpus=provenance, overlap_32b=0, training=training, indexes=reports,
                domains=domains, ood=ood, pooled=pooled, prefix=prefix)


def lookup_checks(root):
    source, index = root / "lookup.bin", root / "lookup.sqlite"
    atomic_bytes(source, random.Random(3319).randbytes(65536) * 3)
    started = time.perf_counter()
    report = SSDEngramStore.build(source, index)
    elapsed = time.perf_counter() - started
    trials = [json.loads(subprocess.check_output(
        [sys.executable, "-c", LOOKUP_WORKER, str(index), report["index_sha256"]], cwd=ROOT)) for _ in range(2)]
    return dict(build_seconds=elapsed, report=report, trials=trials, exact=trials[0]["digest"] == trials[1]["digest"])


def verdict(rows, lookup, guard):
    scores = [v for r in rows for arm in ("domains", "ood") for v in r[arm].values()]
    donor = guard.donor_forward_calls + sum(t["donor_forward_calls"] for r in rows for t in r["training"].values())
    wins = sum(r["pooled"]["engram"] < r["pooled"]["neural"] for r in rows)
    drift = max(v["delta_bpb"] for r in rows for v in r["ood"].values())
    safety = drift <= 0.01
    prefix = rows[0]["prefix"]
    speed = max(t["ms_per_byte"] for t in lookup["trials"])
    rss = max(t["peak_rss_bytes"] for t in lookup["trials"])
    cold_diff = max(v["cold_max_diff"] for v in scores)
    artifacts = all(t["cold_equal"] and t["updates"] == STEPS and t["target_bytes"] == STEPS * CONTEXT * 3
                    and t["archive"]["exact_state"] and t["archive"]["round_trip_probabilities"] == 0.0
                    and len(t["archive"]["parity"]) == 4 and not any(t["archive"]["parity"].values())
                    for r in rows for t in r["training"].values())
    engineering = (not donor and artifacts and all(r["overlap_32b"] == 0 for r in rows)
                   and prefix["active"] == 100 and not prefix["maximum"] and not prefix["failures"]
                   and not cold_diff and all(v["same_hits"] for v in scores)
                   and lookup["exact"] and speed <= 1.0 and rss <= 64 * 1024**2
                   and all(0 < t["hits"] < t["queries"] for t in lookup["trials"]))
    benefit = wins == len(SEEDS)
    science = "INVALID" if not engineering else "PASS" if benefit and safety else "MIXED" if benefit or safety else "FAIL"
    return dict(engineering="PASS" if engineering else "FAIL", scientific=science, pooled_win_seeds=wins,
                ood_max_degradation_bpb=drift, ood_pass=safety, donor_forward_calls=donor,
                cold_max_diff=cold_diff, max_lookup_ms_per_byte=speed, max_lookup_peak_rss_bytes=rss)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out.resolve()
    intent = out.with_suffix(".intent.json")
    head = git("rev-parse", "HEAD").decode().strip()
    if (out.is_relative_to(ROOT) or out.exists() or intent.exists() or git("status", "--porcelain")
            or git("rev-parse", "HEAD^").decode().strip() != BASE or not git("ls-tree", head, PREREG)):
        parser.error("run from the clean preregistration commit with a fresh external output")
    identity = dict(base=BASE, source_commit=head, source_tree=git("rev-parse", "HEAD^{tree}").decode().strip(),
                    prereg_sha256=digest(ROOT / PREREG), corpus_source_sha=SOURCE)
    atomic(intent, identity)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix="phase19-ssd-engram-") as scratch, forbid_donor_forward() as guard:
        root = Path(scratch)
        lookup = lookup_checks(root)
        for seed in SEEDS:
            directory = root / str(seed)
            directory.mkdir()
            rows.append(seed_run(directory, seed))
            print(f"Phase 19 seed {seed} complete", flush=True)
    if git("rev-parse", "HEAD").decode().strip() != head or git("status", "--porcelain"):
        parser.error("the committed source changed during scoring")
    summary = verdict(rows, lookup, guard)
    atomic(out, dict(schema="nexus.phase19-ssd-engram.v1", **identity, platform=platform.platform(),
                     torch=str(torch.__version__), offline=True, seconds=time.perf_counter() - start,
                     parent_peak_rss_bytes=peak_rss(), seeds=rows, lookup=lookup, summary=summary))
    print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
