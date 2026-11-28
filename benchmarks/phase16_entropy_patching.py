"""Matched-step CPU comparison of flat and entropy-patched byte cores."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import gate17_gpu_scaling_harness as scaling
from benchmarks.dev_artifacts import atomic, digest, save_pt
from benchmarks.gate_custody import peak_rss, update
from benchmarks.phase13_warmstart_train import build_corpus
from benchmarks.phase3_multi_skill_routing import SOURCE, git
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.entropy_patching import EntropyPatchedByteCore
from nexus_expanse.jepa import state_hash

BASE = "d1936aaa760c644915653ddc1dcefedf0ce32aa6"
PREREG = "docs/PHASE16_ENTROPY_BYTE_PATCHING_PREREG.md"
DOMAINS = scaling.DOMAINS
SEEDS = tuple(range(3001, 3006))
CONTEXTS = (512, 1024)
TRAIN_STEPS = 48
REPETITIONS = 3
CONFIG = dict(dim=32, n_heads=4, n_layers=2, ffn_mult=2, max_seq_len=1024)
PATCH = dict(entropy_threshold_bits=5.0, max_patch_bytes=8, local_kernel_size=3)
CPU = torch.device("cpu")


def tensor(raw):
    return torch.tensor(list(raw))[None]


def max_difference(a, b):
    return float((a - b).abs().max()) if a.numel() else 0.0


@torch.inference_mode()
def prefix_case(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        fitted = b"abc def! abc abc def! " * 150 if seed % 2 else b""
        model = EntropyPatchedByteCore(dim=16, max_seq_len=128, training_bytes=fitted, **PATCH).eval()
        vocabulary = tensor(b"abc def!").flatten() if seed % 2 else torch.arange(256)
        x = vocabulary[torch.randint(len(vocabulary), (1, 95))]
        cut = int(torch.randint(1, x.shape[1], ()))
        changed = x.clone()
        changed[:, cut:] = (x[:, cut:] + torch.randint(1, 256, x[:, cut:].shape)) % 256
        logits, hidden, trace = model._forward(x)
        differences = dict(boundaries=0, completed_ends=0, completed_latents=0.0,
                           visible_latents=0.0, logits=0.0, hidden=0.0)
        known = (trace.completed_ends >= 0) & (trace.completed_ends < cut - 1)
        for candidate in (x[:, :cut], changed):
            a, b, t = model._forward(candidate)
            current = (t.completed_ends >= 0) & (t.completed_ends < cut - 1)
            differences["boundaries"] += int((trace.starts[:, :cut] != t.starts[:, :cut]).sum())
            equal_ends = torch.equal(trace.completed_ends[known], t.completed_ends[current])
            differences["completed_ends"] += int(not equal_ends)
            if equal_ends:
                differences["completed_latents"] = max(differences["completed_latents"],
                                                       max_difference(trace.latents[known], t.latents[current]))
            for key, value in (
                ("visible_latents", max_difference(trace.visible_latents[:, :cut], t.visible_latents[:, :cut])),
                ("logits", max_difference(logits[:, :cut], a[:, :cut])),
                ("hidden", max_difference(hidden[:, :cut], b[:, :cut])),
            ):
                differences[key] = max(differences[key], value)
        return dict(seed=seed, prefix_bytes=cut, **differences)


def models(seed, train):
    torch.manual_seed(seed)
    flat = NexusExpanseByteCore(**CONFIG, enable_skip=False, dropout=0.0, ffn_kind="dense", use_sdpa=True,
                               untied_readout=True)
    torch.manual_seed(seed)
    patched = EntropyPatchedByteCore(**CONFIG, **PATCH, training_bytes=train.values())
    reference = flat.state_dict()
    for name, value in patched.state_dict().items():
        if name in reference and not torch.equal(value, reference[name]):
            raise ValueError(f"matched initialization differs at {name}")
    if not torch.equal(patched.patch_position.weight, flat.pos_emb.weight):
        raise ValueError("matched position initialization differs")
    return {"flat": flat, "patched": patched}


def batch_for(raw, block):
    batch, keep, valid = scaling.make_batch([(raw, block)], CONTEXTS[0], CPU)
    return batch[:, :-1], batch[:, 1:], keep, valid


def train_arm(model, batches):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    with torch.no_grad():
        model.forward_neural(batches[0][0])
    target_bytes = 0
    losses = []
    started = time.perf_counter()
    for tokens, expected, keep, valid in batches:
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_neural(tokens)[0]
        losses.append(update(model, optimizer, F.cross_entropy(logits[keep], expected[keep])))
        target_bytes += valid
    elapsed = time.perf_counter() - started
    model.eval()
    return dict(updates=len(losses), losses=losses, target_bytes=target_bytes, wall_seconds=elapsed,
                target_bytes_per_second=target_bytes / elapsed, state_sha256=state_hash(model),
                parameters=sum(p.numel() for p in model.parameters()))


@torch.inference_mode()
def evaluate(model, batches):
    scores = {}
    for domain, chunks in batches.items():
        loss, target_bytes = 0.0, 0
        for x, y, keep, valid in chunks:
            loss += float(F.cross_entropy(model.forward_neural(x)[0][keep], y[keep], reduction="sum"))
            target_bytes += valid
        scores[domain] = dict(bpb=loss / (target_bytes * math.log(2)), target_bytes=target_bytes)
    total = sum(item["target_bytes"] for item in scores.values())
    pooled = sum(item["target_bytes"] * item["bpb"] for item in scores.values()) / total
    return dict(domains=scores, pooled_bpb=pooled)


@torch.inference_mode()
def speed(arms, held, context, order):
    arms["flat"].max_seq_len = context
    inputs = {domain: tensor(raw[:context]) for domain, raw in held.items()}
    reports = {}
    for domain, tokens in inputs.items():
        for model in arms.values():
            model.forward_neural(tokens)
        times = {name: [] for name in arms}
        for trial in range(REPETITIONS):
            for name in (order if trial % 2 == 0 else order[::-1]):
                started = time.perf_counter()
                arms[name].forward_neural(tokens)
                times[name].append(time.perf_counter() - started)
        counts = arms["patched"].patch_stats(tokens)
        trace = arms["patched"].patch_trace(tokens)
        completed = int((trace.completed_ends >= 0).sum())
        width = arms["patched"].PATCH_BLOCK
        reports[domain] = dict(**counts, completed_patch_steps=completed,
                               padded_attention_steps=math.ceil(completed / width) * width, trials_seconds=times,
                               flat_bytes_per_second=context / statistics.median(times["flat"]),
                               patched_bytes_per_second=context / statistics.median(times["patched"]))
    seconds = {arm: sum(statistics.median(item["trials_seconds"][arm]) for item in reports.values()) for arm in arms}
    total = sum(row["byte_count"] for row in reports.values())
    return dict(domains=reports, compression_ratio=total / sum(row["patch_count"] for row in reports.values()),
                flat_bytes_per_second=total / seconds["flat"], patched_bytes_per_second=total / seconds["patched"],
                speedup=seconds["flat"] / seconds["patched"])


@torch.inference_mode()
def check_stream(model, tokens, expected):
    stream = model.new_stream()
    logits, hidden = zip(*(stream.append(byte) for byte in tokens[0].tolist()))
    return dict(logits=max_difference(torch.stack(logits), expected[0][0]),
                hidden=max_difference(torch.stack(hidden), expected[1][0]))


def tensor_digest(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


@torch.inference_mode()
def restore_probe(path):
    data = torch.load(path, weights_only=True, map_location="cpu")
    model = EntropyPatchedByteCore(**CONFIG, **PATCH).eval()
    model.load_state_dict(data["state"], strict=True)
    forecasts = {d: [tensor_digest(t) for t in model.forward_neural(x)] for d, x in data["inputs"].items()}
    return dict(state_sha256=state_hash(model), worker_pid=os.getpid(), forecast_digests=forecasts)


def parity(root, model, held):
    inputs = {d: tensor(raw[:CONTEXTS[0]]) for d, raw in held.items()}
    with torch.inference_mode():
        forecasts = {d: model.forward_neural(x) for d, x in inputs.items()}
        streamed = {d: check_stream(model, x, forecasts[d]) for d, x in inputs.items()}
        expected = {d: [tensor_digest(t) for t in pair] for d, pair in forecasts.items()}
    path = root / "restore.pt"
    save_pt(path, dict(state=model.state_dict(), inputs=inputs))
    cold = json.loads(subprocess.check_output([sys.executable, __file__, "--restore", str(path)], cwd=ROOT))
    exact = (cold["worker_pid"] != os.getpid() and cold["state_sha256"] == state_hash(model)
             and cold["forecast_digests"] == expected)
    return dict(streaming=streamed, cold=cold, forecast_digests=expected, exact_cold_restore=exact)


def seed_run(root, seed):
    corpus = root / "corpus"
    provenance = build_corpus(corpus, seed)
    train, held = ({d: (corpus / f"{d}-{split}.bin").read_bytes() for d in DOMAINS} for split in ("train", "eval"))
    hashes = {path.name: digest(path) for path in sorted(corpus.iterdir())}
    arms = models(seed, train)
    batches = []
    for step in range(TRAIN_STEPS):
        domain_index = step % len(DOMAINS)
        raw = train[DOMAINS[domain_index]]
        block = scaling.shuffled_index(step // len(DOMAINS), scaling.block_count(raw, CONTEXTS[0]), seed + domain_index)
        batches.append(batch_for(raw, block))
    order = tuple(arms) if seed % 2 else tuple(reversed(arms))
    training = {name: train_arm(arms[name], batches) for name in order}
    validation = {d: [batch_for(raw, i) for i in range(scaling.block_count(raw, CONTEXTS[0]))] for d, raw in held.items()}
    arms["flat"].max_seq_len = CONTEXTS[0]
    scores = {name: evaluate(model, validation) for name, model in arms.items()}
    speeds = {str(ctx): speed(arms, held, ctx, order) for ctx in CONTEXTS}
    checks = parity(root, arms["patched"], held)
    return dict(seed=seed, corpus_provenance=provenance, corpus_sha256=hashes, training=training, validation=scores,
                forward=speeds, parity=checks, training_speedup=training["patched"]["target_bytes_per_second"] /
                training["flat"]["target_bytes_per_second"])


def verdict(rows, prefix, calls):
    prefix_exact = sum(not any(value for key, value in row.items() if key not in ("seed", "prefix_bytes")) for row in prefix)
    exact = sum(row["parity"]["exact_cold_restore"] and
                all(not any(d.values()) for d in row["parity"]["streaming"].values()) for row in rows)
    compression = min(context["compression_ratio"] for row in rows for context in row["forward"].values())
    deltas = [row["validation"]["patched"]["pooled_bpb"] - row["validation"]["flat"]["pooled_bpb"] for row in rows]
    mean_delta = statistics.mean(deltas)
    within = sum(delta <= 0.05 for delta in deltas)
    forward = {str(ctx): statistics.median(row["forward"][str(ctx)]["speedup"] for row in rows) for ctx in CONTEXTS}
    train = statistics.median(row["training_speedup"] for row in rows)
    engineering = prefix_exact == 100 and exact == len(SEEDS) and compression >= 2.0 and not calls
    quality, faster = mean_delta <= 0.05 and within >= 4, min(forward.values()) >= 1.25 and train >= 1.0
    science = ("INVALID" if not engineering else "PASS" if quality and faster else
               "MIXED" if quality or faster else "FAIL")
    return dict(scored_engineering="PASS" if engineering else "FAIL", scientific=science, prefix_exact_seeds=prefix_exact,
                exact_streaming_and_cold_seeds=exact, minimum_pooled_compression_ratio=compression,
                mean_pooled_delta_bpb=mean_delta, quality_within_0_05_seeds=within,
                median_forward_speedups=forward, median_training_speedup=train, donor_forward_calls=calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out", type=Path)
    action.add_argument("--restore", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.restore:
        with forbid_donor_forward() as guard:
            report = restore_probe(args.restore)
        print(json.dumps(dict(report, donor_forward_calls=guard.donor_forward_calls)))
        return
    out = args.out.resolve()
    intent = out.with_suffix(".intent.json")
    head = git("rev-parse", "HEAD").decode().strip()
    tree = git("rev-parse", "HEAD^{tree}").decode().strip()
    if (out.is_relative_to(ROOT) or out.exists() or intent.exists() or git("status", "--porcelain")
            or git("rev-parse", "HEAD^").decode().strip() != BASE or not git("ls-tree", head, PREREG)):
        parser.error("run once from the clean Phase 16 preregistration commit with a new external output")
    identity = dict(base=BASE, source_commit=head, source_tree=tree, prereg_sha256=digest(ROOT / PREREG))
    atomic(intent, identity)
    started = time.perf_counter()
    rows = []
    with forbid_donor_forward() as guard, tempfile.TemporaryDirectory(prefix="nexus-phase16-") as scratch:
        prefix = [prefix_case(seed) for seed in range(100)]
        for seed in SEEDS:
            root = Path(scratch) / str(seed)
            root.mkdir()
            rows.append(seed_run(root, seed))
            print(f"Phase 16 seed {seed} complete", flush=True)
    if git("rev-parse", "HEAD").decode().strip() != head or git("status", "--porcelain"):
        parser.error("the committed source changed during scoring")
    calls = guard.donor_forward_calls + sum(row["parity"]["cold"]["donor_forward_calls"] for row in rows)
    result = dict(schema="nexus.phase16-entropy-patching.v1", **identity, corpus_source_sha=SOURCE,
                  platform=platform.platform(), torch=str(torch.__version__), threads=1, device="cpu",
                  external_downloads=0, wall_seconds=time.perf_counter() - started, parent_peak_rss_bytes=peak_rss(),
                  prefix_cases=prefix, seeds=rows, summary=verdict(rows, prefix, calls))
    atomic(out, result)
    print(json.dumps(result["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
