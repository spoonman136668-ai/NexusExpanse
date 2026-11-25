"""Exact prefix, matched supervised capacity, and fresh-process CPU speed."""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from benchmarks.gate11_selective_kuramoto_dev import (
    Supervised, capacity_batches, capacity_eval, capacity_loss, expanded_train, identity, manifest, setup,
)
from benchmarks.gate12_deltanet_dev import ATTENTION, DELTA, DIMS, SEEDS, run_spec, timing as attention_timing
from benchmarks.gate9_jepa_dev import atomic, digest
from benchmarks.gate_custody import EpochRun, peak_rss, prepare_directory, update, writer
from nexus_expanse.hybrid_backbone import HybridDeltaAttentionCore
from nexus_expanse.jepa import state_hash

GATE12 = ROOT / "evidence/gate12-deltanet-dev-001.json"
EPOCHS = 15
STEPS = 200
HYBRID = "hybrid"


def model_for(seed):
    torch.manual_seed(seed)
    model = Supervised(ATTENTION)
    torch.manual_seed(seed)
    model.encoder = HybridDeltaAttentionCore()
    return model


@torch.inference_mode()
def parity(seeds=50):
    runs = []
    for dim in DIMS:
        for seed in range(seeds):
            torch.manual_seed(seed)
            model = HybridDeltaAttentionCore(dim=dim).eval()
            tokens = torch.randint(256, (1, 32))
            logits, hidden, _ = model.forward_neural(tokens)
            result = dict(seed=seed, dim=dim, prefix_logits=0.0, prefix_hidden=0.0,
                          streaming_logits=0.0, streaming_hidden=0.0,
                          streaming_probabilities=0.0)
            for cut in (1, 7, 13, 25, 31):
                a, b, _ = model.forward_neural(tokens[:, :cut])
                result["prefix_logits"] = max(result["prefix_logits"], (a - logits[:, :cut]).abs().max().item())
                result["prefix_hidden"] = max(result["prefix_hidden"], (b - hidden[:, :cut]).abs().max().item())
            stream = model.new_stream()
            size = stream.tensor_bytes
            for offset, byte in enumerate(tokens[0].tolist()):
                a, b = stream.step(byte)
                result["streaming_logits"] = max(result["streaming_logits"], (a - logits[0, offset]).abs().max().item())
                result["streaming_hidden"] = max(result["streaming_hidden"], (b - hidden[0, offset]).abs().max().item())
                p = a.double().softmax(0)
                ref = logits[0, offset].double().softmax(0)
                result["streaming_probabilities"] = max(result["streaming_probabilities"], (p - ref).abs().max().item())
                if stream.tensor_bytes != size:
                    raise ValueError("stream state footprint changed")
            result["state_tensor_bytes"] = size
            runs.append(result)
    maxima = {k: max(r[k] for r in runs) for k in runs[0] if k.startswith(("prefix_", "streaming_"))}
    return dict(seeds=seeds, dims=list(DIMS), maxima=maxima, passed=not any(maxima.values()), runs=runs)


def spec(directory, ident, seed):
    return EpochRun(directory, f"{seed}-{HYBRID}", ident, EPOCHS, STEPS)


def subrun(*args):
    return json.loads(subprocess.check_output([sys.executable, __file__, *map(str, args)], text=True))


def run_seed(directory, ident, seed, epoch_limit):
    split = manifest()["splits"]
    ev = split["eval"]
    train = expanded_train(split["train"], ev)
    model = model_for(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)

    def epoch(number):
        losses = []
        for rows in capacity_batches(train, seed, number):
            opt.zero_grad(set_to_none=True)
            losses.append(update(model, opt, capacity_loss(model, rows)))
        return losses

    spec(directory, ident, seed).advance(
        model, opt, epoch_limit, epoch, lambda path: subrun("--cold", path, "--seed", seed),
        lambda: dict(train=capacity_eval(model, train), heldout=capacity_eval(model, ev)),
        state_hash(model.encoder),
    )


@torch.inference_mode()
def timing(seed, dim, checkpoint):
    torch.manual_seed(seed)
    if dim == 64:
        supervised = model_for(seed)
        supervised.load_state_dict(torch.load(checkpoint, weights_only=True))
        net = supervised.encoder.eval()
    else:
        net = HybridDeltaAttentionCore(dim=dim).eval()
    stream = net.new_stream()
    data = np.random.default_rng(seed).integers(0, 256, size=net.max_seq_len).tolist()
    ref = net.forward_neural(torch.tensor([data]))[0][0]
    actual = torch.stack([stream.append(b) for b in data])
    error = (ref - actual).abs().max().item()
    if error != 0.0:
        raise ValueError(f"timed hybrid checkpoint failed exactness: {error}")

    def measure():
        stream.reset()
        start = time.perf_counter()
        for byte in data:
            stream.append(byte).double().softmax(0)
        return (time.perf_counter() - start) * 1000 / len(data)

    measure()
    samples = [measure() for _ in range(5)]
    return dict(ms_per_byte=statistics.median(samples), samples_ms_per_byte=samples,
                tensor_bytes=stream.tensor_bytes, parity_max_logit_diff=error,
                fresh_process_peak_rss_bytes=peak_rss(),
                weights="trained" if dim == 64 else "initialized")


def decision(runs, exact):
    scores = [r["capacity"]["heldout"][task]["accuracy"] for r in runs for task in ("math", "science")]
    cells = [t for r in runs for t in r["timing"].values()]
    reasoning = min(scores) >= .95
    speed_cache = all(c["speedup"] >= 1.5 and c["cache_reduction"] >= .40 for c in cells)
    gains = all(c["speedup"] > 1.0 and c["cache_reduction"] > 0.0 for c in cells)
    if not exact or min(scores) < .90 or not gains:
        status = "FAIL"
    elif reasoning and speed_cache:
        status = "PASS"
    elif reasoning or speed_cache:
        status = "MIXED"
    else:
        status = "FAIL"
    return dict(status=status, reasoning_bar=reasoning, speed_cache_bar=speed_cache,
                minimum_reasoning=min(scores), positive_speed_and_cache=gains)


def merge(directory, control_dir, ident, engineering, dest):
    previous = json.loads(GATE12.read_text())
    runs = []
    for seed in SEEDS:
        hybrid = spec(directory, ident, seed).read()
        historical = next(r for r in previous["runs"] if r["seed"] == seed)["capacity"]
        control = run_spec(control_dir, previous["identity"], seed, ATTENTION)
        if control.read() != historical[ATTENTION]:
            raise ValueError("attention control checkpoint receipt differs from Gate 12")
        cells = {}
        for dim in DIMS:
            by_arm = {}
            for arm in ((HYBRID, ATTENTION) if seed % 2 == 0 else (ATTENTION, HYBRID)):
                source = directory / hybrid["checkpoint"] if arm == HYBRID else control.checkpoint
                by_arm[arm] = subrun("--timing", source, "--seed", seed, "--dim", dim, "--arm", arm)
            h, a = by_arm[HYBRID], by_arm[ATTENTION]
            cells[str(dim)] = dict(by_arm, speedup=a["ms_per_byte"] / h["ms_per_byte"],
                                   cache_reduction=1 - h["tensor_bytes"] / a["tensor_bytes"])
        runs.append(dict(seed=seed, capacity=hybrid, timing=cells,
                         gate12_capacity={arm: historical[arm]["heldout"] for arm in (ATTENTION, DELTA)}))
    verdict = decision(runs, engineering["passed"])
    atomic(dest, dict(schema="nexus-gate15-hybrid-dev-v1", qualification="REUSED_AUTHORED_RELATIONAL_DEV",
                      identity=ident, parity=engineering, runs=runs, **verdict))
    print(json.dumps(verdict))


def main():
    setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path)
    p.add_argument("--control-directory", type=Path)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--arm", choices=(HYBRID, ATTENTION), default=HYBRID)
    p.add_argument("--dim", type=int, choices=DIMS)
    p.add_argument("--epoch-limit", type=int, default=2, choices=range(1, EPOCHS + 1))
    action = p.add_mutually_exclusive_group()
    action.add_argument("--parity", action="store_true")
    action.add_argument("--merge", type=Path)
    action.add_argument("--cold", type=Path)
    action.add_argument("--timing", type=Path)
    args = p.parse_args()
    if args.cold:
        start = time.monotonic()
        model = model_for(args.seed)
        model.load_state_dict(torch.load(args.cold, weights_only=True))
        print(json.dumps([state_hash(model), time.monotonic() - start, peak_rss()]))
        return
    if args.timing:
        result = timing(args.seed, args.dim, args.timing) if args.arm == HYBRID else attention_timing(
            args.seed, ATTENTION, args.dim, args.timing)
        print(json.dumps(result))
        return
    if args.directory is None:
        p.error("--directory required")
    ident = dict(identity(), gate12_sha256=digest(GATE12))
    directory = prepare_directory(args.directory, ROOT)
    parity_path = directory / "parity.json"
    if args.parity:
        with writer(directory, "parity"):
            result = parity()
            atomic(parity_path, dict(identity=ident, report=result))
            print(json.dumps(result["maxima"]))
            sys.exit(not result["passed"])
    checked = json.loads(parity_path.read_text())
    if checked["identity"] != ident or not checked["report"]["passed"]:
        raise ValueError("matching passing engineering receipt required before training")
    if args.merge and args.control_directory:
        merge(directory, args.control_directory.resolve(), ident, checked["report"], args.merge)
    elif args.seed is not None and not args.merge and args.arm == HYBRID:
        run_seed(directory, ident, args.seed, args.epoch_limit)
    else:
        p.error("provide --seed for training, or --control-directory with --merge")


if __name__ == "__main__":
    main()
