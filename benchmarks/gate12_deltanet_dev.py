"""Gate 12 expanded capacity, exact causal parity, and CPU stream measurements."""
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

from benchmarks.gate_custody import EpochRun, peak_rss, prepare_directory, update, writer
from benchmarks.gate9_jepa_dev import atomic
from benchmarks.gate11_selective_kuramoto_dev import (
    SEEDS, Supervised, attention, capacity_batches, capacity_eval, capacity_loss,
    expanded_train, identity, manifest, setup,
)
from nexus_expanse.deltanet import DeltaNetMonarchByteCore
from nexus_expanse.jepa import state_hash

DELTA = "deltanet_monarch"
ATTENTION = "causal_attention_monarch"
ARMS = (DELTA, ATTENTION)
DIMS = (64, 128)
EPOCHS = 15
STEPS = 200
PRIOR = "evidence/gate11-selective-kuramoto-complete-summary-dev-001.json"
BASE = "70c0c6b3b5f668893ca97fbaecfe6cd5f78f39aa"


def model_for(seed, arm):
    torch.manual_seed(seed)
    model = Supervised(ATTENTION)
    if arm == DELTA:
        torch.manual_seed(seed)
        model.encoder = DeltaNetMonarchByteCore()
    return model


def state_bytes(states):
    return sum(t.numel() * t.element_size() for t in states)


@torch.inference_mode()
def parity(seeds=50):
    runs = []
    for dim in DIMS:
        for seed in range(seeds):
            torch.manual_seed(seed)
            model = DeltaNetMonarchByteCore(dim=dim).eval()
            tokens = torch.randint(256, (1, 32))
            logits, hidden, _ = model.forward_neural(tokens)
            record = dict(dim=dim, seed=seed, prefix_logits=0.0, prefix_hidden=0.0,
                          streaming_logits=0.0, streaming_hidden=0.0,
                          streaming_probabilities=0.0)
            for cut in (1, 7, 13, 25, 31):
                a, b, _ = model.forward_neural(tokens[:, :cut])
                record["prefix_logits"] = max(record["prefix_logits"], (logits[:, :cut] - a).abs().max().item())
                record["prefix_hidden"] = max(record["prefix_hidden"], (hidden[:, :cut] - b).abs().max().item())
            states = model.init_stream_state()
            size = state_bytes(states)
            for offset, byte in enumerate(tokens[0].tolist()):
                a, b, states = model.step_logits(byte, states)
                record["streaming_logits"] = max(record["streaming_logits"], (logits[0, offset] - a).abs().max().item())
                record["streaming_hidden"] = max(record["streaming_hidden"], (hidden[0, offset] - b).abs().max().item())
                probability = logits[0, offset].double().softmax(0)
                record["streaming_probabilities"] = max(
                    record["streaming_probabilities"], (probability - a.double().softmax(0)).abs().max().item(),
                )
                if state_bytes(states) != size or any(s.dtype != torch.float64 for s in states):
                    raise ValueError("streaming state grew or changed dtype")
            record["state_tensor_bytes"] = size
            runs.append(record)
    maxima = {key: max(r[key] for r in runs) for key in runs[0] if key.startswith(("prefix_", "streaming_"))}
    return dict(seeds=seeds, dimensions=list(DIMS), maxima=maxima, zero_kv_cache=True,
                passed=all(v == 0.0 for v in maxima.values()), runs=runs)


def run_spec(directory, ident, seed, arm):
    return EpochRun(directory, f"{seed}-{arm}", ident, EPOCHS, STEPS)


def subrun(*args):
    return json.loads(subprocess.check_output([sys.executable, __file__, *map(str, args)], text=True))


def run_arm(directory, ident, seed, arm, epoch_limit):
    train, ev = (manifest()["splits"][key] for key in ("train", "eval"))
    train = expanded_train(train, ev)
    model = model_for(seed, arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)

    def epoch(number):
        losses = []
        for rows in capacity_batches(train, seed, number):
            optimizer.zero_grad(set_to_none=True)
            losses.append(update(model, optimizer, capacity_loss(model, rows)))
        return losses

    spec = run_spec(directory, ident, seed, arm)
    return spec.advance(
        model, optimizer, epoch_limit, epoch,
        lambda path: subrun("--cold", path, "--seed", seed, "--arm", arm),
        lambda: dict(train=capacity_eval(model, train), heldout=capacity_eval(model, ev)),
        state_hash(model.encoder),
    )


@torch.inference_mode()
def timing(seed, arm, dim, checkpoint):
    torch.manual_seed(seed)
    tensor_bytes, weights = 0, "initialized"
    if dim == 64:
        supervised = model_for(seed, arm)
        supervised.load_state_dict(torch.load(checkpoint, weights_only=True))
        backbone = supervised.encoder
        weights = "trained"
    else:
        backbone = DeltaNetMonarchByteCore(dim=dim) if arm == DELTA else attention(dim)
    backbone.eval()
    stream = np.random.default_rng(seed).integers(0, 256, size=256).tolist()
    parity_diff = 0.0
    if arm == DELTA:
        tensor_bytes = state_bytes(backbone.init_stream_state())

        def measure():
            states = backbone.init_stream_state()
            started = time.perf_counter()
            for byte in stream:
                _, states = backbone.step_streaming(byte, states)
            return 1000 * (time.perf_counter() - started) / len(stream)
    else:
        for layer in backbone.layers:
            layer.ffn.enable_skip = True
            layer.ffn.tau_eval.fill_(-torch.inf)
        cache = backbone.new_kv_cache()
        tensor_bytes = cache.tensor_bytes
        reference = backbone.forward_neural(torch.tensor([stream]))[0][0]
        cached = torch.stack([cache.append(byte) for byte in stream])
        parity_diff = (reference - cached).abs().max().item()
        if not torch.allclose(reference, cached, atol=2e-5, rtol=1e-5):
            raise ValueError(f"attention speed control failed kernel parity: {parity_diff}")

        def measure():
            cache.reset()
            started = time.perf_counter()
            for byte in stream:
                cache.append(byte).double().softmax(0)
            return 1000 * (time.perf_counter() - started) / len(stream)

    measure()
    samples = [measure() for _ in range(5)]
    return dict(ms_per_byte=statistics.median(samples), samples_ms_per_byte=samples,
                tensor_bytes=tensor_bytes, weights=weights, parity_max_logit_diff=parity_diff,
                fresh_process_peak_rss_bytes=peak_rss())


def decision(runs, checked_parity):
    def reaches(math, science):
        return all(r["capacity"][DELTA]["heldout"]["math"]["accuracy"] >= math
                   and r["capacity"][DELTA]["heldout"]["science"]["accuracy"] >= science
                   for r in runs)

    science_ready = reaches(.50, .75)
    high_accuracy = reaches(.85, .90)
    fast = all(cell["speedup"] >= 2.0 for r in runs for cell in r["timing"].values())
    exact = checked_parity["passed"]
    status = "FAIL"
    if exact and high_accuracy and fast:
        status = "PASS"
    elif exact and (science_ready or fast):
        status = "MIXED"
    return dict(status=status, exact=exact, capacity_bar=high_accuracy,
                speed_bar=fast, delta_eligible_for_gate13=exact and science_ready)


def merge(directory, ident, checked_parity, dest):
    previous = subprocess.check_output(["git", "show", f"{BASE}:{PRIOR}"], cwd=ROOT)
    historical = json.loads(previous)["capacity"]["decisive"]
    runs = []
    for seed in SEEDS:
        capacity = {arm: run_spec(directory, ident, seed, arm).read() for arm in ARMS}
        measurements = {}
        for dim in DIMS:
            cell = {}
            for arm in (ARMS if seed % 2 == 0 else ARMS[::-1]):
                cell[arm] = subrun("--timing", directory / capacity[arm]["checkpoint"],
                                   "--seed", seed, "--arm", arm, "--dim", dim)
            cell["speedup"] = cell[ATTENTION]["ms_per_byte"] / cell[DELTA]["ms_per_byte"]
            measurements[str(dim)] = cell
        runs.append(dict(seed=seed, capacity=capacity, timing=measurements,
                         historical_gate11=historical[str(seed)]))
    result = dict(schema="nexus-gate12-deltanet-dev-v1", qualification="REUSED_AUTHORED_RELATIONAL_DEV",
                  identity=ident, parity=checked_parity, runs=runs, **decision(runs, checked_parity))
    atomic(dest, result)
    print(json.dumps({key: result[key] for key in decision(runs, checked_parity)}))


def main():
    setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path)
    p.add_argument("--epoch-limit", type=int, default=2, choices=range(1, 16))
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--dim", type=int, choices=DIMS)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--parity", action="store_true")
    action.add_argument("--merge", type=Path)
    action.add_argument("--cold", type=Path)
    action.add_argument("--timing", type=Path)
    args = p.parse_args()
    if args.cold:
        start = time.monotonic()
        model = model_for(args.seed, args.arm)
        model.load_state_dict(torch.load(args.cold, weights_only=True))
        print(json.dumps([state_hash(model), time.monotonic() - start, peak_rss()]))
        return
    if args.timing:
        print(json.dumps(timing(args.seed, args.arm, args.dim, args.timing)))
        return
    if args.directory is None:
        p.error("--directory required")
    ident = identity()
    directory = prepare_directory(args.directory, ROOT)
    parity_path = directory / "parity.json"
    if args.parity:
        with writer(directory, "parity"):
            report = parity()
            atomic(parity_path, dict(identity=ident, report=report))
            print(json.dumps(report["maxima"]))
            sys.exit(not report["passed"])
    report = json.loads(parity_path.read_text())
    if report["identity"] != ident or not report["report"]["passed"]:
        raise ValueError("frozen Gate 12 needs passing parity before training")
    if args.merge:
        merge(directory, ident, report["report"], args.merge)
    elif args.seed is not None and args.arm is not None:
        run_arm(directory, ident, args.seed, args.arm, args.epoch_limit)
    else:
        p.error("--seed and --arm required for training")


if __name__ == "__main__":
    main()
