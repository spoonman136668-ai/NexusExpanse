"""Frozen stabilized-hybrid capacity and two-width CPU timing experiment."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from benchmarks.dev_artifacts import atomic, digest
from benchmarks.gate11_selective_kuramoto_dev import (
    Supervised, capacity_batches, capacity_eval, capacity_loss, expanded_train, identity, manifest, setup,
)
from benchmarks.gate12_deltanet_dev import ATTENTION, run_spec, timing as attention_timing
from benchmarks.gate15_hybrid_dev import DIMS, EPOCHS, STEPS, parity
from benchmarks.gate_custody import EpochRun, peak_rss, prepare_directory, update, writer
from nexus_expanse.hybrid_backbone import HybridDeltaAttentionCore
from nexus_expanse.jepa import state_hash

SEEDS = (42, 43, 44, 45, 46)
V2 = "hybrid_v2"
GATE12 = ROOT / "evidence/gate12-deltanet-dev-001.json"
GATE15 = ROOT / "evidence/gate15-hybrid-dev-001.json"


def rate(step, total, base=.003):
    warm = total // 10
    if not 1 <= step <= total or warm < 1:
        raise ValueError("invalid optimizer step")
    if step <= warm:
        return base * step / warm
    return .5 * base * (1 + math.cos(math.pi * (step - warm) / (total - warm)))


def model_for(seed, arm):
    torch.manual_seed(seed)
    model = Supervised(ATTENTION)
    if arm == V2:
        torch.manual_seed(seed)
        model.encoder = HybridDeltaAttentionCore(stabilized=True)
    return model


def spec(directory, ident, seed, arm):
    return EpochRun(directory, f"{seed}-{arm}", ident, EPOCHS, STEPS)


def subrun(*args):
    return json.loads(subprocess.check_output([sys.executable, __file__, *map(str, args)], text=True))


def run_seed(directory, ident, seed, arm, epoch_limit):
    split = manifest()["splits"]
    ev = split["eval"]
    train = expanded_train(split["train"], ev)
    model = model_for(seed, arm)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)

    def epoch(number):
        result = []
        for i, rows in enumerate(capacity_batches(train, seed, number), 1):
            if arm == V2:
                opt.param_groups[0]["lr"] = rate((number - 1) * STEPS + i, EPOCHS * STEPS)
            opt.zero_grad(set_to_none=True)
            result.append(update(model, opt, capacity_loss(model, rows)))
        return result

    spec(directory, ident, seed, arm).advance(
        model, opt, epoch_limit, epoch, lambda path: subrun("--cold", path, "--seed", seed, "--arm", arm),
        lambda: dict(heldout=capacity_eval(model, ev)), state_hash(model.encoder),
    )


@torch.inference_mode()
def timing(seed, dim, path):
    torch.manual_seed(seed)
    if dim == 64:
        model = model_for(seed, V2)
        model.load_state_dict(torch.load(path, weights_only=True))
        net = model.encoder.eval()
    else:
        net = HybridDeltaAttentionCore(dim=dim, stabilized=True).eval()
    stream = net.new_stream()
    data = np.random.default_rng(seed).integers(0, 256, size=net.max_seq_len).tolist()
    expected = net.forward_neural(torch.tensor([data]))[0][0]
    actual = torch.stack([stream.append(b) for b in data])
    error = (actual - expected).abs().max().item()
    if error != 0.:
        raise ValueError("trained hybrid timing parity failed")

    def measure():
        stream.reset()
        start = time.perf_counter()
        for b in data:
            stream.append(b).double().softmax(0)
        return (time.perf_counter() - start) * 1000 / len(data)

    measure()
    samples = [measure() for _ in range(5)]
    return dict(ms_per_byte=statistics.median(samples), samples_ms_per_byte=samples,
                tensor_bytes=stream.tensor_bytes, parity_max_logit_diff=error,
                fresh_process_peak_rss_bytes=peak_rss(),
                weights="trained" if dim == 64 else "initialized")


def decision(runs, exact):
    eligible = [r["seed"] for r in runs if all(v["accuracy"] >= .95 for v in r["v2"]["heldout"].values())]
    fast = all(c["speedup"] >= 5. and c["cache_reduction"] >= .40
               for r in runs for c in r["timing"].values())
    status = "FAIL" if not exact or len(eligible) <= 3 else "PASS" if len(eligible) == 5 and fast else "MIXED"
    return dict(status=status, qualifying_seeds=eligible, speed_cache_bar=fast)


def merge(directory, control_directory, ident, checked, dest):
    prior = json.loads(GATE12.read_text())
    historic = {r["seed"]: r["capacity"][ATTENTION] for r in prior["runs"]}
    runs = []
    for seed in SEEDS:
        candidate = spec(directory, ident, seed, V2)
        if seed in historic:
            control = run_spec(control_directory, prior["identity"], seed, ATTENTION)
            if control.read() != historic[seed]:
                raise ValueError("published historical control does not match checkpoint custody")
        else:
            control = spec(directory, ident, seed, ATTENTION)
        fit = candidate.read()
        reference = control.read()
        cells = {}
        for dim in DIMS:
            pair = {}
            for arm, trial in ((V2, candidate), (ATTENTION, control))[::1 if seed % 2 == 0 else -1]:
                pair[arm] = subrun("--timing", trial.checkpoint, "--seed", seed, "--dim", dim, "--arm", arm)
            a, b = pair[V2], pair[ATTENTION]
            cells[str(dim)] = dict(pair, speedup=b["ms_per_byte"] / a["ms_per_byte"],
                                   cache_reduction=1 - a["tensor_bytes"] / b["tensor_bytes"])
        runs.append(dict(seed=seed, v2=fit, attention=reference, timing=cells))
    result = decision(runs, checked["passed"])
    atomic(dest, dict(schema="nexus-gate18-hybrid-v2-dev-v1", qualification="REUSED_AUTHORED_RELATIONAL_DEV",
                      identity=ident, parity=checked, runs=runs, **result))
    print(json.dumps(result))


def main():
    setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path)
    p.add_argument("--control-directory", type=Path)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--dim", type=int, choices=DIMS)
    p.add_argument("--arm", choices=(V2, ATTENTION), default=V2)
    p.add_argument("--epoch-limit", type=int, choices=range(1, 16), default=3)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--parity", action="store_true")
    group.add_argument("--cold", type=Path)
    group.add_argument("--timing", type=Path)
    group.add_argument("--merge", type=Path)
    a = p.parse_args()
    if a.cold:
        start = time.monotonic()
        model = model_for(a.seed, a.arm)
        model.load_state_dict(torch.load(a.cold, weights_only=True))
        print(json.dumps([state_hash(model), time.monotonic() - start, peak_rss()]))
        return
    if a.timing:
        r = timing(a.seed, a.dim, a.timing) if a.arm == V2 else attention_timing(a.seed, a.arm, a.dim, a.timing)
        print(json.dumps(r))
        return
    directory = prepare_directory(a.directory, ROOT)
    ident = dict(identity(), gate12_sha256=digest(GATE12), gate15_sha256=digest(GATE15))
    path = directory / "parity.json"
    if a.parity:
        with writer(directory, "parity"):
            report = parity(stabilized=True)
            atomic(path, dict(identity=ident, report=report))
        print(json.dumps(report["maxima"]))
        sys.exit(not report["passed"])
    checked = json.loads(path.read_text())
    if checked["identity"] != ident or not checked["report"]["passed"]:
        raise ValueError("passing parity at the frozen source required")
    if a.merge and a.control_directory:
        merge(directory, a.control_directory.resolve(), ident, checked["report"], a.merge)
    elif a.seed is not None and (a.arm == V2 or a.seed in (45, 46)):
        run_seed(directory, ident, a.seed, a.arm, a.epoch_limit)
    else:
        p.error("provide the applicable seed and arm or merge controls")


if __name__ == "__main__":
    main()
