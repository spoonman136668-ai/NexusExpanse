"""Matched sequential byte-stream retention with exact cold return."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from benchmarks.dev_artifacts import atomic, digest, save_pt
from benchmarks.gate4_corpus import MANIFEST, read_split
from benchmarks.gate11_selective_kuramoto_dev import SEEDS, identity, manifest, setup
from benchmarks.gate_custody import peak_rss, prepare_directory, writer
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.jepa import state_hash
from nexus_expanse.patching import CompactBigram
from nexus_expanse.sleep_consolidation import BUDGET, REPLAY_LIMIT, adaptive_bytes, match_random_prune
from nexus_expanse.unified import NexusExpanseUnified

NO_SLEEP = "no_sleep_unified"
RANDOM = "random_prune_control"
SLEEP = "sleep_consolidated_unified"
ARMS = (NO_SLEEP, RANDOM, SLEEP)
PHASES = ("math", "code", "science_prose", "cold_return")
GATE19 = ROOT / "evidence/gate19-po-agency-dev-001.json"


def fixtures(seed):
    source = manifest()["splits"]
    selected = {}
    for split, count in (("train", 32), ("eval", 16)):
        selected[split] = {}
        for index, domain in enumerate(("math", "code")):
            rows = [bytes.fromhex(r["ctx"]) + bytes.fromhex(r["fut"])
                    for r in source[split] if r["domain"] == domain]
            selected[split][domain] = random.Random(seed + index + (100 if split == "eval" else 0)).sample(rows, count)
    prose = [row[start:start + 48] for row in read_split()["dev"] for start in range(0, len(row) - 47, 48)]
    selected["train"]["science_prose"] = random.Random(seed + 200).sample(prose, 24)
    if any(set(selected["train"][d]) & set(selected["eval"][d]) for d in ("math", "code")):
        raise ValueError("cold return contains an exact training block")
    hashed = {s: {d: [hashlib.sha256(b).hexdigest() for b in rows] for d, rows in block.items()}
              for s, block in selected.items()}
    return selected, hashed


def create(seed):
    torch.manual_seed(seed)
    core = NexusExpanseByteCore(n_layers=2, dim=32, n_heads=4, ffn_mult=2, dropout=0.,
                                max_seq_len=64, enable_skip=False)
    return NexusExpanseUnified(core, CompactBigram.from_dense(np.zeros((256, 256), dtype="<u4")),
                              sleep_budget=True)


def mutable_digest(model):
    mixer = model.arbitrator
    raw = (model.episodic.snapshot(max_bytes=BUDGET) + model.reasoning.snapshot()
           + struct.pack("<ddQ", mixer.share, mixer.weight(0.), mixer.updates))
    return hashlib.sha256(raw).hexdigest()


def fingerprint(model):
    model.reset_context()
    deep_calls = model.reader.deep_calls
    held = [model.step(b, adapt=False).mixed.contiguous().numpy().tobytes() for b in b"probe raw\n"]
    model.reset_context()
    model.reader.deep_calls = deep_calls
    return dict(adaptive=mutable_digest(model), weights=state_hash(model.backbone),
                forecast=hashlib.sha256(b"".join(held)).hexdigest())


def collect(model, rows, *, adapt):
    outputs = []
    ledger = {}
    peak = adaptive_bytes(model)
    time_start, cpu_start = time.perf_counter(), time.process_time()
    before = mutable_digest(model)
    for block in rows:
        model.reset_context()
        forecast = model.step(block[0], adapt=adapt)
        loss = []
        for i, byte in enumerate(block[1:], 1):
            context = block[max(0, i - 8):i]
            nll = -math.log2(max(float(forecast.mixed[byte]), 1e-300))
            loss.append(nll)
            forecast = model.step(byte, adapt=adapt)
            if adapt and (i % 6 == 0 or i == len(block) - 1):
                model.observe_relation(context[-2:], byte)
            if adapt and (i <= 3 or i >= len(block) - 3):
                key = context, byte
                ledger[key] = max(nll, ledger.get(key, 0.))
                if len(ledger) > REPLAY_LIMIT:
                    del ledger[min(ledger, key=lambda k: (ledger[k], k))]
            # One length byte, context, outcome and FP64 surprisal per ledger row.
            peak = max(peak, adaptive_bytes(model) + sum(10 + len(ctx) for ctx, _ in ledger))
            if peak > BUDGET:
                raise ValueError("adaptive state and selected boundary ledger exceed 40 KB")
        outputs.append(dict(sha256=hashlib.sha256(block).hexdigest(), transition_bpb=loss))
    model.reset_context()
    if not adapt and mutable_digest(model) != before:
        raise ValueError("cold score changed the adaptive state")
    count = sum(len(row["transition_bpb"]) for row in outputs)
    total = sum(sum(row["transition_bpb"]) for row in outputs)
    return (dict(blocks=outputs, transitions=count, loss_bits=total, bpb=total / count,
                 peak_adaptive_bytes=peak, episodic_bytes=model.episodic.serialized_size(),
                 hopfield_entries=model.reasoning.used, adaptive_bytes=adaptive_bytes(model),
                 seconds=time.perf_counter()-time_start, cpu_seconds=time.process_time()-cpu_start),
            [(ctx, byte, nll) for (ctx, byte), nll in ledger.items()])


def paths(directory, seed, i):
    prefix = directory / f"{seed}-sleep-{i + 1}"
    return prefix.with_suffix(".intent.json"), prefix.with_suffix(".pt"), prefix.with_suffix(".json")


def receipts(directory, ident, seed, fixture):
    history = []
    parent = None
    for i, phase in enumerate(PHASES):
        intent, ck, meta = paths(directory, seed, i)
        exists = [p.exists() for p in (intent, ck, meta)]
        if not any(exists):
            continue
        if not all(exists) or len(history) != i:
            raise ValueError("ambiguous consumed sleep phase, replay refused")
        expected = dict(identity=ident, seed=seed, phase=phase, parent=parent, fixture=fixture)
        record = json.loads(meta.read_text())
        if json.loads(intent.read_text()) != expected or any(record.get(k) != v for k, v in expected.items()):
            raise ValueError("sleep experiment identity drift")
        if digest(ck) != record["checkpoint_sha256"]:
            raise ValueError("sleep checkpoint hash mismatch")
        parent = record["checkpoint_sha256"]
        history.append(record)
    return history


def advance(directory, ident, seed):
    examples, fixture = fixtures(seed)
    with writer(directory, f"{seed}-sleep"):
        history = receipts(directory, ident, seed, fixture)
        i = len(history)
        if i >= len(PHASES):
            return
        phase = PHASES[i]
        previous = history[-1] if history else None
        restores = {}
        if previous:
            raw = torch.load(paths(directory, seed, i - 1)[1], weights_only=True, map_location="cpu")
            models = {a: NexusExpanseUnified.restore(raw[a]) for a in ARMS}
            for a, model in models.items():
                actual = fingerprint(model)
                restores[a] = dict(fresh_process=previous["pid"] != os.getpid(),
                                   fingerprint=actual, equal=actual == previous["fingerprints"][a])
                if not restores[a]["fresh_process"] or not restores[a]["equal"]:
                    raise ValueError("sleep fresh-process parity failed")
        else:
            models = {a: create(seed) for a in ARMS}
        intent, ck, meta = paths(directory, seed, i)
        data = dict(identity=ident, seed=seed, phase=phase,
                    parent=previous["checkpoint_sha256"] if previous else None, fixture=fixture)
        atomic(intent, data)
        outcome = {}
        intervention = None
        if phase == "cold_return":
            for arm, model in models.items():
                domains = {d: collect(model, rows, adapt=False)[0] for d, rows in examples["eval"].items()}
                bits = sum(r["loss_bits"] for r in domains.values())
                n = sum(r["transitions"] for r in domains.values())
                outcome[arm] = dict(domains=domains, bpb=bits / n, transitions=n)
        else:
            replay = None
            for arm, model in models.items():
                outcome[arm], observed = collect(model, examples["train"][phase], adapt=True)
                if arm == SLEEP:
                    replay = observed
            sleep = models[SLEEP].consolidate_sleep(replay)
            matched = match_random_prune(models[SLEEP], models[RANDOM], seed * 10 + i)
            intervention = dict(sleep=sleep, match=matched)
            for arm, model in models.items():
                outcome[arm]["post_boundary_adaptive_bytes"] = adaptive_bytes(model)
        saved = {a: m.snapshot() for a, m in models.items()}
        save_pt(ck, saved)
        state = {a: fingerprint(m) for a, m in models.items()}
        result = dict(data, arms=outcome, intervention=intervention, restores=restores,
                      checkpoint_sha256=digest(ck), fingerprints=state, pid=os.getpid(),
                      peak_rss_bytes=peak_rss())
        atomic(meta, result)
        history.append(result)
        print(json.dumps(dict(seed=seed, phase=phase,
                              metric={a: round(r["bpb"], 6) for a, r in outcome.items()},
                              matched_bytes=intervention["match"]["target_bytes"] if intervention else None)))
        if len(history) == len(PHASES):
            atomic(directory / f"{seed}-sleep.json", dict(seed=seed, phases=history))


def decision(runs):
    checks = []
    for r in runs:
        phases = r["phases"]
        cold = phases[-1]
        scores = cold["arms"]
        margins = {a: scores[a]["bpb"] - scores[SLEEP]["bpb"] for a in (NO_SLEEP, RANDOM)}
        budget = all(phase["arms"][a]["peak_adaptive_bytes"] <= BUDGET
                     and phase["arms"][a]["post_boundary_adaptive_bytes"] <= BUDGET
                     for phase in phases[:-1] for a in ARMS)
        budget = budget and all(record["peak_adaptive_bytes"] <= BUDGET
                                for arm in scores.values() for record in arm["domains"].values())
        byte_match = all(p["arms"][SLEEP]["post_boundary_adaptive_bytes"] ==
                         p["arms"][RANDOM]["post_boundary_adaptive_bytes"] for p in phases[:-1])
        restore = all(x["equal"] and x["fresh_process"] for x in cold["restores"].values())
        checks.append(dict(seed=r["seed"], margins_over_controls=margins, budget=budget,
                           exact_byte_match=byte_match, exact_cold_restore=restore))
    engineering = all(c["budget"] and c["exact_byte_match"] and c["exact_cold_restore"] for c in checks)
    margins = [x for c in checks for x in c["margins_over_controls"].values()]
    status = "FAIL" if not engineering else "PASS" if min(margins) >= .05 else "MIXED" if max(margins) > 0 else "FAIL"
    return dict(status=status, engineering=engineering, checks=checks)


def merge(directory, ident, target):
    runs = []
    for seed in SEEDS:
        _, fixture = fixtures(seed)
        phases = receipts(directory, ident, seed, fixture)
        record = json.loads((directory / f"{seed}-sleep.json").read_text())
        if len(phases) != 4 or record != dict(seed=seed, phases=phases):
            raise ValueError("missing complete sleep trial receipt")
        runs.append(record)
    result = decision(runs)
    atomic(target, dict(schema="nexus-gate20-sleep-dev-v1", qualification="REUSED_SEQUENTIAL_CORPORA_DEV",
                        identity=ident, runs=runs, **result))
    print(json.dumps(result))


def main():
    setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True, type=Path)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--arm", choices=("sleep",), default="sleep", help=argparse.SUPPRESS)
    p.add_argument("--epoch-limit", default=1, type=int, choices=(1,))
    p.add_argument("--merge", type=Path)
    args = p.parse_args()
    dest = prepare_directory(args.directory, ROOT)
    ident = dict(identity(), gate19_sha256=digest(GATE19), prose_manifest_sha256=digest(MANIFEST))
    if args.merge:
        merge(dest, ident, args.merge)
    elif args.seed is not None:
        advance(dest, ident, args.seed)
    else:
        p.error("provide --seed or --merge")


if __name__ == "__main__":
    main()
