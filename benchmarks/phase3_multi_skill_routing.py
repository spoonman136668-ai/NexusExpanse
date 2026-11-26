"""Frozen Phase 3A RAW-byte CPU experiment. Donor training runs in a separate process."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.checkpoint_transfer_study import digest, optim, rules, setup_donor, source_pack, update
from benchmarks.dev_artifacts import atomic
from nexus_expanse.checkpoint_transport import load_skill
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.jepa import state_hash
from nexus_expanse.multi_skill_router import MultiSkillBank, load_bank, save_bank

SOURCE = "105e6e754abe940de5c66ca41a75c3bf825aa1cb"
PREREG = "docs/PHASE3_MULTI_SKILL_ROUTING_PREREG.md"
SEEDS = tuple(range(1501, 1506))
DOMAINS = ("base_python", "rule_A", "rule_B", "markdown_prose")
ROUTERS = ("bayes_bma", "bayes_argmax", "surprise_sparse")
ARMS = ("oracle_manual", *ROUTERS, "uniform_mixture", "base_only")
SWITCHES = ((1, 2), (0, 3), (3, 1))
SWITCH_NAMES = tuple(f"{DOMAINS[a]}_to_{DOMAINS[b]}" for a, b in SWITCHES)
TEXT_PATHS = {
    0: (
        ("nexus_expanse/core.py", "nexus_expanse/unified.py"),
        ("nexus_expanse/verifier_agent.py", "nexus_expanse/deltanet.py"),
    ),
    3: (
        ("README.md", "docs/ARCHITECTURE_V1.md"),
        ("docs/GATE25_PREREG.md", "docs/CHECKPOINT_TRANSFER_PROTOCOL.md"),
    ),
}
FIT_WINDOWS = 96
CHECK_WINDOWS = 8
BYTES = 64
SWITCH_AT = 32
WARMUP = 8
TRAIN_STEPS = 192
BATCH = 8


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def grams(raw):
    return {raw[i:i + 32] for i in range(len(raw) - 31)}


def text_windows(paths, rng):
    chunks = []
    for path in paths:
        data = git("show", f"{SOURCE}:{path}")
        chunks.extend(data[i:i + BYTES] for i in range(0, len(data) - BYTES + 1, BYTES))
    rng.shuffle(chunks)
    return chunks


def rule_windows(table, rng, size):
    return [
        bytes(byte for key in rng.sample(range(32), BYTES // 4)
              for byte in (40 + key % 8, 80 + key // 8, 125, table[key]))
        for _ in range(size)
    ]


def select_windows(candidates, count, used):
    selected = []
    for block in candidates:
        keys = grams(block)
        if keys & used:
            continue
        used.update(keys)
        selected.append(block)
        if len(selected) == count:
            return selected
    raise ValueError("insufficient byte-disjoint windows for frozen split")


def data_for(seed):
    rng = random.Random(seed + 30000)
    tables = [table.tolist() for table in rules(seed)]
    fit, held = {}, {}
    fit_grams = set()
    for k in range(len(DOMAINS)):
        raw = text_windows(TEXT_PATHS[k][0], rng) if k in TEXT_PATHS else rule_windows(tables[k - 1], rng, 256)
        fit[k] = select_windows(raw, FIT_WINDOWS, fit_grams)
    held_grams = fit_grams.copy()
    for k in range(len(DOMAINS)):
        raw = text_windows(TEXT_PATHS[k][1], rng) if k in TEXT_PATHS else rule_windows(tables[k - 1], rng, 256)
        held[k] = select_windows(raw, 3 * CHECK_WINDOWS, held_grams)
    single = [raw for k in range(len(DOMAINS)) for raw in held[k][:CHECK_WINDOWS]]
    switched = []
    for index, (a, b) in enumerate(SWITCHES):
        offset = CHECK_WINDOWS * (2 if index == 2 else 1)
        # Reserve independent Markdown windows for its two switch roles.
        for i in range(offset, offset + CHECK_WINDOWS):
            raw = held[a][i][:SWITCH_AT] + held[b][i][:SWITCH_AT]
            if grams(raw) & fit_grams:
                raise ValueError("switch boundary created an exact 32-byte FIT overlap")
            switched.append(raw)
    suites = {
        "single": dict(raw=torch.tensor([list(raw) for raw in single]),
                       labels=torch.arange(len(DOMAINS)).repeat_interleave(CHECK_WINDOWS)[:, None].expand(-1, BYTES)),
        "switch": dict(raw=torch.tensor([list(raw) for raw in switched]),
                       labels=torch.tensor([[a] * SWITCH_AT + [b] * SWITCH_AT for a, b in SWITCHES])
                       .repeat_interleave(CHECK_WINDOWS, dim=0)),
    }
    manifest = {
        "fit": {DOMAINS[k]: digest(b"".join(v)) for k, v in fit.items()},
        "held_out": {key: digest(value["raw"].to(torch.uint8).numpy().tobytes()) for key, value in suites.items()},
        "fit_check_32byte_overlap": 0,
    }
    return {DOMAINS[k]: torch.tensor([list(block) for block in blocks]) for k, blocks in fit.items()}, suites, manifest


def produce(directory, seed, domains=DOMAINS):
    fit = torch.load(directory / "fit.pt", weights_only=True, map_location="cpu")
    receipt = {}
    for k, name in enumerate(DOMAINS):
        if name not in domains:
            continue
        started = time.perf_counter()
        child_seed = seed * 19 + k
        donor = setup_donor(child_seed)
        opt = optim(donor)
        rng = torch.Generator().manual_seed(child_seed + 40000)
        rows = torch.randperm(256, generator=rng)
        for _ in range(TRAIN_STEPS):
            batch = fit[name][torch.randint(FIT_WINDOWS, (BATCH,), generator=rng)]
            x, y = batch[:, :-1], batch[:, 1:]
            update(donor, opt, donor(rows[x])[..., rows], y)
        path = directory / f"{name}.pt"
        source_pack(path, donor, rows)
        receipt[name] = dict(
            source_sha256=digest(path.read_bytes()), bytes=path.stat().st_size,
            seconds=time.perf_counter() - started, steps=TRAIN_STEPS,
            producer_forward_calls=donor.forward_calls,
        )
        del donor, opt
    atomic(directory / "producer.json", receipt)


def result(bank, raw, labels, arm):
    k = len(bank.skills)
    if arm in ROUTERS:
        logp, route, *counts = getattr(bank, f"forward_{arm}")(raw)
        active = route.argmax(-1) if arm == "bayes_bma" else route
        calls = counts[0] if counts else k * (raw.size(1) - 1) / raw.size(1)
        return logp, active, route, calls
    if arm == "uniform_mixture":
        logp = bank.component_log_probs(raw).logsumexp(-2) - math.log(k)
        return logp, torch.zeros_like(labels), None, k * (raw.size(1) - 1) / raw.size(1)
    active = labels if arm == "oracle_manual" else torch.zeros_like(labels)
    return bank.component_log_probs(raw, selected=active), active, None, (raw.size(1) - 1) / raw.size(1)


def output_digest(logp, route, calls):
    h = hashlib.sha256()
    for value in (logp, route):
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    h.update(float(calls).hex().encode())
    return h.hexdigest()


def bank_identity(bank):
    return {
        "state_sha256": state_hash(bank), "skill_names": list(bank.skill_names),
        "gamma": bank.gamma.hex(), "surprise_threshold_bits": bank.surprise_threshold_bits.hex(),
        "probe_interval": bank.probe_interval,
    }


def restore(directory):
    with forbid_donor_forward() as guard:
        bank = load_bank(directory / "bank.pt")
        suites = torch.load(directory / "suites.pt", weights_only=True, map_location="cpu")
        outputs = {}
        for name, suite in suites.items():
            outputs[name] = {}
            for arm in ROUTERS:
                logp, _, route, calls = result(bank, **suite, arm=arm)
                outputs[name][arm] = output_digest(logp, route, calls)
    print(json.dumps(dict(bank=bank_identity(bank), outputs=outputs, donor_forward_calls=guard.donor_forward_calls)))


def accuracy(match):
    return float(match.double().mean() * 100)


def metrics(logp, active, suite, name):
    raw, labels = suite["raw"], suite["labels"]
    nll = -logp.gather(-1, raw.unsqueeze(-1)).squeeze(-1).double() / math.log(2)
    match = logp.argmax(-1) == raw
    routing = active == labels
    if name == "single":
        rule_rows = (labels[:, 0] == 1) | (labels[:, 0] == 2)
        return dict(
            rule_accuracy_pct=accuracy(match[rule_rows, 3::4]),
            rule_a_accuracy_pct=accuracy(match[labels[:, 0] == 1, 3::4]),
            rule_b_accuracy_pct=accuracy(match[labels[:, 0] == 2, 3::4]),
            python_bpb=float(nll[labels[:, 0] == 0].mean()),
            markdown_bpb=float(nll[labels[:, 0] == 3].mean()),
            overall_bpb=float(nll.mean()), routing_accuracy_pct=accuracy(routing[:, WARMUP:]),
            domain_routing_accuracy_pct={
                domain: accuracy(routing[labels[:, 0] == k, WARMUP:])
                for k, domain in enumerate(DOMAINS)
            },
        )
    transitions = {}
    for i, transition in enumerate(SWITCH_NAMES):
        rows = slice(i * CHECK_WINDOWS, (i + 1) * CHECK_WINDOWS)
        after = routing[rows, SWITCH_AT:]
        latency = next((offset for offset in range(after.size(1) - WARMUP + 1)
                        if accuracy(after[:, offset:offset + WARMUP]) >= 80), None)
        transitions[transition] = dict(
            latency_bytes=latency, routing_accuracy_pct=accuracy(after),
            prediction_accuracy_pct=accuracy(match[rows, SWITCH_AT:]),
            bpb=float(nll[rows, SWITCH_AT:].mean()),
        )
    latencies = [entry["latency_bytes"] for entry in transitions.values()]
    return dict(
        post_switch_bpb=float(nll[:, SWITCH_AT:].mean()),
        post_switch_prediction_accuracy_pct=accuracy(match[:, SWITCH_AT:]),
        post_switch_routing_accuracy_pct=accuracy(routing[:, SWITCH_AT:]),
        worst_latency_bytes=None if None in latencies else max(latencies),
        transitions=transitions,
    )


def run_seed(seed):
    fit, suites, manifest = data_for(seed)
    with tempfile.TemporaryDirectory(prefix="nexus-phase3-") as tmp:
        directory = Path(tmp)
        torch.save(fit, directory / "fit.pt")
        subprocess.run([sys.executable, __file__, "--produce", tmp, "--seed", str(seed)], check=True, cwd=ROOT)
        producer = json.loads((directory / "producer.json").read_text())
        for name, item in producer.items():
            if digest((directory / f"{name}.pt").read_bytes()) != item["source_sha256"]:
                raise ValueError(f"{name} donor checkpoint custody mismatch")
        with forbid_donor_forward() as guard:
            started = time.perf_counter()
            skills = {name: load_skill(directory / f"{name}.pt", method="residual8") for name in DOMAINS}
            bank = MultiSkillBank(skills.pop(DOMAINS[0]), skills)
            import_seconds = time.perf_counter() - started
            base_before = state_hash(bank.base)
            rows = {}
            outputs = {}
            for name, suite in suites.items():
                rows[name] = {}
                outputs[name] = {}
                for arm in ARMS:
                    started = time.perf_counter()
                    logp, active, routing, count = result(bank, **suite, arm=arm)
                    elapsed = time.perf_counter() - started
                    rows[name][arm] = metrics(logp, active, suite, name)
                    rows[name][arm].update(
                        avg_active_skills_per_byte=count,
                        wall_ms_per_byte=1000 * elapsed / suite["raw"].numel(),
                    )
                    if arm in ROUTERS:
                        outputs[name][arm] = output_digest(logp, routing, count)
            base_after = state_hash(bank.base)
            save_bank(bank, directory / "bank.pt")
            bank_sha256 = digest((directory / "bank.pt").read_bytes())
        torch.save(suites, directory / "suites.pt")
        cold = json.loads(subprocess.check_output([sys.executable, __file__, "--cold", tmp], cwd=ROOT))
    identity = bank_identity(bank)
    return dict(
        seed=seed, corpus_manifest=manifest, producer=producer, import_seconds=import_seconds,
        donor_forward_calls=guard.donor_forward_calls + cold["donor_forward_calls"],
        base_state_sha256=base_before, base_preserved=base_before == base_after,
        bank_sha256=bank_sha256, bank=identity,
        cold_restore_match=identity == cold["bank"] and outputs == cold["outputs"],
        output_digests=outputs, cold_output_digests=cold["outputs"], **rows,
    )


def compare_controls(row, arm):
    rules_and_bpb = row["single"][arm]
    return all(
        rules_and_bpb["rule_accuracy_pct"] > row["single"][control]["rule_accuracy_pct"]
        and rules_and_bpb["overall_bpb"] < row["single"][control]["overall_bpb"]
        for control in ("uniform_mixture", "base_only")
    )


def science(rows):
    dense = {}
    for arm in ROUTERS[:2]:
        dense[arm] = dict(
            beats_both=sum(compare_controls(row, arm) for row in rows),
            routing_at_least_90=sum(row["single"][arm]["routing_accuracy_pct"] >= 90 for row in rows),
            recovers_within_8=sum(row["switch"][arm]["worst_latency_bytes"] is not None
                                  and row["switch"][arm]["worst_latency_bytes"] <= 8 for row in rows),
        )
    sparse_wins = sum(compare_controls(row, "surprise_sparse") for row in rows)
    sparse_saves = sum(
        all(row[suite]["surprise_sparse"]["avg_active_skills_per_byte"] <= 0.6 * len(DOMAINS)
            for suite in ("single", "switch"))
        for row in rows
    )
    passed = sparse_wins == sparse_saves == len(SEEDS) and any(
        c["beats_both"] == c["routing_at_least_90"] == len(SEEDS) and c["recovers_within_8"] >= 4
        for c in dense.values()
    )
    verdict = "PASS" if passed else "MIXED" if any(c["beats_both"] >= 4 for c in dense.values()) else "FAIL"
    return dict(verdict=verdict, dense=dense, sparse_beats_both=sparse_wins, sparse_saves_40_pct=sparse_saves)


def run(destination):
    if destination.exists():
        raise FileExistsError(destination)
    if git("status", "--porcelain") or git("rev-parse", "HEAD^").decode().strip() != SOURCE:
        raise ValueError("execute on the clean Phase 3A preregistration commit")
    source = git("rev-parse", "HEAD").decode().strip()
    started = time.perf_counter()
    rows = []
    for seed in SEEDS:
        row = run_seed(seed)
        rows.append(row)
        print(f"seed {seed} complete, cold_restore={row['cold_restore_match']}", flush=True)
    report = dict(
        schema="nexus-phase3-multi-skill-routing-dev-v1", source_sha=source,
        source_tree=git("rev-parse", "HEAD^{tree}").decode().strip(),
        prereg_sha256=digest((ROOT / PREREG).read_bytes()), fixture_source_sha=SOURCE,
        platform=platform.platform(), python=platform.python_version(),
        torch=str(torch.__version__), numpy=np.__version__, torch_threads=torch.get_num_threads(),
        elapsed_seconds=time.perf_counter() - started,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        receiver_checks=dict(
            forbidden_calls=sum(row["donor_forward_calls"] for row in rows),
            cold_restore_matches=sum(row["cold_restore_match"] for row in rows),
            frozen_base_matches=sum(row["base_preserved"] for row in rows),
            seeds=len(SEEDS),
        ),
        scientific=science(rows), rows=rows,
    )
    atomic(destination, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path)
    mode.add_argument("--produce", type=Path, help=argparse.SUPPRESS)
    mode.add_argument("--cold", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.produce:
        if args.seed is None:
            parser.error("--produce requires --seed")
        produce(args.produce, args.seed)
    elif args.cold:
        restore(args.cold)
    else:
        run(args.out)


if __name__ == "__main__":
    main()
