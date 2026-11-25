"""Preregistered Sticky-HMM raw-byte routing confirmation on fresh Phase 3A fixtures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import phase3_multi_skill_routing as phase3
from benchmarks.dev_artifacts import atomic, digest
from nexus_expanse.checkpoint_transport import load_skill
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.jepa import state_hash
from nexus_expanse.multi_skill_router import MultiSkillBank, load_bank, save_bank

BASE = "cc1a462efce81ffb01ec42b692d7657af2dcd214"
PREREG = "docs/PHASE3C_STICKY_HMM_ROUTER_PREREG.md"
SEEDS = tuple(range(1601, 1606))
HMM_ARMS = ("hmm_bma", "hmm_sparse")
ARMS = ("oracle_manual", *HMM_ARMS, "bayes_bma", "surprise_sparse", "uniform_mixture")


def outcome(bank, raw, labels, arm):
    if arm not in HMM_ARMS:
        return phase3.result(bank, raw, labels, arm)
    logp, weights, *measured = getattr(bank, f"forward_{arm}")(raw)
    calls = measured[0] if measured else len(bank.skills) * (raw.size(1) - 1) / raw.size(1)
    return logp, weights.argmax(-1), weights, calls


def measures(logp, active, suite, name):
    scores = phase3.metrics(logp, active, suite, name)
    if name == "switch":
        raw, labels = suite["raw"], suite["labels"]
        rule = (labels == 1) | (labels == 2)
        positions = torch.arange(raw.size(1))
        answer = rule & (positions >= phase3.SWITCH_AT) & (positions % 4 == 3)
        match = logp.argmax(-1) == raw
        scores["post_switch_rule_accuracy_pct"] = phase3.accuracy(match[answer])
        for i, transition in enumerate(phase3.SWITCH_NAMES):
            rows = slice(i * phase3.CHECK_WINDOWS, (i + 1) * phase3.CHECK_WINDOWS)
            mask = answer[rows]
            scores["transitions"][transition]["rule_accuracy_pct"] = (
                phase3.accuracy(match[rows][mask]) if mask.any() else None
            )
    return scores


def cold_restore(directory):
    with forbid_donor_forward() as guard:
        bank = load_bank(directory / "bank.pt")
        suites = torch.load(directory / "suites.pt", weights_only=True, map_location="cpu")
        outputs = {}
        for name, suite in suites.items():
            outputs[name] = {}
            for arm in HMM_ARMS:
                logp, _, weights, calls = outcome(bank, **suite, arm=arm)
                outputs[name][arm] = phase3.output_digest(logp, weights, calls)
    return dict(bank=phase3.bank_identity(bank), outputs=outputs, forbidden_calls=guard.donor_forward_calls)


def run_seed(seed):
    fit, suites, manifest = phase3.data_for(seed)
    with tempfile.TemporaryDirectory(prefix="nexus-phase3c-") as temp:
        directory = Path(temp)
        torch.save(fit, directory / "fit.pt")
        subprocess.run([sys.executable, __file__, "--produce", temp, "--seed", str(seed)], check=True, cwd=ROOT)
        producer = json.loads((directory / "producer.json").read_text())
        for name, record in producer.items():
            if digest(directory / f"{name}.pt") != record["source_sha256"]:
                raise ValueError(f"donor checkpoint custody mismatch for {name}")

        with forbid_donor_forward() as guard:
            started = time.perf_counter()
            skills = {name: load_skill(directory / f"{name}.pt", "residual8") for name in phase3.DOMAINS}
            bank = MultiSkillBank(skills.pop(phase3.DOMAINS[0]), skills)
            import_seconds = time.perf_counter() - started
            base_before = state_hash(bank.base)
            scores, outputs = {}, {}
            for name, suite in suites.items():
                scores[name], outputs[name] = {}, {}
                for arm in ARMS:
                    started = time.perf_counter()
                    logp, active, routing, calls = outcome(bank, **suite, arm=arm)
                    milliseconds = 1000 * (time.perf_counter() - started) / suite["raw"].numel()
                    scores[name][arm] = measures(logp, active, suite, name)
                    scores[name][arm].update(avg_active_skills_per_byte=calls, wall_ms_per_byte=milliseconds)
                    if arm in HMM_ARMS:
                        outputs[name][arm] = phase3.output_digest(logp, routing, calls)
            base_preserved = base_before == state_hash(bank.base)
            save_bank(bank, directory / "bank.pt")
            snapshot_digest = digest(directory / "bank.pt")
        torch.save(suites, directory / "suites.pt")
        restored = json.loads(subprocess.check_output([sys.executable, __file__, "--cold", temp], cwd=ROOT))

    identity = phase3.bank_identity(bank)
    return dict(
        seed=seed, corpus_manifest=manifest, producer=producer, import_seconds=import_seconds,
        bank=identity, bank_sha256=snapshot_digest, base_state_sha256=base_before, base_preserved=base_preserved,
        forbidden_calls=guard.donor_forward_calls + restored["forbidden_calls"],
        cold_restore_match=identity == restored["bank"] and outputs == restored["outputs"],
        output_digests=outputs, cold_output_digests=restored["outputs"], **scores,
    )


def scientific_verdict(rows):
    dense_wins = recovery = sparse_savings = sparse_baseline = sparse_uniform = 0
    single_routing = {arm: 0 for arm in HMM_ARMS}
    sparse_recovery = 0
    for row in rows:
        switch = row["switch"]
        dense, discounted = switch["hmm_bma"], switch["bayes_bma"]
        dense_wins += (dense["post_switch_bpb"] < discounted["post_switch_bpb"]
                       and dense["post_switch_routing_accuracy_pct"] > discounted["post_switch_routing_accuracy_pct"])
        latency = dense["worst_latency_bytes"]
        recovery += latency is not None and latency <= 8
        sparse = switch["hmm_sparse"]
        sparse_latency = sparse["worst_latency_bytes"]
        sparse_recovery += sparse_latency is not None and sparse_latency <= 8
        sparse_savings += all(
            row[suite]["hmm_sparse"]["avg_active_skills_per_byte"] <= 0.6 * len(phase3.DOMAINS)
            for suite in ("single", "switch")
        )
        sparse_baseline += sparse["post_switch_routing_accuracy_pct"] > switch["surprise_sparse"]["post_switch_routing_accuracy_pct"]
        sparse_uniform += sparse["post_switch_routing_accuracy_pct"] > switch["uniform_mixture"]["post_switch_routing_accuracy_pct"]
        for arm in HMM_ARMS:
            single_routing[arm] += row["single"][arm]["routing_accuracy_pct"] >= 90
    count = len(SEEDS)
    passed = (dense_wins == count and recovery >= 4 and sparse_savings == count
              and sparse_baseline >= 4 and sparse_uniform == count)
    verdict = "PASS" if passed else "MIXED" if dense_wins >= 4 else "FAIL"
    return dict(
        verdict=verdict, dense_beats_bayes_on_post_switch_bpb_and_routing=dense_wins,
        dense_all_transitions_recover_within_8_bytes=recovery,
        sparse_all_transitions_recover_within_8_bytes=sparse_recovery,
        sparse_reduces_calls_by_40_pct_on_both_suites=sparse_savings,
        sparse_beats_surprise_sparse_on_post_switch_routing=sparse_baseline,
        sparse_beats_uniform_on_post_switch_routing=sparse_uniform,
        single_domain_routing_at_least_90_pct=single_routing,
    )


def run(destination):
    if destination.exists():
        raise FileExistsError(destination)
    source = phase3.git("rev-parse", "HEAD").decode().strip()
    tree = phase3.git("rev-parse", "HEAD^{tree}").decode().strip()
    if phase3.git("status", "--porcelain") or phase3.git("rev-parse", "HEAD^").decode().strip() != BASE:
        raise ValueError("execute on the clean Phase 3C preregistration commit")
    started = time.perf_counter()
    rows = []
    for seed in SEEDS:
        row = run_seed(seed)
        rows.append(row)
        print(f"seed {seed} complete, cold_restore={row['cold_restore_match']}", flush=True)
    if phase3.git("status", "--porcelain") or phase3.git("rev-parse", "HEAD").decode().strip() != source:
        raise ValueError("preregistered checkout changed during scoring")
    atomic(destination, dict(
        schema="nexus-phase3c-sticky-hmm-routing-dev-v1", source_sha=source, source_tree=tree,
        base_sha=BASE, fixture_source_sha=phase3.SOURCE, prereg_sha256=digest(ROOT / PREREG),
        platform=platform.platform(), python=platform.python_version(), torch=str(torch.__version__),
        numpy=np.__version__, torch_threads=torch.get_num_threads(),
        elapsed_seconds=time.perf_counter() - started,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        receiver_checks=dict(
            forbidden_calls=sum(row["forbidden_calls"] for row in rows),
            cold_restore_matches=sum(row["cold_restore_match"] for row in rows),
            frozen_base_matches=sum(row["base_preserved"] for row in rows), seeds=len(SEEDS),
        ),
        scientific=scientific_verdict(rows), rows=rows,
    ))


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
        phase3.produce(args.produce, args.seed)
    elif args.cold:
        print(json.dumps(cold_restore(args.cold)))
    else:
        run(args.out)


if __name__ == "__main__":
    main()
