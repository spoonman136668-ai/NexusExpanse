"""Preregistered online skill birth and cold-retention DEV confirmation."""
from __future__ import annotations

import argparse
import copy
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

from benchmarks import phase3_multi_skill_routing as phase3
from benchmarks.dev_artifacts import atomic, digest
from nexus_expanse.checkpoint_transport import load_skill
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.jepa import state_hash
from nexus_expanse.multi_skill_router import MultiSkillBank, load_bank, save_bank
from nexus_expanse.skill_birth import ADAPT_STEPS, NOVELTY_THRESHOLD_BPB, NOVELTY_WINDOW, OpenWorldSkillBank, adapt_skill

BASE = "76ab0e94f814a79fcbf4e78701772197efc630ed"
PREREG = "docs/PHASE4_SKILL_BIRTH_PREREG.md"
SEEDS = tuple(range(1701, 1706))
NOVEL = "rule_C"
DOMAINS = (*phase3.DOMAINS, NOVEL)
FIRST_PROBE_BLOCKS = 4
MAX_ENCOUNTER_BLOCKS = 8
RETENTION_LIMIT_BPB = 0.02
ARMS = ("open_world_birth", "fixed_k4_bank", "continual_finetune_base", "oracle_5_skill")


def rule_c_data(seed, original_fit, original_check):
    rng = random.Random(seed + 47000)
    table = [224 + value for _ in range(4) for value in rng.sample(range(8), 8)]
    seen = set()
    for data in (*original_fit.values(), original_check):
        for row in data.tolist():
            seen.update(phase3.grams(bytes(row)))

    candidates = []
    for _ in range(128):
        keys = rng.sample(range(32), phase3.BYTES // 4)
        candidates.append(bytes(byte for key in keys
                                for byte in (176 + key % 8, 192 + key // 8, 208, table[key])))
    encounter = phase3.select_windows(candidates[:64], MAX_ENCOUNTER_BLOCKS, seen)
    check = phase3.select_windows(candidates[64:], phase3.CHECK_WINDOWS, seen)
    encounter_grams = phase3.grams(b"".join(encounter))
    if any(phase3.grams(block) & encounter_grams for block in check):
        raise ValueError("Rule C overlap at concatenated encounter boundaries")
    manifest = dict(
        encounter_sha256=phase3.digest(b"".join(encounter)),
        check_sha256=phase3.digest(b"".join(check)), fit_check_32byte_overlap=0,
    )
    return torch.tensor([list(row) for row in encounter]), torch.tensor([list(row) for row in check]), manifest


def skill_hashes(bank):
    return {name: state_hash(skill) for name, skill in bank.skills.items()}


def route(bank, raw, labels, *, oracle=False):
    if oracle:
        return bank.component_log_probs(raw, selected=labels), labels
    logp, weights = bank.forward_hmm_bma(raw)
    return logp, weights


def summarize(raw, labels, logp, routing):
    active = routing.argmax(-1) if routing.ndim == 3 else routing
    nll = -logp.gather(-1, raw.unsqueeze(-1)).squeeze(-1).double() / math.log(2)
    matches = logp.argmax(-1) == raw
    results = {}
    for k, name in enumerate(DOMAINS):
        rows = labels[:, 0] == k
        result = dict(bpb=float(nll[rows].mean()), routing_accuracy_pct=phase3.accuracy(active[rows] == k))
        if k in (1, 2, 4):
            result["rule_accuracy_pct"] = phase3.accuracy(matches[rows, 3::4])
        results[name] = result
    return dict(original_bpb=float(nll[labels != 4].mean()), domains=results)


def restore(directory):
    suite = torch.load(directory / "test.pt", weights_only=True, map_location="cpu")
    with forbid_donor_forward() as guard:
        born = load_bank(directory / "born.pt")
        oracle_path = directory / "oracle.pt"
        oracle = load_bank(oracle_path) if oracle_path.exists() else born
        logp, weights = route(born, **suite)
        digest_before = phase3.output_digest(logp, weights, 0)
        open_world = summarize(**suite, logp=logp, routing=weights)
        oracle_logp, oracle_routes = route(oracle, **suite, oracle=True)
        manual = summarize(**suite, logp=oracle_logp, routing=oracle_routes)
    return dict(
        open_world_birth=open_world, oracle_5_skill=manual,
        born_identity=phase3.bank_identity(born), oracle_identity=phase3.bank_identity(oracle),
        output_sha256=digest_before,
        oracle_output_sha256=phase3.output_digest(oracle_logp, oracle_routes, 0),
        donor_forward_calls=guard.donor_forward_calls,
    )


def run_seed(seed):
    started = time.perf_counter()
    original_fit, suites, manifest = phase3.data_for(seed)
    original = suites["single"]["raw"]
    encounter, novel, novel_manifest = rule_c_data(seed, original_fit, original)
    raw = torch.cat([original, novel])
    labels = torch.arange(len(DOMAINS)).repeat_interleave(phase3.CHECK_WINDOWS)[:, None].expand_as(raw)
    suite = dict(raw=raw, labels=labels)
    with tempfile.TemporaryDirectory(prefix="nexus-phase4-") as temp:
        directory = Path(temp)
        torch.save(original_fit, directory / "fit.pt")
        subprocess.run([sys.executable, __file__, "--produce", temp, "--seed", str(seed)], check=True, cwd=ROOT)
        producer = json.loads((directory / "producer.json").read_text())
        for name, receipt in producer.items():
            if digest(directory / f"{name}.pt") != receipt["source_sha256"]:
                raise ValueError(f"donor checkpoint custody mismatch for {name}")

        with forbid_donor_forward() as guard:
            skills = {name: load_skill(directory / f"{name}.pt", "residual8") for name in phase3.DOMAINS}
            bank = MultiSkillBank(skills.pop(phase3.DOMAINS[0]), skills)
            world = OpenWorldSkillBank(bank)
            resident_before = skill_hashes(bank)
            component_before = bank.component_log_probs(original)
            fixed_logp, fixed_routing = route(bank, **suite)
            fixed = summarize(**suite, logp=fixed_logp, routing=fixed_routing)

            negatives = original.reshape(-1, 2 * phase3.BYTES)
            false_flags = world.detect_novelty(negatives).view(len(phase3.DOMAINS), -1)
            false_births = {name: int(flags.sum()) for name, flags in zip(phase3.DOMAINS, false_flags, strict=True)}
            finetune_bank = copy.deepcopy(bank)
            birth_at = None
            probes = []
            adaptation_seconds = 0.0
            for blocks in range(FIRST_PROBE_BLOCKS, MAX_ENCOUNTER_BLOCKS + 1):
                observed = encounter[:blocks].reshape(1, -1)
                start = time.perf_counter()
                born = world.observe(NOVEL, observed)
                probes.append(dict(observed_bytes=observed.numel(), triggered=born is not None))
                if born is not None:
                    birth_at = observed.numel()
                    adaptation_seconds = time.perf_counter() - start
                    break
            resident_after = {name: state_hash(bank.skills[name]) for name in phase3.DOMAINS}
            component_after = bank.component_log_probs(original)[:, :, :len(phase3.DOMAINS)]
            resident_predictions_exact = torch.equal(component_before, component_after)

            control_start = time.perf_counter()
            adapt_skill(finetune_bank.base, observed, ADAPT_STEPS)
            control_seconds = time.perf_counter() - control_start
            ft_logp, ft_routing = route(finetune_bank, **suite)
            continual = summarize(**suite, logp=ft_logp, routing=ft_routing)

            oracle = bank
            if birth_at is None:
                oracle_world = OpenWorldSkillBank(copy.deepcopy(bank))
                oracle_world.spawn_and_adapt_skill(NOVEL, observed)
                oracle = oracle_world.bank
                save_bank(oracle, directory / "oracle.pt")
            expected = phase3.output_digest(*route(bank, **suite), 0)
            oracle_expected = phase3.output_digest(*route(oracle, **suite, oracle=True), 0)
            world.save_bank(directory / "born.pt")
            snapshot_sha256 = digest(directory / "born.pt")

        torch.save(suite, directory / "test.pt")
        cold = json.loads(subprocess.check_output([sys.executable, __file__, "--cold", temp], cwd=ROOT))

    cold_match = cold["born_identity"] == phase3.bank_identity(bank) and cold["output_sha256"] == expected
    oracle_cold = cold["oracle_identity"] == phase3.bank_identity(oracle) and cold["oracle_output_sha256"] == oracle_expected
    open_world = cold["open_world_birth"]
    drift = {name: open_world["domains"][name]["bpb"] - fixed["domains"][name]["bpb"] for name in phase3.DOMAINS}
    return dict(
        seed=seed, corpus_manifest=manifest, novel_manifest=novel_manifest, producer=producer,
        probes=probes, birth_at_byte=birth_at, adaptation_bytes=observed.numel(),
        adaptation_sha256=phase3.digest(bytes(observed[0].tolist())),
        open_world_probe_and_adaptation_seconds=adaptation_seconds,
        finetune_adaptation_seconds=control_seconds, steps=ADAPT_STEPS, false_births=false_births,
        old_state_hashes_before=resident_before, old_state_hashes_after=resident_after,
        resident_weights_exact=resident_before == resident_after, resident_predictions_exact=resident_predictions_exact,
        cold_restore_exact=cold_match, expanded_cold_restore_exact=oracle_cold and len(oracle.skills) == 5,
        restored_skill_names=cold["born_identity"]["skill_names"],
        open_world_output_sha256=expected, cold_output_sha256=cold["output_sha256"],
        snapshot_sha256=snapshot_sha256, forbidden_calls=guard.donor_forward_calls + cold["donor_forward_calls"],
        retention_drift_bpb=drift, elapsed_seconds=time.perf_counter() - started,
        open_world_birth=open_world, fixed_k4_bank=fixed, continual_finetune_base=continual,
        oracle_5_skill=cold["oracle_5_skill"],
    )


def verdict(rows):
    count = len(SEEDS)
    births = wins = retention = joint = routed = forgetting = 0
    for row in rows:
        open_world, fixed, continual, _ = (row[arm] for arm in ARMS)
        candidate, control = (arm["domains"][NOVEL] for arm in (open_world, fixed))
        triggered = row["birth_at_byte"] is not None
        wins_fixed = candidate["bpb"] < control["bpb"] and candidate["rule_accuracy_pct"] > control["rule_accuracy_pct"]
        retained = max(row["retention_drift_bpb"].values()) <= RETENTION_LIMIT_BPB
        beats_continual = open_world["original_bpb"] < continual["original_bpb"]
        births += triggered
        wins += wins_fixed
        retention += retained and beats_continual
        forgetting += not retained
        joint += triggered and wins_fixed and retained and beats_continual
        routed += triggered and candidate["routing_accuracy_pct"] >= 80
    status = "PASS" if joint == count and routed >= 4 else "MIXED" if joint >= 4 and not forgetting else "FAIL"
    return dict(
        verdict=status, novelty_births=births, beats_fixed_k4_on_c_bpb_and_rule_accuracy=wins,
        beats_continual_and_within_retention_drift=retention, joint_prediction_and_retention_wins=joint,
        cold_rule_c_routing_at_least_80_pct=routed, seeds_exceeding_retention_drift=forgetting,
    )


def run(destination):
    if destination.exists():
        raise FileExistsError(destination)
    source = phase3.git("rev-parse", "HEAD").decode().strip()
    tree = phase3.git("rev-parse", "HEAD^{tree}").decode().strip()
    if phase3.git("status", "--porcelain") or phase3.git("rev-parse", "HEAD^").decode().strip() != BASE:
        raise ValueError("execute on the clean Phase 4 preregistration commit")
    started = time.perf_counter()
    rows = []
    for seed in SEEDS:
        row = run_seed(seed)
        rows.append(row)
        print(f"seed {seed} complete, birth_at={row['birth_at_byte']}, cold={row['cold_restore_exact']}", flush=True)
    if phase3.git("status", "--porcelain") or phase3.git("rev-parse", "HEAD").decode().strip() != source:
        raise ValueError("preregistered checkout changed during scoring")
    atomic(destination, dict(
        schema="nexus-phase4-skill-birth-dev-v1", source_sha=source, source_tree=tree,
        base_sha=BASE, fixture_source_sha=phase3.SOURCE, prereg_sha256=digest(ROOT / PREREG),
        platform=platform.platform(), python=platform.python_version(), torch=str(torch.__version__),
        numpy=np.__version__, torch_threads=torch.get_num_threads(),
        elapsed_seconds=time.perf_counter() - started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        protocol=dict(threshold_bpb=NOVELTY_THRESHOLD_BPB, novelty_window=NOVELTY_WINDOW,
                      adaptation_steps=ADAPT_STEPS, retention_drift_bpb=RETENTION_LIMIT_BPB),
        receiver_checks=dict(
            seeds=len(rows), forbidden_calls=sum(row["forbidden_calls"] for row in rows),
            false_births=sum(sum(row["false_births"].values()) for row in rows),
            in_distribution_probes=len(rows) * len(phase3.DOMAINS) * (phase3.CHECK_WINDOWS // 2),
            exact_resident_preservation=sum(row["resident_weights_exact"] and row["resident_predictions_exact"] for row in rows),
            exact_cold_restores=sum(row["cold_restore_exact"] and row["expanded_cold_restore_exact"] for row in rows),
        ),
        scientific=verdict(rows), rows=rows,
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
        print(json.dumps(restore(args.cold)))
    else:
        run(args.out)


if __name__ == "__main__":
    main()
