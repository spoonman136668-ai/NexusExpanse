"""Frozen confidence-gated routing and per-skill novelty confirmation."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
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
from benchmarks import phase4_skill_birth as phase4
from benchmarks import phase6_hierarchical_scaling as phase6
from benchmarks import phase7_hard_novelty_sparse_k7 as phase7
from benchmarks.dev_artifacts import atomic, digest
from nexus_expanse.checkpoint_transport import load_skill
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.jepa import state_hash
from nexus_expanse.multi_skill_router import MultiSkillBank
from nexus_expanse.skill_birth import BIRTH_STEPS, EPISODIC_BIRTH, OpenWorldSkillBank, PerSkillNoveltyDetector, adapt_skill

BASE = "50e135d27f9390f7b911370c13931a7a743116a8"
PREREG = "docs/PHASE8_PER_SKILL_NOVELTY_PREREG.md"
SEEDS = tuple(range(2101, 2106))
SPARSE, DENSE, FIXED = "conf_sparse_k7", "dense_k7", phase7.FIXED
PER_SKILL, GLOBAL, CONTINUAL = "per_skill_novelty_birth", "global_mean_birth", phase7.CONTINUAL
NOVEL = phase7.NOVEL


def validation_a(seed, val_b, fit, original, exposures, checks):
    """Use independent ID validation windows for all seven native-byte skills."""
    val = {name: phase7.ASCII_PERMUTATION[raw] if name in ("rule_A", "rule_B") else raw
           for name, raw in val_b.items()}
    used = set().union(*(phase7.window_grams(raw) for raw in fit.values()))
    for raw in exposures.values():
        used.update(phase3.grams(phase7.bytes_for(raw)))
    for name, raw in val.items():
        if phase7.window_grams(raw) & used:
            raise ValueError(f"Suite A resident validation overlaps FIT or encounter for {name}")
        used.update(phase7.window_grams(raw))
    for index, (name, (a, b, delimiter, target)) in enumerate(phase6.SPECS.items()):
        mapping_rng = random.Random(seed + 47000 + index * 1000)
        table = [target + value for _ in range(4) for value in mapping_rng.sample(range(8), 8)]
        val_rng = random.Random(seed + 68100 + index)
        candidates = []
        for _ in range(384):
            keys = val_rng.sample(range(32), phase3.BYTES // 4)
            candidates.append(bytes(byte for key in keys
                                    for byte in (a + key % 8, b + key // 8, delimiter, table[key])))
        selected = phase3.select_windows(candidates, phase7.VAL_WINDOWS, used)
        val[name] = torch.tensor([list(block) for block in selected]).reshape(-1, 2 * phase3.BYTES)
        used.update(phase7.window_grams(val[name]))

    check_grams = set().union(*(phase7.window_grams(raw) for raw in (original, *checks.values())))
    if used & check_grams:
        raise ValueError("Suite A CHECK overlaps FIT, encounter or VAL")
    return val


def identity(bank):
    return dict(**phase6.identity(bank), byte_nll_q99=bank.byte_nll_q99)


def snapshot(world, path):
    world.save_bank(path)
    return dict(identity=identity(world.bank), sha256=digest(path), bytes=path.stat().st_size)


def evaluate_a(bank, suite, arm):
    if arm == SPARSE:
        logp, weights, calls = bank.forward_hmm_conf_sparse_hierarchical(suite["raw"])
        scores = phase6.summarize(bank, suite, logp, weights)
        return dict(**scores, avg_active_skills_per_byte=calls), phase3.output_digest(logp, weights, calls)
    return phase7.evaluate_a(bank, suite, phase7.DENSE if arm == DENSE else FIXED)


def evaluate_b(bank, suite, arm):
    return phase7.evaluate_b(bank, suite, phase7.TAIL if arm in (PER_SKILL, GLOBAL) else arm)


def run_warm_a(bank, data, original, encounters, val, directory):
    world = OpenWorldSkillBank(bank)
    initial = phase4.skill_hashes(bank)
    before = bank.component_log_probs(original)
    fixed = evaluate_a(bank, data, FIXED)[0]
    negative = phase7.check_negatives(world, original)
    stages, _ = phase6.grow(world, encounters)
    hashes = {name: state_hash(bank.skills[name]) for name in initial}
    immutable = initial == hashes and torch.equal(before, bank.component_log_probs(original)[:, :, :len(initial)])
    for stage in stages:
        if stage["born"]:
            immutable = immutable and state_hash(bank.skills[stage["domain"]]) == stage["born_state_sha256"]
    limits = bank.calibrate_byte_nll_q99({name: val[name] for name in bank.skill_names})
    outputs = {arm: evaluate_a(bank, data, arm)[1] for arm in (SPARSE, DENSE)}
    seal = snapshot(world, directory / "a.pt")
    return dict(fixed=fixed, stages=stages, negative=negative, resident_exact=immutable, outputs=outputs,
                snapshot=seal, bank_size=len(bank.skills), byte_nll_q99=limits,
                val_sha256={name: phase3.digest(phase7.bytes_for(raw)) for name, raw in val.items()},
                val_fit_check_32byte_overlap=0)


def assess(detector, bank, raw):
    flags, bpb, dominant, ceiling = detector.assess(bank, raw)
    rows = [
        dict(flag=bool(flag), bpb=float(bits), dominant=detector.skill_names[int(slot)], ceiling_bpb=float(limit))
        for flag, bits, slot, limit in zip(flags, bpb, dominant, ceiling, strict=True)
    ]
    return flags, rows


def run_warm_b(bank, data, original, val, encounter, directory):
    world = OpenWorldSkillBank(bank)
    detector = PerSkillNoveltyDetector.calibrate(bank, val)
    old = original.reshape(-1, 2 * phase3.BYTES)
    _, known = assess(detector, bank, old)
    per_domain = phase3.CHECK_WINDOWS // 2
    negatives = {name: sum(row["flag"] for row in known[i * per_domain:(i + 1) * per_domain])
                 for i, name in enumerate(phase3.DOMAINS)}
    fixed = evaluate_b(bank, data, FIXED)[0]
    initial = phase4.skill_hashes(bank)
    arms, observed = {}, {}
    for arm in (PER_SKILL, GLOBAL):
        candidate = OpenWorldSkillBank(copy.deepcopy(bank))
        probes = []
        started = time.perf_counter()
        for blocks in range(phase4.FIRST_PROBE_BLOCKS, phase4.MAX_ENCOUNTER_BLOCKS + 1):
            raw = encounter[:blocks].reshape(1, -1)
            if arm == PER_SKILL:
                flags, scores = assess(detector, candidate.bank, raw)
                probe = scores[0]
                triggered = bool(flags[0])
            else:
                triggered = bool(candidate.detect_novelty(raw))
                probe = dict(flag=triggered)
            probes.append(dict(observed_bytes=raw.numel(), **probe))
            if triggered:
                candidate.spawn_and_adapt_skill(NOVEL, raw, mode=EPISODIC_BIRTH)
                break
        observed[arm] = raw
        arms[arm] = dict(
            born=triggered, birth_at_byte=raw.numel() if triggered else None, probes=probes,
            adaptation_prefix_sha256=phase3.digest(phase7.bytes_for(raw)), seconds=time.perf_counter() - started,
            output_sha256=evaluate_b(candidate.bank, data, arm)[1],
            snapshot=snapshot(candidate, directory / f"{arm}.pt"),
            resident_weights_exact=initial == {name: state_hash(candidate.bank.skills[name]) for name in initial},
        )
    arms[PER_SKILL]["false_births"] = negatives
    arms[GLOBAL]["false_births"] = phase7.check_negatives(world, original)
    continual = copy.deepcopy(bank)
    raw = observed[PER_SKILL]
    adapt_skill(continual.base, raw, BIRTH_STEPS[EPISODIC_BIRTH])
    return dict(
        calibration=dict(**asdict(detector), ceilings_bpb=list(detector.ceilings_bpb)), id_probes=known,
        fixed=fixed, continual=evaluate_b(continual, data, CONTINUAL)[0],
        continual_observed_bytes=raw.numel(), arms=arms,
    )


def cold_restore(directory):
    suites = torch.load(directory / "test.pt", weights_only=True, map_location="cpu")
    with forbid_donor_forward() as guard:
        world = OpenWorldSkillBank.load_bank(directory / "a.pt")
        a = {}
        for arm in (SPARSE, DENSE):
            scores, output = evaluate_a(world.bank, suites["a"], arm)
            a[arm] = dict(scores=scores, output_sha256=output)
        original = suites["a"]["raw"][:len(phase3.DOMAINS) * phase3.CHECK_WINDOWS]
        negative = phase7.check_negatives(world, original)
        identity_a = identity(world.bank)
        b = {}
        for arm in (PER_SKILL, GLOBAL):
            bank = OpenWorldSkillBank.load_bank(directory / f"{arm}.pt").bank
            scores, output = evaluate_b(bank, suites["b"], arm)
            b[arm] = dict(scores=scores, output_sha256=output, identity=identity(bank))
    return dict(a=a, identity_a=identity_a, negative=negative, b=b, forbidden_calls=guard.donor_forward_calls)


def run_seed(seed):
    started = time.perf_counter()
    fit, suites, resident_manifest = phase3.data_for(seed)
    original = suites["single"]["raw"]
    exposures, checks, manifest_a = phase6.novel_data(seed, fit, original)
    original_b, val_b, exposure_b, check_b, manifest_b = phase7.fixtures_b(seed, fit, original)
    val_a = validation_a(seed, val_b, fit, original, exposures, checks)
    data_a = phase7.suite_from(torch.cat((original, *checks.values())), len(phase6.DOMAINS))
    data_b = phase7.suite_from(torch.cat((original_b, check_b)), len(phase3.DOMAINS) + 1)
    with tempfile.TemporaryDirectory(prefix="nexus-phase8-") as temp:
        directory = Path(temp)
        torch.save(fit, directory / "fit.pt")
        subprocess.run([sys.executable, __file__, "--produce", temp, "--seed", str(seed)], check=True, cwd=ROOT)
        producer = json.loads((directory / "producer.json").read_text())
        for name, receipt in producer.items():
            if digest(directory / f"{name}.pt") != receipt["source_sha256"]:
                raise ValueError(f"donor checkpoint custody mismatch for {name}")
        with forbid_donor_forward() as guard:
            skills = {name: load_skill(directory / f"{name}.pt", "residual8") for name in phase3.DOMAINS}
            a = MultiSkillBank(skills.pop("base_python"), skills)
            b, remapped = phase7.make_ascii_bank(a, directory)
            warm_a = run_warm_a(a, data_a, original, exposures, val_a, directory)
            warm_b = run_warm_b(b, data_b, original_b, val_b, exposure_b, directory)
        torch.save(dict(a=data_a, b=data_b), directory / "test.pt")
        cold = json.loads(subprocess.check_output([sys.executable, __file__, "--cold", temp], cwd=ROOT))

    fixed_a = warm_a.pop("fixed")
    a_rows = {arm: phase7.with_drift(cold["a"][arm]["scores"], fixed_a) for arm in (SPARSE, DENSE)}
    a_rows[FIXED] = phase7.with_drift(fixed_a, fixed_a)
    rate = a_rows[SPARSE]["avg_active_skills_per_byte"]
    a_rows[SPARSE]["savings_vs_nominal_k7_pct"] = 100 * (1 - rate / len(phase6.DOMAINS))
    a_rows[SPARSE]["savings_vs_measured_dense_pct"] = 100 * (1 - rate / a_rows[DENSE]["avg_active_skills_per_byte"])
    exact_a = (warm_a["bank_size"] == len(phase6.DOMAINS) and warm_a["snapshot"]["identity"] == cold["identity_a"]
               and all(warm_a["outputs"][arm] == cold["a"][arm]["output_sha256"] for arm in (SPARSE, DENSE)))
    fixed_b = warm_b.pop("fixed")
    for arm in (PER_SKILL, GLOBAL):
        record, recovered = warm_b["arms"][arm], cold["b"][arm]
        record["cold_restore_exact"] = (record["snapshot"]["identity"] == recovered["identity"]
                                        and record["output_sha256"] == recovered["output_sha256"])
        record.update(phase7.with_drift(recovered["scores"], fixed_b))
    warm_b["arms"][FIXED] = phase7.with_drift(fixed_b, fixed_b)
    warm_b["arms"][CONTINUAL] = phase7.with_drift(warm_b.pop("continual"), fixed_b)
    return dict(
        seed=seed, producer=producer, corpus_manifest=resident_manifest, ascii_checkpoint_sha256=remapped,
        forbidden_calls=guard.donor_forward_calls + cold["forbidden_calls"], seconds=time.perf_counter() - started,
        suite_a=dict(**warm_a, arms=a_rows, manifest=manifest_a, cold_negative=cold["negative"],
                     cold_restore_exact=exact_a,
                     cold_output_sha256={arm: result["output_sha256"] for arm, result in cold["a"].items()}),
        suite_b=dict(**warm_b, manifest=manifest_b),
    )


def verdicts(rows):
    calls = mastery = retention = joint_a = mixed_a = 0
    births = learned = retained_b = joint_b = false = 0
    for row in rows:
        a = row["suite_a"]["arms"][SPARSE]
        saved = row["suite_a"]["bank_size"] == len(phase6.DOMAINS) and a["avg_active_skills_per_byte"] <= 2.8
        master = a["novel_min_answer_pct"] >= 85
        retain = a["max_original_bpb_drift"] <= 0.015
        calls += saved
        mastery += master
        retention += retain
        joint_a += saved and master and retain
        mixed_a += sum((saved, master, retain)) >= 2

        arms = row["suite_b"]["arms"]
        candidate = arms[PER_SKILL]
        answer = candidate["domains"][NOVEL]["rule_accuracy_pct"]
        master = answer >= 80 and all(answer > arms[control]["domains"][NOVEL]["rule_accuracy_pct"]
                                      for control in (FIXED, GLOBAL))
        retain = candidate["max_original_bpb_drift"] <= 0.02
        birth = candidate["born"]
        births += birth
        learned += birth and master
        retained_b += retain
        joint_b += birth and master and retain
        false += sum(candidate["false_births"].values())

    return dict(
        suite_a=dict(verdict="PASS" if joint_a == len(SEEDS) else "MIXED" if mixed_a >= 4 else "FAIL",
                     call_budget=calls, mastery_all_three=mastery, within_retention_limit=retention,
                     joint_pass=joint_a, seeds_with_two_criteria=mixed_a),
        suite_b=dict(verdict="PASS" if joint_b == len(SEEDS) and not false else
                     "MIXED" if learned >= 4 and not false else "FAIL",
                     births=births, birth_and_mastery=learned, within_retention_limit=retained_b,
                     false_births=false, joint_pass=joint_b),
    )


def run(destination):
    if destination.exists():
        raise FileExistsError(destination)
    source = phase3.git("rev-parse", "HEAD").decode().strip()
    tree = phase3.git("rev-parse", "HEAD^{tree}").decode().strip()
    if phase3.git("status", "--porcelain") or phase3.git("rev-parse", "HEAD^").decode().strip() != BASE:
        raise ValueError("execute on the clean Phase 8 preregistration commit")
    started = time.perf_counter()
    rows = []
    for seed in SEEDS:
        row = run_seed(seed)
        rows.append(row)
        print(f"seed {seed} complete, K={row['suite_a']['bank_size']}, "
              f"per_skill_birth={row['suite_b']['arms'][PER_SKILL]['born']}", flush=True)
    if phase3.git("status", "--porcelain") or phase3.git("rev-parse", "HEAD").decode().strip() != source:
        raise ValueError("preregistered checkout changed during scoring")
    false_a = sum(sum(row["suite_a"][key].values()) for row in rows for key in ("negative", "cold_negative"))
    false_b = {arm: sum(sum(row["suite_b"]["arms"][arm]["false_births"].values()) for row in rows)
               for arm in (PER_SKILL, GLOBAL)}
    probes = len(rows) * len(phase3.DOMAINS) * (phase3.CHECK_WINDOWS // 2)
    atomic(destination, dict(
        schema="nexus-phase8-per-skill-novelty-dev-v1", source_sha=source, source_tree=tree, base_sha=BASE,
        fixture_source_sha=phase3.SOURCE, prereg_sha256=digest(ROOT / PREREG), platform=platform.platform(),
        python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__, torch_threads=torch.get_num_threads(),
        elapsed_seconds=time.perf_counter() - started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        receiver_checks=dict(
            seeds=len(rows), forbidden_calls=sum(row["forbidden_calls"] for row in rows), false_births_a=false_a,
            false_births_b=false_b, in_distribution_probes_a=2 * probes, in_distribution_probes_per_b_detector=probes,
            exact_resident_preservation_a=sum(row["suite_a"]["resident_exact"] for row in rows),
            exact_cold_restores_a=sum(row["suite_a"]["cold_restore_exact"] for row in rows),
            exact_cold_restores_b=sum(all(row["suite_b"]["arms"][arm]["cold_restore_exact"] for arm in (PER_SKILL, GLOBAL))
                                      for row in rows),
        ),
        scientific=verdicts(rows), rows=rows,
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
