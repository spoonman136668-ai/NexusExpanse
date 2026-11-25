"""Preregistered sparse K=7 and same-alphabet novelty DEV evaluations."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
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
from benchmarks import phase4_skill_birth as phase4
from benchmarks import phase5_skill_mastery as phase5
from benchmarks import phase6_hierarchical_scaling as phase6
from benchmarks.dev_artifacts import atomic, digest
from nexus_expanse.checkpoint_transport import load_skill
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.jepa import state_hash
from nexus_expanse.multi_skill_router import MultiSkillBank
from nexus_expanse.skill_birth import BIRTH_STEPS, EPISODIC_BIRTH, OpenWorldSkillBank, adapt_skill

BASE = "916b10151e761dad738a7c2df834bd267e6f1cb6"
PREREG = "docs/PHASE7_HARD_NOVELTY_SPARSE_K7_PREREG.md"
SEEDS = tuple(range(2001, 2006))
SPARSE, DENSE, FIXED = "hierarchical_sparse_k7", "hierarchical_dense_k7", "fixed_k4_bank"
TAIL, MEAN, CONTINUAL = "open_world_tail_novelty", "mean_bpb_only_birth", "continual_finetune_base"
NOVEL = "rule_same_alpha"
VAL_WINDOWS = 48
QUANTILE = 0.90
TAIL_MARGIN = 0.5
VAL_TEXT_PATHS = {
    "base_python": ("nexus_expanse/cli.py", "nexus_expanse/donor_assimilation.py"),
    "markdown_prose": ("docs/GATE19_PREREG.md", "docs/GATE20_PREREG.md", "docs/GATE21_PREREG.md"),
}
ASCII_PERMUTATION = torch.arange(256)
ASCII_PERMUTATION[145:153] = torch.arange(97, 105)
ASCII_PERMUTATION[97:105] = torch.arange(145, 153)


def bytes_for(raw):
    return raw.to(torch.uint8).contiguous().numpy().tobytes()


def window_grams(raw):
    return set().union(*(phase3.grams(bytes(row)) for row in raw.tolist()))


def alternate_map(tables, rng):
    result = []
    for start in range(0, 32, 8):
        while True:
            choice = rng.sample(range(97, 105), 8)
            if all(value not in (tables[0][start + index], tables[1][start + index])
                   for index, value in enumerate(choice)):
                result.extend(choice)
                break
    return result


def fixtures_b(seed, fit, original):
    """Keep the resident grammar and map all Rule A/B answer bytes into ASCII."""
    n = phase3.CHECK_WINDOWS
    original = original.clone()
    original[n:3 * n] = ASCII_PERMUTATION[original[n:3 * n]]
    mapped_fit = {name: ASCII_PERMUTATION[raw] if name in ("rule_A", "rule_B") else raw for name, raw in fit.items()}
    tables = [ASCII_PERMUTATION[table].tolist() for table in phase3.rules(seed)]
    rng_val = random.Random(seed + 57100)
    used = set().union(*(window_grams(raw) for raw in mapped_fit.values()))
    val = {}
    for idx, name in enumerate(phase3.DOMAINS):
        candidates = (phase3.text_windows(VAL_TEXT_PATHS[name], rng_val)
                      if name in VAL_TEXT_PATHS else phase3.rule_windows(tables[idx - 1], rng_val, 384))
        windows = phase3.select_windows(candidates, VAL_WINDOWS, used)
        val[name] = torch.tensor([list(window) for window in windows]).reshape(-1, 2 * phase3.BYTES)
        used.update(window_grams(val[name]))
    original_pairs = original.reshape(-1, 2 * phase3.BYTES)
    # CHECK is an independent split. A collision invalidates the fixture.
    original_check_grams = window_grams(original_pairs)
    if original_check_grams & used:
        raise ValueError("Suite B resident CHECK overlaps FIT or VAL")
    used.update(original_check_grams)

    rng_novel = random.Random(seed + 57101)
    candidates = phase3.rule_windows(alternate_map(tables, rng_novel), rng_novel, 128)
    encountered = phase3.select_windows(candidates[:64], phase4.MAX_ENCOUNTER_BLOCKS, used)
    used.update(phase3.grams(b"".join(encountered)))
    check = phase3.select_windows(candidates[64:], n, used)
    alphabet = set(range(40, 48)) | set(range(80, 84)) | {125} | set(range(97, 105))
    if any(set(bytes_for(mapped_fit[name])) != alphabet for name in ("rule_A", "rule_B")):
        raise ValueError("resident rule alphabet changed")
    if set(b"".join(encountered)) != alphabet or set(b"".join(check)) != alphabet:
        raise ValueError("novel rule alphabet does not exactly match resident rules")
    manifest = dict(
        ascii_alphabet=sorted(alphabet), redirects=[[value, int(ASCII_PERMUTATION[value])] for value in range(145, 153)],
        val_sha256={name: phase3.digest(bytes_for(raw)) for name, raw in val.items()},
        val_fit_check_32byte_overlap=0,
        resident_check_sha256=phase3.digest(bytes_for(original)),
        encounter_sha256=phase3.digest(b"".join(encountered)), check_sha256=phase3.digest(b"".join(check)),
        encounter_check_32byte_overlap=0,
    )
    return original, val, torch.tensor([list(block) for block in encountered]), torch.tensor([list(block) for block in check]), manifest


def suite_from(raw, count):
    labels = torch.arange(count).repeat_interleave(phase3.CHECK_WINDOWS)[:, None].expand_as(raw)
    return dict(raw=raw, labels=labels)


def make_ascii_bank(bank, directory):
    skills = {name: copy.deepcopy(skill) for name, skill in bank.skills.items()}
    remapped = {}
    for name in ("rule_A", "rule_B"):
        pack = torch.load(directory / f"{name}.pt", weights_only=True, map_location="cpu")
        pack["byte_to_row"] = pack["byte_to_row"][ASCII_PERMUTATION]
        path = directory / f"{name}-ascii.pt"
        torch.save(pack, path)
        skills[name] = load_skill(path, "residual8")
        remapped[name] = digest(path)
    return MultiSkillBank(skills.pop(bank.base_name), skills), remapped


def evaluate_a(bank, suite, arm):
    raw = suite["raw"]
    if arm == SPARSE:
        logp, weights, calls = bank.forward_hmm_sparse_hierarchical(raw)
    elif arm == DENSE:
        logp, weights = bank.forward_hmm_bma_hierarchical(raw)
        calls = len(bank.skills) * (raw.size(1) - 1) / raw.size(1)
    else:
        logp, weights = bank.forward_hmm_bma(raw)
        calls = len(bank.skills) * (raw.size(1) - 1) / raw.size(1)
    scored = phase6.summarize(bank, suite, logp, weights)
    return dict(**scored, avg_active_skills_per_byte=calls), phase3.output_digest(logp, weights, calls)


def evaluate_b(bank, suite, arm):
    raw, labels = suite["raw"], suite["labels"]
    if arm in (TAIL, MEAN):
        logp, weights = bank.forward_hmm_bma_hierarchical(raw)
    else:
        logp, weights = bank.forward_hmm_bma(raw)
    scores = phase4.summarize(raw, labels, logp, weights)
    novel = scores["domains"].pop(phase4.NOVEL)
    if NOVEL not in bank.skills:
        novel["routing_accuracy_pct"] = 0.0
    scores["domains"][NOVEL] = novel
    return scores, phase3.output_digest(logp, weights, 0)


def check_negatives(world, raw, tail=None):
    groups = world.detect_novelty(raw.reshape(-1, 2 * phase3.BYTES), tail=tail).view(len(phase3.DOMAINS), -1)
    return {name: int(flags.sum()) for name, flags in zip(phase3.DOMAINS, groups, strict=True)}


def with_drift(result, fixed):
    drift = {name: result["domains"][name]["bpb"] - fixed["domains"][name]["bpb"] for name in phase3.DOMAINS}
    return dict(**result, retention_drift_bpb=drift, max_original_bpb_drift=max(drift.values()))


def snapshot(world, path):
    world.save_bank(path)
    return dict(identity=phase6.identity(world.bank), sha256=digest(path), bytes=path.stat().st_size)


def run_warm_a(bank, data, original, encounters, directory):
    world = OpenWorldSkillBank(bank)
    initial = phase4.skill_hashes(bank)
    before = bank.component_log_probs(original)
    fixed = evaluate_a(bank, data, FIXED)[0]
    negative = check_negatives(world, original)
    stages, _ = phase6.grow(world, encounters)
    hashes = {name: state_hash(bank.skills[name]) for name in initial}
    immutable = initial == hashes and torch.equal(before, bank.component_log_probs(original)[:, :, :len(initial)])
    outputs = {arm: evaluate_a(bank, data, arm)[1] for arm in (SPARSE, DENSE)}
    seal = snapshot(world, directory / "a.pt")
    return dict(fixed=fixed, stages=stages, negative=negative, resident_exact=immutable,
                outputs=outputs, snapshot=seal, bank_size=len(bank.skills))


def run_warm_b(bank, data, original, val, encounter, directory):
    world = OpenWorldSkillBank(bank)
    calibration = world.calibrate_tail_surprise(val, quantile=QUANTILE, margin_bpb=TAIL_MARGIN)
    negatives = {TAIL: check_negatives(world, original, tail=calibration), MEAN: check_negatives(world, original)}
    initial = phase4.skill_hashes(bank)
    fixed = evaluate_b(bank, data, FIXED)[0]
    observed_by_arm, arms = {}, {}
    for arm in (TAIL, MEAN):
        candidate = OpenWorldSkillBank(copy.deepcopy(bank))
        probes = []
        began = time.perf_counter()
        for n in range(phase4.FIRST_PROBE_BLOCKS, phase4.MAX_ENCOUNTER_BLOCKS + 1):
            raw = encounter[:n].reshape(1, -1)
            born = candidate.observe(NOVEL, raw, mode=EPISODIC_BIRTH, tail=calibration if arm == TAIL else None)
            probes.append(dict(observed_bytes=raw.numel(), triggered=born is not None))
            if born is not None:
                break
        observed_by_arm[arm] = raw
        output = evaluate_b(candidate.bank, data, arm)[1]
        seal = snapshot(candidate, directory / f"{arm}.pt")
        exact = initial == {name: state_hash(candidate.bank.skills[name]) for name in initial}
        arms[arm] = dict(
            born=born is not None, birth_at_byte=raw.numel() if born is not None else None,
            probes=probes, adaptation_prefix_sha256=phase3.digest(bytes_for(raw)),
            seconds=time.perf_counter() - began, output_sha256=output, snapshot=seal,
            resident_weights_exact=exact, false_births=negatives[arm],
        )

    continual = copy.deepcopy(bank)
    raw = observed_by_arm[TAIL]
    adapt_skill(continual.base, raw, BIRTH_STEPS[EPISODIC_BIRTH])
    scores = evaluate_b(continual, data, CONTINUAL)[0]
    return dict(
        calibration=asdict(calibration), calibration_margin_bpb=TAIL_MARGIN,
        fixed=fixed, continual=scores, continual_observed_bytes=raw.numel(), arms=arms,
    )


def cold_restore(directory):
    suites = torch.load(directory / "test.pt", weights_only=True, map_location="cpu")
    with forbid_donor_forward() as guard:
        world = OpenWorldSkillBank.load_bank(directory / "a.pt")
        scores_a = {}
        for arm in (SPARSE, DENSE):
            metrics, output = evaluate_a(world.bank, suites["a"], arm)
            scores_a[arm] = dict(metrics=metrics, output_sha256=output)
        original = suites["a"]["raw"][:len(phase3.DOMAINS) * phase3.CHECK_WINDOWS]
        negative = check_negatives(world, original)
        identity_a = phase6.identity(world.bank)
        scores_b = {}
        for arm in (TAIL, MEAN):
            bank = OpenWorldSkillBank.load_bank(directory / f"{arm}.pt").bank
            metrics, output = evaluate_b(bank, suites["b"], arm)
            scores_b[arm] = dict(metrics=metrics, output_sha256=output, identity=phase6.identity(bank))
    return dict(a=scores_a, identity_a=identity_a, a_negative=negative, b=scores_b,
                forbidden_calls=guard.donor_forward_calls)


def run_seed(seed):
    started = time.perf_counter()
    fit, suites, manifest = phase3.data_for(seed)
    original = suites["single"]["raw"]
    exposures, checks, manifest_a = phase6.novel_data(seed, fit, original)
    original_b, val, encounter_b, check_b, manifest_b = fixtures_b(seed, fit, original)
    data_a = suite_from(torch.cat((original, *checks.values())), len(phase6.DOMAINS))
    data_b = suite_from(torch.cat((original_b, check_b)), len(phase3.DOMAINS) + 1)

    with tempfile.TemporaryDirectory(prefix="nexus-phase7-") as temp:
        directory = Path(temp)
        torch.save(fit, directory / "fit.pt")
        subprocess.run([sys.executable, __file__, "--produce", temp, "--seed", str(seed)], check=True, cwd=ROOT)
        producer = json.loads((directory / "producer.json").read_text())
        for name, receipt in producer.items():
            if digest(directory / f"{name}.pt") != receipt["source_sha256"]:
                raise ValueError(f"donor checkpoint custody mismatch for {name}")
        with forbid_donor_forward() as guard:
            skills = {name: load_skill(directory / f"{name}.pt", "residual8") for name in phase3.DOMAINS}
            bank_a = MultiSkillBank(skills.pop("base_python"), skills)
            bank_b, remapped = make_ascii_bank(bank_a, directory)
            warm_a = run_warm_a(bank_a, data_a, original, exposures, directory)
            warm_b = run_warm_b(bank_b, data_b, original_b, val, encounter_b, directory)
        torch.save(dict(a=data_a, b=data_b), directory / "test.pt")
        cold = json.loads(subprocess.check_output([sys.executable, __file__, "--cold", temp], cwd=ROOT))

    exact_a = (warm_a["bank_size"] == len(phase6.DOMAINS) and warm_a["snapshot"]["identity"] == cold["identity_a"]
               and all(warm_a["outputs"][arm] == cold["a"][arm]["output_sha256"] for arm in (SPARSE, DENSE)))
    a_rows = {arm: with_drift(cold["a"][arm]["metrics"], warm_a["fixed"]) for arm in (SPARSE, DENSE)}
    a_rows[FIXED] = with_drift(warm_a["fixed"], warm_a["fixed"])
    rate = a_rows[SPARSE]["avg_active_skills_per_byte"]
    a_rows[SPARSE]["savings_vs_nominal_k7_pct"] = 100 * (1 - rate / len(phase6.DOMAINS))
    if warm_a["bank_size"] == len(phase6.DOMAINS):
        a_rows[SPARSE]["savings_vs_measured_dense_pct"] = 100 * (
            1 - rate / a_rows[DENSE]["avg_active_skills_per_byte"]
        )

    for arm in (TAIL, MEAN):
        record = warm_b["arms"][arm]
        recovered = cold["b"][arm]
        record["cold_restore_exact"] = (record["snapshot"]["identity"] == recovered["identity"]
                                        and record["output_sha256"] == recovered["output_sha256"])
        record.update(with_drift(recovered["metrics"], warm_b["fixed"]))
    warm_b["arms"][FIXED] = with_drift(warm_b["fixed"], warm_b["fixed"])
    warm_b["arms"][CONTINUAL] = with_drift(warm_b["continual"], warm_b["fixed"])
    del warm_b["fixed"], warm_b["continual"]
    return dict(
        seed=seed, corpus_manifest=manifest, producer=producer, ascii_checkpoint_sha256=remapped,
        forbidden_calls=guard.donor_forward_calls + cold["forbidden_calls"], seconds=time.perf_counter() - started,
        suite_a=dict(arms=a_rows, manifest=manifest_a, stages=warm_a["stages"], bank_size=warm_a["bank_size"],
                     false_births_k4=warm_a["negative"], false_births_cold=cold["a_negative"],
                     resident_exact=warm_a["resident_exact"], cold_restore_exact=exact_a,
                     snapshot=warm_a["snapshot"], output_sha256=warm_a["outputs"],
                     cold_output_sha256={arm: value["output_sha256"] for arm, value in cold["a"].items()}),
        suite_b=dict(**warm_b, manifest=manifest_b),
    )


def verdicts(rows):
    count = len(SEEDS)
    compute = mastery = retention = wins_a = mixed_a = births = learned = retained_b = wins_b = 0
    false = 0
    for row in rows:
        a = row["suite_a"]["arms"][SPARSE]
        savings = row["suite_a"]["bank_size"] == len(phase6.DOMAINS) and a["avg_active_skills_per_byte"] <= 2.8
        answer = a["novel_min_answer_pct"] >= 85
        drift = a["max_original_bpb_drift"] <= 0.015
        compute += savings
        mastery += answer
        retention += drift
        wins_a += savings and answer and drift
        mixed_a += sum((savings, answer, drift)) >= 2

        b = row["suite_b"]["arms"]
        tail = b[TAIL]
        score = tail["domains"][NOVEL]["rule_accuracy_pct"]
        birth = tail["born"]
        master = score >= 80 and score > b[FIXED]["domains"][NOVEL]["rule_accuracy_pct"]
        retain = tail["max_original_bpb_drift"] <= 0.02
        false += sum(tail["false_births"].values())
        births += birth
        learned += birth and master
        retained_b += retain
        wins_b += birth and master and retain
    status_a = "PASS" if wins_a == count else "MIXED" if mixed_a >= 4 else "FAIL"
    status_b = "PASS" if wins_b == count and not false else "MIXED" if learned >= 4 and not false else "FAIL"
    return dict(
        suite_a=dict(verdict=status_a, call_budget=compute, mastery_all_three=mastery,
                     within_retention_limit=retention, joint_pass=wins_a, seeds_with_two_criteria=mixed_a),
        suite_b=dict(verdict=status_b, births=births, birth_and_mastery=learned,
                     within_retention_limit=retained_b, false_births=false, joint_pass=wins_b),
    )


def run(destination):
    if destination.exists():
        raise FileExistsError(destination)
    source = phase3.git("rev-parse", "HEAD").decode().strip()
    tree = phase3.git("rev-parse", "HEAD^{tree}").decode().strip()
    if phase3.git("status", "--porcelain") or phase3.git("rev-parse", "HEAD^").decode().strip() != BASE:
        raise ValueError("execute on the clean Phase 7 preregistration commit")
    started = time.perf_counter()
    rows = []
    for seed in SEEDS:
        row = run_seed(seed)
        rows.append(row)
        arms = row["suite_b"]["arms"]
        print(f"seed {seed} complete, K={row['suite_a']['bank_size']}, "
              f"tail_birth={arms[TAIL]['born']}, mean_birth={arms[MEAN]['born']}", flush=True)
    if phase3.git("status", "--porcelain") or phase3.git("rev-parse", "HEAD").decode().strip() != source:
        raise ValueError("preregistered checkout changed during scoring")
    false_a = sum(sum(row["suite_a"][key].values()) for row in rows for key in ("false_births_k4", "false_births_cold"))
    false_b = {arm: sum(sum(row["suite_b"]["arms"][arm]["false_births"].values()) for row in rows) for arm in (TAIL, MEAN)}
    probes = len(rows) * len(phase3.DOMAINS) * (phase3.CHECK_WINDOWS // 2)
    atomic(destination, dict(
        schema="nexus-phase7-hard-novelty-sparse-k7-dev-v1", source_sha=source, source_tree=tree, base_sha=BASE,
        fixture_source_sha=phase3.SOURCE, prereg_sha256=digest(ROOT / PREREG), platform=platform.platform(),
        python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__, torch_threads=torch.get_num_threads(),
        elapsed_seconds=time.perf_counter() - started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        receiver_checks=dict(
            seeds=len(rows), forbidden_calls=sum(row["forbidden_calls"] for row in rows), false_births_a=false_a,
            false_births_b=false_b, in_distribution_probes_a=probes * 2, in_distribution_probes_per_b_detector=probes,
            exact_resident_preservation_a=sum(row["suite_a"]["resident_exact"] for row in rows),
            exact_cold_restores_a=sum(row["suite_a"]["cold_restore_exact"] for row in rows),
            exact_cold_restores_b=sum(all(row["suite_b"]["arms"][arm]["cold_restore_exact"] for arm in (TAIL, MEAN))
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
