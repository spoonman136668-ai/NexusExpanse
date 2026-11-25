"""Preregistered resident-anchored routing and sequential K=7 birth confirmation."""
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
from benchmarks import phase4_skill_birth as phase4
from benchmarks import phase5_skill_mastery as phase5
from benchmarks.dev_artifacts import atomic, digest
from nexus_expanse.checkpoint_transport import load_skill
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.jepa import state_hash
from nexus_expanse.multi_skill_router import MultiSkillBank
from nexus_expanse.skill_birth import BIRTH_STEPS, EPISODIC_BIRTH, OpenWorldSkillBank, adapt_skill

BASE = "362b8ccc84fde134e890b5862ad6a1760e279776"
PREREG = "docs/PHASE6_HIERARCHICAL_SCALING_PREREG.md"
SEEDS = tuple(range(1901, 1906))
SPECS = {
    "rule_C": (176, 192, 208, 224),
    "rule_D": (160, 168, 172, 184),
    "rule_E": (200, 209, 213, 232),
}
DOMAINS = (*phase3.DOMAINS, *SPECS)
HIER = "hierarchical_episodic_k7"
UNIFORM = "uniform_episodic_k7"
FIXED = "fixed_k4_bank"
CONTINUAL = "continual_finetune_base"
ARMS = (HIER, UNIFORM, FIXED, CONTINUAL)
SWITCH = 0.05
BORN_MASS = 0.02
BORN_SWITCH = 0.10
DRIFT_LIMIT = 0.015


def novel_data(seed, original_fit, original_check):
    """Generate mutually disjoint raw alphabets and disjoint encounter/test windows."""
    seen = set()
    for rows in (*original_fit.values(), original_check):
        for row in rows.tolist():
            seen.update(phase3.grams(bytes(row)))
    alphabets = set()
    encounters, checks, manifest = {}, {}, {}
    for index, (name, (a, b, delimiter, target)) in enumerate(SPECS.items()):
        alphabet = set(range(a, a + 8)) | set(range(b, b + 4)) | {delimiter} | set(range(target, target + 8))
        if alphabet & alphabets or len(alphabet) != 21:
            raise ValueError("novel alphabets must be disjoint")
        alphabets.update(alphabet)
        rng = random.Random(seed + 47000 + index * 1000)
        table = [target + value for _ in range(4) for value in rng.sample(range(8), 8)]
        candidates = []
        for _ in range(128):
            keys = rng.sample(range(32), phase3.BYTES // 4)
            candidates.append(bytes(byte for key in keys
                                    for byte in (a + key % 8, b + key // 8, delimiter, table[key])))
        exposure = phase3.select_windows(candidates[:64], phase4.MAX_ENCOUNTER_BLOCKS, seen)
        joined = b"".join(exposure)
        seen.update(phase3.grams(joined))
        check = phase3.select_windows(candidates[64:], phase3.CHECK_WINDOWS, seen)
        encounters[name] = torch.tensor([list(window) for window in exposure])
        checks[name] = torch.tensor([list(window) for window in check])
        manifest[name] = dict(
            alphabet=sorted(alphabet), encounter_sha256=phase3.digest(joined),
            check_sha256=phase3.digest(b"".join(check)), encounter_check_32byte_overlap=0,
        )
    return encounters, checks, manifest


def predict(bank, raw, arm):
    if arm == HIER:
        return bank.forward_hmm_bma_hierarchical(raw, SWITCH, BORN_MASS, BORN_SWITCH)
    return bank.forward_hmm_bma(raw, SWITCH)


def summarize(bank, suite, logp, weights):
    raw, labels = suite["raw"], suite["labels"]
    active = weights.argmax(-1)
    losses = -logp.gather(-1, raw.unsqueeze(-1)).squeeze(-1).double() / math.log(2)
    correct = logp.argmax(-1) == raw
    results = {}
    for index, name in enumerate(DOMAINS):
        rows = labels[:, 0] == index
        skill_index = bank.skill_names.index(name) if name in bank.skills else -1
        result = dict(bpb=float(losses[rows].mean()),
                      routing_accuracy_pct=phase3.accuracy(active[rows] == skill_index))
        if index in (1, 2) or name in SPECS:
            result["answer_accuracy_pct"] = phase3.accuracy(correct[rows, 3::4])
        if name in SPECS:
            # Mean first assigned position, with a non-recovered stream charged the full length.
            assigned = active[rows] == skill_index
            positions = torch.arange(raw.size(1)).expand_as(assigned)
            first = torch.where(assigned, positions, raw.size(1)).min(-1).values
            result["first_route_mean_bytes"] = float(first.double().mean())
        results[name] = result
    novel = [results[name] for name in SPECS]
    answer = [result["answer_accuracy_pct"] for result in novel]
    routing = [result["routing_accuracy_pct"] for result in novel]
    return dict(
        original_bpb=float(losses[labels < len(phase3.DOMAINS)].mean()), domains=results,
        novel_mean_answer_pct=sum(answer) / len(answer), novel_min_answer_pct=min(answer),
        novel_mean_routing_pct=sum(routing) / len(routing), novel_min_routing_pct=min(routing),
    )


def evaluate(bank, suite, arm):
    logp, weights = predict(bank, suite["raw"], arm)
    return summarize(bank, suite, logp, weights), phase3.output_digest(logp, weights, 0)


def identity(bank):
    return dict(**phase3.bank_identity(bank), resident_names=list(bank.resident_names))


def check_revisit(world, rule_c):
    pairs = rule_c.reshape(-1, 2 * phase3.BYTES)
    bank = world.bank
    if "rule_C" not in bank.skills:
        return []
    _, weights = predict(bank, pairs, HIER)
    matches = weights.argmax(-1) == bank.skill_names.index("rule_C")
    results = []
    for raw, correct in zip(pairs, matches, strict=True):
        disposable = OpenWorldSkillBank(copy.deepcopy(bank))
        # A fresh label prevents the registered-name guard from suppressing a duplicate.
        result = disposable.observe("unlabeled_revisit", raw[None], mode=EPISODIC_BIRTH)
        results.append(dict(born_duplicate=result is not None, rule_c_routing_pct=phase3.accuracy(correct)))
    return results


def cold_restore(directory):
    suite = torch.load(directory / "test.pt", weights_only=True, map_location="cpu")
    with forbid_donor_forward() as guard:
        world = OpenWorldSkillBank.load_bank(directory / "expanded.pt")
        bank = world.bank
        results = {}
        for arm in (HIER, UNIFORM):
            scores, output = evaluate(bank, suite, arm)
            results[arm] = dict(scores=scores, output_sha256=output)
        old = suite["labels"][:, 0] < len(phase3.DOMAINS)
        negative = phase5.negative_probes(world, suite["raw"][old])
        rule_c = suite["raw"][suite["labels"][:, 0] == len(phase3.DOMAINS)]
        revisits = check_revisit(world, rule_c)
    return dict(arms=results, identity=identity(bank), negative=negative, revisits=revisits,
                forbidden_calls=guard.donor_forward_calls)


def grow(world, encounters):
    stages, used = [], []
    for name, encounter in encounters.items():
        started = time.perf_counter()
        probes = []
        for blocks in range(phase4.FIRST_PROBE_BLOCKS, phase4.MAX_ENCOUNTER_BLOCKS + 1):
            observed = encounter[:blocks].reshape(1, -1)
            born = world.observe(name, observed, mode=EPISODIC_BIRTH)
            probes.append(dict(observed_bytes=observed.numel(), triggered=born is not None))
            if born is not None:
                break
        used.append(observed)
        stages.append(dict(
            domain=name, born=born is not None, bank_size=len(world.bank.skills),
            observed_bytes=observed.numel(), probes=probes, seconds=time.perf_counter() - started,
            encounter_prefix_sha256=phase3.digest(bytes(observed[0].tolist())),
            born_state_sha256=state_hash(born) if born is not None else None,
        ))
    return stages, used


def run_seed(seed):
    started = time.perf_counter()
    fit, suites, corpus_manifest = phase3.data_for(seed)
    original = suites["single"]["raw"]
    encounters, checks, novel_manifest = novel_data(seed, fit, original)
    raw = torch.cat((original, *checks.values()))
    labels = torch.arange(len(DOMAINS)).repeat_interleave(phase3.CHECK_WINDOWS)[:, None].expand_as(raw)
    suite = dict(raw=raw, labels=labels)

    with tempfile.TemporaryDirectory(prefix="nexus-phase6-") as temp:
        directory = Path(temp)
        torch.save(fit, directory / "fit.pt")
        subprocess.run([sys.executable, __file__, "--produce", temp, "--seed", str(seed)], check=True, cwd=ROOT)
        producer = json.loads((directory / "producer.json").read_text())
        for name, receipt in producer.items():
            if digest(directory / f"{name}.pt") != receipt["source_sha256"]:
                raise ValueError(f"donor checkpoint custody mismatch for {name}")

        with forbid_donor_forward() as guard:
            skills = {name: load_skill(directory / f"{name}.pt", "residual8") for name in phase3.DOMAINS}
            bank = MultiSkillBank(skills.pop(phase3.DOMAINS[0]), skills)
            world = OpenWorldSkillBank(bank)
            finetune = copy.deepcopy(bank)
            resident_hashes = phase4.skill_hashes(bank)
            original_before = bank.component_log_probs(original)
            fixed, _ = evaluate(bank, suite, FIXED)
            negative_before = phase5.negative_probes(world, original)

            stages, used = grow(world, encounters)
            resident_exact = resident_hashes == {name: state_hash(bank.skills[name]) for name in bank.resident_names}
            resident_predictions_exact = torch.equal(
                original_before, bank.component_log_probs(original)[:, :, :len(bank.resident_names)]
            )
            earlier_births_exact = all(stage["born_state_sha256"] == state_hash(bank.skills[stage["domain"]])
                                       for stage in stages if stage["born"])

            fit_start = time.perf_counter()
            for observed in used:
                adapt_skill(finetune.base, observed, BIRTH_STEPS[EPISODIC_BIRTH])
            continual_seconds = time.perf_counter() - fit_start
            continual, _ = evaluate(finetune, suite, CONTINUAL)

            hot = {arm: evaluate(bank, suite, arm)[1] for arm in (HIER, UNIFORM)}
            path = directory / "expanded.pt"
            world.save_bank(path)
            snapshot = dict(sha256=digest(path), bytes=path.stat().st_size, identity=identity(bank))

        torch.save(suite, directory / "test.pt")
        cold = json.loads(subprocess.check_output([sys.executable, __file__, "--cold", temp], cwd=ROOT))

    cold_exact = snapshot["identity"] == cold["identity"] and all(
        hot[arm] == cold["arms"][arm]["output_sha256"] for arm in (HIER, UNIFORM)
    )
    results = {FIXED: fixed, CONTINUAL: continual}
    for arm in (HIER, UNIFORM):
        results[arm] = cold["arms"][arm]["scores"]
    for arm in ARMS:
        drift = {name: results[arm]["domains"][name]["bpb"] - fixed["domains"][name]["bpb"]
                 for name in phase3.DOMAINS}
        results[arm].update(retention_drift_bpb=drift, max_original_bpb_drift=max(drift.values()))
    return dict(
        seed=seed, stages=stages, corpus_manifest=corpus_manifest, novel_manifest=novel_manifest, producer=producer,
        old_state_hashes=resident_hashes, resident_weights_exact=resident_exact,
        resident_forecasts_exact=resident_predictions_exact, earlier_births_exact=earlier_births_exact,
        cold_restore_exact=cold_exact, final_bank_size=len(bank.skills), snapshot=snapshot,
        hot_output_sha256=hot, cold_output_sha256={arm: item["output_sha256"] for arm, item in cold["arms"].items()},
        forbidden_calls=guard.donor_forward_calls + cold["forbidden_calls"],
        false_births={"k4": negative_before, "cold_expanded": cold["negative"]},
        duplicate_births=sum(item["born_duplicate"] for item in cold["revisits"]), revisits=cold["revisits"],
        continual_fit_seconds=continual_seconds, elapsed_seconds=time.perf_counter() - started, **results,
    )


def science(rows):
    mastery = routing = retention = beats_uniform = beats_continual = passed = 0
    for row in rows:
        candidate = row[HIER]
        answer = candidate["novel_min_answer_pct"] >= 85
        routed = candidate["novel_min_routing_pct"] >= 90
        drift = candidate["max_original_bpb_drift"]
        retained = drift <= DRIFT_LIMIT
        uniform_win = drift < row[UNIFORM]["max_original_bpb_drift"]
        continual_win = drift < row[CONTINUAL]["max_original_bpb_drift"]
        mastery += answer
        routing += routed
        retention += retained
        beats_uniform += uniform_win
        beats_continual += continual_win
        passed += answer and routed and retained and uniform_win and continual_win
    verdict = "PASS" if passed == len(SEEDS) else "MIXED" if mastery == len(SEEDS) and beats_uniform >= 4 else "FAIL"
    return dict(
        verdict=verdict, mastery_all_three=mastery, cold_routing_all_three=routing,
        within_per_domain_retention_limit=retention, beats_uniform_max_drift=beats_uniform,
        beats_continual_max_drift=beats_continual, all_pass_criteria=passed,
    )


def run(destination):
    if destination.exists():
        raise FileExistsError(destination)
    source = phase3.git("rev-parse", "HEAD").decode().strip()
    tree = phase3.git("rev-parse", "HEAD^{tree}").decode().strip()
    if phase3.git("status", "--porcelain") or phase3.git("rev-parse", "HEAD^").decode().strip() != BASE:
        raise ValueError("execute on the clean Phase 6 preregistration commit")
    started = time.perf_counter()
    rows = []
    for seed in SEEDS:
        row = run_seed(seed)
        rows.append(row)
        print(f"seed {seed} complete, K={row['final_bank_size']}, "
              f"cold={row['cold_restore_exact']}, duplicates={row['duplicate_births']}", flush=True)
    if phase3.git("status", "--porcelain") or phase3.git("rev-parse", "HEAD").decode().strip() != source:
        raise ValueError("preregistered checkout changed during scoring")
    atomic(destination, dict(
        schema="nexus-phase6-hierarchical-scaling-dev-v1", source_sha=source, source_tree=tree, base_sha=BASE,
        fixture_source_sha=phase3.SOURCE, prereg_sha256=digest(ROOT / PREREG),
        platform=platform.platform(), python=platform.python_version(), torch=str(torch.__version__),
        numpy=np.__version__, torch_threads=torch.get_num_threads(), elapsed_seconds=time.perf_counter() - started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        protocol=dict(switch_prob=SWITCH, born_prior_mass=BORN_MASS, born_switch_share=BORN_SWITCH,
                      per_domain_drift_limit=DRIFT_LIMIT, steps_per_encounter=BIRTH_STEPS[EPISODIC_BIRTH]),
        receiver_checks=dict(
            seeds=len(rows), full_k7_seeds=sum(row["final_bank_size"] == len(DOMAINS) for row in rows),
            forbidden_calls=sum(row["forbidden_calls"] for row in rows),
            false_births=sum(sum(sum(counts.values()) for counts in row["false_births"].values()) for row in rows),
            in_distribution_probes=len(rows) * 2 * len(phase3.DOMAINS) * (phase3.CHECK_WINDOWS // 2),
            duplicate_births=sum(row["duplicate_births"] for row in rows),
            rule_c_revisits=sum(len(row["revisits"]) for row in rows),
            correctly_routed_revisits=sum(sum(item["rule_c_routing_pct"] >= 90 for item in row["revisits"]) for row in rows),
            exact_resident_preservation=sum(row["resident_weights_exact"] and row["resident_forecasts_exact"]
                                            and row["earlier_births_exact"] for row in rows),
            exact_k7_cold_restores=sum(row["cold_restore_exact"] and row["final_bank_size"] == len(DOMAINS)
                                       for row in rows),
        ),
        scientific=science(rows), rows=rows,
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
