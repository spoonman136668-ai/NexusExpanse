"""Preregistered comparison of episodic and neural skill consolidation."""
from __future__ import annotations

import argparse
import copy
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

from benchmarks import phase4_skill_birth as phase4
from benchmarks.dev_artifacts import atomic, digest
from nexus_expanse.checkpoint_transport import load_skill
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.episodic_born_skill import MEMORY_WEIGHT, SUFFIX_BYTES
from nexus_expanse.jepa import state_hash
from nexus_expanse.multi_skill_router import MultiSkillBank, load_bank
from nexus_expanse.skill_birth import (
    BIRTH_STEPS, CONSOLIDATED_BIRTH, EPISODIC_BIRTH, NOVELTY_THRESHOLD_BPB, NOVELTY_WINDOW,
    PHASE4_BIRTH, OpenWorldSkillBank, adapt_skill,
)

phase3 = phase4.phase3
BASE = "61be699e0c254c6361e3a3cd1cb2a80350e6ef85"
PREREG = "docs/PHASE5_SKILL_MASTERY_PREREG.md"
SEEDS = tuple(range(1801, 1806))
NOVEL = phase4.NOVEL
BIRTH_ARMS = (EPISODIC_BIRTH, CONSOLIDATED_BIRTH, PHASE4_BIRTH)
FIXED = "fixed_k4_bank"
CONTINUAL = "continual_finetune_base"
ARMS = (*BIRTH_ARMS, FIXED, CONTINUAL)


def evaluate(bank, suite):
    logp, weights = phase4.route(bank, **suite)
    scores = phase4.summarize(**suite, logp=logp, routing=weights)
    return scores, phase3.output_digest(logp, weights, 0)


def cold_restore(directory):
    suite = torch.load(directory / "test.pt", weights_only=True, map_location="cpu")
    with forbid_donor_forward() as guard:
        results = {}
        for arm in BIRTH_ARMS:
            bank = load_bank(directory / f"{arm}.pt")
            scores, output = evaluate(bank, suite)
            results[arm] = dict(scores=scores, output_sha256=output, identity=phase3.bank_identity(bank))
    return dict(arms=results, forbidden_calls=guard.donor_forward_calls)


def byte_size(skill):
    return sum(tensor.numel() * tensor.element_size() for tensor in skill.state_dict().values())


def encounter_prefix(bank, encounter):
    """Use the Phase 4 novelty check once for the identical initial banks."""
    world = OpenWorldSkillBank(bank)
    probes = []
    started = time.perf_counter()
    for blocks in range(phase4.FIRST_PROBE_BLOCKS, phase4.MAX_ENCOUNTER_BLOCKS + 1):
        observed = encounter[:blocks].reshape(1, -1)
        triggered = bool(world.detect_novelty(observed))
        probes.append(dict(observed_bytes=observed.numel(), triggered=triggered))
        if triggered:
            break
    return observed, probes, time.perf_counter() - started


def negative_probes(world, original):
    raw = original.reshape(-1, 2 * phase3.BYTES)
    flags = world.detect_novelty(raw).view(len(phase3.DOMAINS), -1)
    return {name: int(row.sum()) for name, row in zip(phase3.DOMAINS, flags, strict=True)}


def memory_coverage(skill, novel):
    # Only the counted answer positions are used for this held-out diagnostic.
    cues = novel.unfold(1, 4, 4)[..., :SUFFIX_BYTES].int()
    keys = (cues[..., 0] << 16) | (cues[..., 1] << 8) | cues[..., 2]
    return dict(covered_answer_positions=int(torch.isin(keys, skill.keys).sum()), total_answer_positions=keys.numel())


def run_seed(seed):
    started = time.perf_counter()
    original_fit, suites, manifest = phase3.data_for(seed)
    original = suites["single"]["raw"]
    encounter, novel, novel_manifest = phase4.rule_c_data(seed, original_fit, original)
    raw = torch.cat((original, novel))
    labels = torch.arange(len(phase4.DOMAINS)).repeat_interleave(phase3.CHECK_WINDOWS)[:, None].expand_as(raw)
    suite = dict(raw=raw, labels=labels)

    with tempfile.TemporaryDirectory(prefix="nexus-phase5-") as temp:
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
            residents = phase4.skill_hashes(bank)
            resident_forecasts = bank.component_log_probs(original)
            false_births = negative_probes(OpenWorldSkillBank(bank), original)
            observed, probes, novelty_seconds = encounter_prefix(bank, encounter)
            birth_at = observed.numel() if probes[-1]["triggered"] else None

            fixed, _ = evaluate(bank, suite)
            results = {FIXED: dict(**fixed, steps=0, birth_wall_seconds=0.0, skill_payload_bytes=0)}
            hot = {}
            for arm in BIRTH_ARMS:
                world = OpenWorldSkillBank(copy.deepcopy(bank))
                start = time.perf_counter()
                skill = world.spawn_and_adapt_skill(NOVEL, observed, mode=arm) if birth_at is not None else None
                duration = time.perf_counter() - start if skill is not None else 0.0
                _, output = evaluate(world.bank, suite)
                original_after = world.bank.component_log_probs(original)[:, :, :len(residents)]
                original_hashes = {name: state_hash(world.bank.skills[name]) for name in residents}
                path = directory / f"{arm}.pt"
                world.save_bank(path)
                record = dict(
                    steps=BIRTH_STEPS[arm] if skill is not None else 0, birth_wall_seconds=duration,
                    skill_payload_bytes=byte_size(skill) if skill is not None else 0,
                    bank_snapshot_bytes=path.stat().st_size, bank_sha256=digest(path),
                    resident_weights_exact=residents == original_hashes,
                    resident_forecasts_exact=torch.equal(resident_forecasts, original_after),
                    identity=phase3.bank_identity(world.bank), output_sha256=output,
                )
                if arm == EPISODIC_BIRTH and skill is not None:
                    record.update(
                        suffix_entries=skill.keys.numel(),
                        suffix_payload_bytes=skill.keys.numel() * (skill.keys.element_size() + skill.successors.element_size()),
                        **memory_coverage(skill, novel),
                    )
                hot[arm] = record

            finetune = copy.deepcopy(bank)
            start = time.perf_counter()
            adapt_skill(finetune.base, observed, BIRTH_STEPS[CONSOLIDATED_BIRTH])
            duration = time.perf_counter() - start
            scores, _ = evaluate(finetune, suite)
            results[CONTINUAL] = dict(**scores, steps=BIRTH_STEPS[CONSOLIDATED_BIRTH],
                                      birth_wall_seconds=duration, skill_payload_bytes=byte_size(finetune.base))

        torch.save(suite, directory / "test.pt")
        cold = json.loads(subprocess.check_output([sys.executable, __file__, "--cold", temp], cwd=ROOT))

    for arm in BIRTH_ARMS:
        expected, actual = hot[arm], cold["arms"][arm]
        results[arm] = dict(
            **actual["scores"], **expected, cold_output_sha256=actual["output_sha256"],
            cold_restore_exact=expected["identity"] == actual["identity"]
            and expected["output_sha256"] == actual["output_sha256"],
        )
    for arm in ARMS:
        drift = {name: results[arm]["domains"][name]["bpb"] - fixed["domains"][name]["bpb"]
                 for name in phase3.DOMAINS}
        results[arm].update(retention_drift_bpb=drift, max_original_bpb_drift=max(drift.values()))
    return dict(
        seed=seed, birth_at_byte=birth_at, probes=probes, novelty_seconds=novelty_seconds,
        adaptation_bytes=observed.numel(), adaptation_sha256=phase3.digest(bytes(observed[0].tolist())),
        corpus_manifest=manifest, novel_manifest=novel_manifest, producer=producer, false_births=false_births,
        old_state_hashes=residents, forbidden_calls=guard.donor_forward_calls + cold["forbidden_calls"],
        elapsed_seconds=time.perf_counter() - started, **results,
    )


def science(rows):
    candidates = {}
    for arm in (EPISODIC_BIRTH, CONSOLIDATED_BIRTH):
        mastered = wins_phase4 = wins_fixed = routed = retained = better_than_continual = 0
        joint = mixed_joint = 0
        for row in rows:
            result = row[arm]
            novel = result["domains"][NOVEL]
            answer = novel["rule_accuracy_pct"]
            mastery = answer >= 75
            win_phase4 = answer > row[PHASE4_BIRTH]["domains"][NOVEL]["rule_accuracy_pct"]
            win_fixed = answer > row[FIXED]["domains"][NOVEL]["rule_accuracy_pct"]
            routing = row["birth_at_byte"] is not None and novel["routing_accuracy_pct"] >= 90
            retention = result["max_original_bpb_drift"] <= phase4.RETENTION_LIMIT_BPB
            beat_continual = result["original_bpb"] < row[CONTINUAL]["original_bpb"]
            mastered += mastery
            wins_phase4 += win_phase4
            wins_fixed += win_fixed
            routed += routing
            retained += retention
            better_than_continual += beat_continual
            joint += mastery and win_phase4 and win_fixed and routing and retention and beat_continual
            mixed_joint += win_phase4 and retention
        candidates[arm] = dict(
            mastery_at_least_75_pct=mastered, beats_phase4_on_answer_accuracy=wins_phase4,
            beats_fixed_on_answer_accuracy=wins_fixed, cold_routing_at_least_90_pct=routed,
            within_per_domain_retention_bound=retained, beats_continual_on_original_bpb=better_than_continual,
            all_pass_criteria=joint, mixed_comparison_wins=mixed_joint,
        )
    passed = any(entry["all_pass_criteria"] == len(SEEDS) for entry in candidates.values())
    mixed = any(entry["within_per_domain_retention_bound"] == len(SEEDS)
                and entry["mixed_comparison_wins"] >= 4
                and entry["mastery_at_least_75_pct"] < len(SEEDS) for entry in candidates.values())
    return dict(verdict="PASS" if passed else "MIXED" if mixed else "FAIL", candidates=candidates)


def run(destination):
    if destination.exists():
        raise FileExistsError(destination)
    source = phase3.git("rev-parse", "HEAD").decode().strip()
    tree = phase3.git("rev-parse", "HEAD^{tree}").decode().strip()
    if phase3.git("status", "--porcelain") or phase3.git("rev-parse", "HEAD^").decode().strip() != BASE:
        raise ValueError("execute on the clean Phase 5 preregistration commit")
    started = time.perf_counter()
    rows = []
    for seed in SEEDS:
        row = run_seed(seed)
        rows.append(row)
        print(f"seed {seed} complete, birth_at={row['birth_at_byte']}, "
              f"cold={all(row[arm]['cold_restore_exact'] for arm in BIRTH_ARMS)}", flush=True)
    if phase3.git("status", "--porcelain") or phase3.git("rev-parse", "HEAD").decode().strip() != source:
        raise ValueError("preregistered checkout changed during scoring")
    atomic(destination, dict(
        schema="nexus-phase5-skill-mastery-dev-v1", source_sha=source, source_tree=tree, base_sha=BASE,
        fixture_source_sha=phase3.SOURCE, prereg_sha256=digest(ROOT / PREREG),
        platform=platform.platform(), python=platform.python_version(), torch=str(torch.__version__),
        numpy=np.__version__, torch_threads=torch.get_num_threads(),
        elapsed_seconds=time.perf_counter() - started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        protocol=dict(novelty_window=NOVELTY_WINDOW, novelty_threshold_bpb=NOVELTY_THRESHOLD_BPB,
                      suffix_bytes=SUFFIX_BYTES, memory_weight=MEMORY_WEIGHT, birth_steps=BIRTH_STEPS),
        receiver_checks=dict(
            seeds=len(rows), forbidden_calls=sum(row["forbidden_calls"] for row in rows),
            false_births=sum(sum(row["false_births"].values()) for row in rows),
            in_distribution_probes=len(rows) * len(phase3.DOMAINS) * (phase3.CHECK_WINDOWS // 2),
            exact_resident_preservation=sum(all(row[arm]["resident_weights_exact"] and row[arm]["resident_forecasts_exact"]
                                                for arm in BIRTH_ARMS) for row in rows),
            exact_cold_restores=sum(all(row[arm]["cold_restore_exact"] for arm in BIRTH_ARMS) for row in rows),
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
