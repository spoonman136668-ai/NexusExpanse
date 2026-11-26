"""Preregistered Phase 12 offline scaling confirmation."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.dev_artifacts import atomic, atomic_bytes, digest
from benchmarks.phase3_multi_skill_routing import SOURCE, git
from benchmarks.phase9_weight_edit_and_assimilate import combine, frozen_data
from benchmarks.phase10b_shard_stream_and_edit import (
    ADAPT_STEPS, checked_score, make_matched_scratch, produce_cross_width_donor,
)
from nexus_expanse.jepa import state_hash
from nexus_expanse.shard_stream_assimilation import adapt_skill, assimilate_zero_exec
from nexus_expanse.weight_editing import exact_skill_receipt, prefix_diff, restore_skill, verify_skill

BASE = "145c00226fbcfd285efac04fad5ff7b322d7c2fe"
PREREG = "docs/PHASE12_SCALED_ZERO_EXEC_PREREG.md"
SEEDS = tuple(range(2601, 2606))
ARMS = ("dim32_2L", "dim64_4L", "dim64_4L_engram", "shuffled_dim64_4L", "scratch_dim64_4L")
STAGES = ("zero_update", "post_adapt")
PREFIX_SEEDS = tuple(range(50))


@torch.inference_mode()
def skill_identity(archive):
    skill = restore_skill(archive)
    probe = torch.arange(24).unsqueeze(0)
    return dict(
        state_sha256=state_hash(skill),
        forecast_sha256=hashlib.sha256(skill(probe).contiguous().numpy().tobytes()).hexdigest(),
        prior_alpha=skill.prior_alpha,
    )


def cold_score(directory, seed):
    splits, hashes = frozen_data(seed)
    arms = {}
    for arm in ARMS:
        stages = {}
        for stage in STAGES:
            source = directory / f"{arm}-{stage}.pt"
            archive = source.read_bytes()
            bpb, calls = checked_score(archive, splits["check"])
            stages[stage] = dict(
                identity=skill_identity(archive), bpb=bpb, donor_forward_calls=calls,
                archive_sha256=digest(source),
            )
        arms[arm] = stages
    return dict(arms=arms, check_hashes=hashes["check"])


def child(mode, path, seed):
    output = subprocess.check_output(
        [sys.executable, str(Path(__file__).resolve()), mode, str(path), "--seed", str(seed)], text=True,
    )
    return json.loads(output)


def seed_run(root, seed):
    started = time.perf_counter()
    splits, hashes = frozen_data(seed)
    donor = root / "donor"
    producer = child("--produce", donor, seed)
    archives = root / "archives"
    archives.mkdir()
    arms = {}
    fit = combine(splits["fit"])
    for arm in ARMS:
        geometry = dict(student_dim=32, student_layers=2) if arm == "dim32_2L" else dict(
            student_dim=64, student_layers=4,
        )
        if arm == "scratch_dim64_4L":
            initial, initial_checks = verify_skill(make_matched_scratch(seed, **geometry))
        else:
            shuffled = arm == "shuffled_dim64_4L"
            initial, initial_checks = assimilate_zero_exec(
                donor, mock_pt=True, seed=seed, shuffle_seed=seed + 50000 if shuffled else None,
                use_engram=shuffled or arm == "dim64_4L_engram", **geometry,
            )
        student = restore_skill(initial)
        initial_diffs = [prefix_diff(student, seed=probe, probes=1) for probe in PREFIX_SEEDS]
        adapted, final_checks = adapt_skill(student, fit, seed=seed + 60000, steps=ADAPT_STEPS)
        final_skill = restore_skill(adapted)
        diffs = [max(before, prefix_diff(final_skill, seed=probe, probes=1))
                 for probe, before in zip(PREFIX_SEEDS, initial_diffs, strict=True)]
        warm = {}
        for stage, archive in zip(STAGES, (initial, adapted), strict=True):
            path = archives / f"{arm}-{stage}.pt"
            atomic_bytes(path, archive)
            warm[stage] = dict(identity=skill_identity(archive), archive_sha256=digest(path))
        arms[arm] = dict(initial=initial_checks, final=final_checks, probe_seed_diffs=diffs, warm=warm)
    cold = child("--cold", archives, seed)
    if cold["check_hashes"] != json.loads(json.dumps(hashes["check"])):
        raise ValueError("cold process used different CHECK bytes")
    for arm, receipt in arms.items():
        receipt["cold"] = cold["arms"][arm]
        receipt["cold_exact"] = all(
            all(receipt["cold"][stage][key] == value for key, value in record.items())
            for stage, record in receipt["warm"].items()
        )
    return dict(seed=seed, fixture=hashes, donor_producer=producer, arms=arms,
                seconds=time.perf_counter() - started)


def scaling_verdict(rows):
    controls = ("shuffled_dim64_4L", "scratch_dim64_4L")
    comparisons = {}
    for name in ("dim64_4L", "dim64_4L_engram"):
        both, baseline = 0, 0
        for row in rows:
            arms = row["arms"]

            def score(arm, stage):
                return arms[arm]["cold"][stage]["bpb"]["combined"]

            both += all(score(name, stage) < score(control, stage) for stage in STAGES for control in controls)
            baseline += score(name, "post_adapt") < score("dim32_2L", "post_adapt")
        comparisons[name] = dict(both_controls_both_stages=both, post_adapt_over_dim32=baseline)
    if any(row["both_controls_both_stages"] == 5 and row["post_adapt_over_dim32"] >= 4
           for row in comparisons.values()):
        status = "PASS"
    elif any(row["both_controls_both_stages"] >= 4 and row["post_adapt_over_dim32"] < 4
             for row in comparisons.values()):
        status = "MIXED"
    else:
        status = "FAIL"
    return dict(status=status, comparisons=comparisons)


def engineering(rows):
    cold_exact = 0
    peak_loaded = peak_disk = donor_calls = 0
    max_prefix = 0.0
    states_exact = True
    for row in rows:
        cold_exact += all(arm["cold_exact"] for arm in row["arms"].values())
        for name, arm in row["arms"].items():
            initial, final = arm["initial"], arm["final"]
            if name != "scratch_dim64_4L":
                states_exact &= initial["peak_loaded_shards"] == 1
                peak_loaded = max(peak_loaded, initial["peak_loaded_shards"])
                peak_disk = max(peak_disk, initial.get("peak_disk_shards", 0))
            donor_calls += sum(stage.get("donor_forward_calls", 0) for stage in (initial, final, *arm["cold"].values()))
            states_exact &= exact_skill_receipt(initial) and exact_skill_receipt(final)
            max_prefix = max(max_prefix, *arm["probe_seed_diffs"])
    passed = (states_exact and cold_exact == len(SEEDS) and donor_calls == 0 and peak_loaded == 1
              and peak_disk <= 1 and max_prefix == 0.0)
    return dict(
        scored_checks="PASS" if passed else "FAIL", exact_cold_seeds=cold_exact,
        peak_loaded_shards=peak_loaded, peak_disk_shards=peak_disk, donor_forward_calls=donor_calls,
        prefix_probe_seeds=len(PREFIX_SEEDS), max_prefix_diff=max_prefix,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    route = parser.add_mutually_exclusive_group(required=True)
    route.add_argument("--out", type=Path)
    route.add_argument("--produce", type=Path, help=argparse.SUPPRESS)
    route.add_argument("--cold", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.produce:
        data, _ = frozen_data(args.seed)
        result = produce_cross_width_donor(args.produce, combine(data["fit"]), args.seed,
                                          dim=128, layers=4, engram=True)
        print(json.dumps(result, allow_nan=False))
        return
    if args.cold:
        print(json.dumps(cold_score(args.cold, args.seed), allow_nan=False))
        return

    head = git("rev-parse", "HEAD").decode().strip()
    tree = git("rev-parse", "HEAD^{tree}").decode().strip()
    out = args.out.resolve()
    intent = out.with_suffix(".intent.json")
    if (out.is_relative_to(ROOT) or out.exists() or intent.exists() or git("status", "--porcelain")
            or git("rev-parse", "HEAD^").decode().strip() != BASE or not git("ls-tree", head, PREREG)):
        parser.error("run the scored check once from the clean first Phase 12 commit with a new external output")
    identity = dict(baseline=BASE, source_commit=head, source_tree=tree, prereg_sha256=digest(ROOT / PREREG))
    atomic(intent, identity)
    start = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix="nexus-phase12-") as directory:
        for seed in SEEDS:
            work = Path(directory) / str(seed)
            work.mkdir()
            rows.append(seed_run(work, seed))
            print(f"Phase 12 seed {seed} complete", flush=True)
    if git("rev-parse", "HEAD").decode().strip() != head or git("status", "--porcelain"):
        parser.error("the repository changed during the scored check")
    result = dict(
        schema="nexus.phase12-scaled-zero-exec.v1", **identity, fixture_source_sha=SOURCE,
        python=platform.python_version(), platform=platform.platform(), torch=str(torch.__version__), device="cpu",
        external_downloads=0, paid_api_calls=0, seeds=rows, engineering=engineering(rows),
        scientific=scaling_verdict(rows), wall_seconds=time.perf_counter() - start,
        max_rss_kib=max(resource.getrusage(kind).ru_maxrss for kind in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)),
    )
    atomic(out, result)
    print(json.dumps(dict(engineering=result["engineering"], scientific=result["scientific"]), allow_nan=False))


if __name__ == "__main__":
    main()
