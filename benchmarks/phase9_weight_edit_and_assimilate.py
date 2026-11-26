"""Frozen Phase 9 Linux CPU DEV benchmark for a known rank-one fault and offline distillation."""

import argparse
from contextlib import ExitStack
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
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import phase3_multi_skill_routing as phase3
from benchmarks.checkpoint_transfer_study import optim, setup_donor, source_pack, update
from benchmarks.dev_artifacts import atomic, digest
from benchmarks.phase7_hard_novelty_sparse_k7 import VAL_TEXT_PATHS
from nexus_expanse.checkpoint_transport import load_skill
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.weight_editing import (
    apply_surgical_edit, exact_skill_receipt, inspect_fault_direction,
    restore_skill, verified_surgical_edit, verify_skill,
)

BASE = "deb473f70bdbdfc6d25192a3140c06d75d5a6d6e"
PREREG = "docs/PHASE9_ASSIMILATE_AND_WEIGHT_EDIT_PREREG.md"
SEEDS = tuple(range(2201, 2206))
DOMAINS = ("base_python", "markdown_prose")
LAYER = 1
TARGET = "attn.out_proj"
ALPHAS = (0.25, 0.5, 0.75, 1.0)
INJECTION_SCALE = 6.0
FIT_COUNT = 96
HELD_COUNT = 24
DONOR_STEPS = 256
STUDENT_STEPS = 192
KL_WEIGHT = 0.9


def frozen_data(seed):
    rng = random.Random(seed + 90000)
    used = set()
    splits = {split: {} for split in ("fit", "val", "check")}
    hashes = {split: {} for split in splits}
    for split in splits:
        for idx, domain in zip((0, 3), DOMAINS, strict=True):
            fit_paths, check_paths = phase3.TEXT_PATHS[idx]
            paths = {"fit": fit_paths, "val": VAL_TEXT_PATHS[domain], "check": check_paths}[split]
            windows = phase3.select_windows(
                phase3.text_windows(paths, rng), FIT_COUNT if split == "fit" else HELD_COUNT, used,
            )
            splits[split][domain] = torch.tensor([list(raw) for raw in windows], dtype=torch.long)
            hashes[split][domain] = dict(
                sha256=phase3.digest(b"".join(windows)), files=paths, windows=len(windows),
            )
    return splits, hashes


def combine(split):
    return torch.cat([split[name] for name in DOMAINS])


def produce_donor(path, seed, fit):
    start = time.perf_counter()
    model = setup_donor(seed + 10000)
    generator = torch.Generator().manual_seed(seed + 20000)
    byte_to_row = torch.randperm(256, generator=generator)
    optimizer = optim(model)
    for _ in range(DONOR_STEPS):
        batch = fit[torch.randint(len(fit), (8,), generator=generator)]
        logits = model(byte_to_row[batch[:, :-1]])[..., byte_to_row]
        update(model, optimizer, logits, batch[:, 1:])
    source_pack(path, model, byte_to_row)
    return dict(steps=DONOR_STEPS, seconds=time.perf_counter() - start,
                sha256=digest(path), bytes=path.stat().st_size, forward_calls=model.forward_calls)


def cli(*args):
    started = time.perf_counter()
    stdout = subprocess.check_output([sys.executable, *map(str, args)], cwd=ROOT, text=True)
    return json.loads(stdout), time.perf_counter() - started


@torch.inference_mode()
def bpb(skill, raw):
    logp = skill(raw[:, :-1]).double().log_softmax(-1)
    return float(-logp.gather(-1, raw[:, 1:, None]).mean() / math.log(2))


def unit(x):
    return x / x.norm()


def search(faulted, direction, val):
    def fault(skill):
        return bpb(skill, val[DOMAINS[0]])

    def retain(skill):
        return bpb(skill, val[DOMAINS[1]])

    candidates = []
    best = faulted
    best_receipt = None
    for alpha in ALPHAS:
        accepted, candidate, receipt = verified_surgical_edit(
            faulted, lambda skill: apply_surgical_edit(
                skill, LAYER, TARGET, "subspace_ablate", direction, alpha,
            ), fault, retain,
        )
        receipt["alpha"] = alpha
        candidates.append(receipt)
        if accepted and (best_receipt is None
                         or receipt["fault_after_bpb"] < best_receipt["fault_after_bpb"]):
            best = candidate
            best_receipt = receipt
    return best, dict(accepted=best_receipt is not None,
                      alpha=None if best_receipt is None else best_receipt["alpha"],
                      validation=candidates)


@torch.inference_mode()
def edit_experiment(donor_path, seed, data):
    clean = load_skill(donor_path, "residual8").eval().requires_grad_(False)
    clean.core.fixed_eval_geometry = True
    receipt, _ = edit_skill_experiment(clean, seed, data)
    return receipt


@torch.inference_mode()
def edit_skill_experiment(clean, seed, data):
    """Run the frozen fault and matched repair arms on an existing native receiver."""
    blocked = 0

    def forbidden(*args, **kwargs):
        nonlocal blocked
        blocked += 1
        raise RuntimeError("autograd invoked during the inference-only editing experiment")

    with forbid_donor_forward() as guard, ExitStack() as stack:
        stack.enter_context(patch.object(torch.autograd, "backward", forbidden))
        stack.enter_context(patch.object(torch.autograd, "grad", forbidden))
        stack.enter_context(patch.object(torch.Tensor, "backward", forbidden))
        fit, val, test = (data[name] for name in ("fit", "val", "check"))
        fit_direction = inspect_fault_direction(clean, fit[DOMAINS[0]], fit[DOMAINS[1]], LAYER)
        v = fit_direction["direction_v"]
        fault_mean = torch.tensor(fit_direction["layers"][LAYER]["fault_mean"], dtype=v.dtype)
        v = v if fault_mean.dot(v) >= 0 else -v
        readout = clean.readout.weight
        u = unit(readout[32] - readout.mean(dim=0))
        faulted = apply_surgical_edit(clean, LAYER, TARGET, "rank1_steer", v, INJECTION_SCALE, u)
        validation = inspect_fault_direction(faulted, val[DOMAINS[0]], val[DOMAINS[1]], LAYER)
        svd_direction = validation["direction_v"]
        random_direction = unit(torch.randn(v.shape, generator=torch.Generator().manual_seed(seed + 30000)))
        svd_skill, svd = search(faulted, svd_direction, val)
        random_skill, random_arm = search(faulted, random_direction, val)
        arms = {
            "clean_reference": clean, "unrepaired_faulted": faulted,
            "svd_subspace_ablate": svd_skill, "random_subspace_ablate": random_skill,
        }
        metrics = {
            name: dict(fault_bpb=bpb(model, test[DOMAINS[0]]), retain_bpb=bpb(model, test[DOMAINS[1]]))
            for name, model in arms.items()
        }
        all_frozen = all(
            not parameter.requires_grad and parameter.grad is None
            for model in arms.values() for parameter in model.parameters()
        )
        fault = metrics["unrepaired_faulted"]
        reference = metrics["clean_reference"]
        damage = fault["fault_bpb"] - reference["fault_bpb"]
        for name in ("svd_subspace_ablate", "random_subspace_ablate"):
            row = metrics[name]
            row["retain_drift_bpb"] = abs(row["retain_bpb"] - fault["retain_bpb"])
            row["retain_difference_from_clean_bpb"] = row["retain_bpb"] - reference["retain_bpb"]
            row["fraction_damage_recovered"] = (
                (fault["fault_bpb"] - row["fault_bpb"]) / damage if damage > 0 else None
            )
            _, row["verification"] = verify_skill(arms[name])
        metrics["svd_subspace_ablate"].update(svd)
        metrics["random_subspace_ablate"].update(random_arm)
        return dict(
            arms=metrics, injected_fault_damage_bpb=damage,
            injected_retain_difference_bpb=fault["retain_bpb"] - reference["retain_bpb"],
            fit_val_direction_abs_cosine=abs(float(torch.dot(v, svd_direction))),
            validation_singular_values=validation["layers"][LAYER]["singular_values"],
            donor_forward_calls=guard.donor_forward_calls, autograd_calls=blocked, all_parameters_frozen=all_frozen,
        ), svd_skill


def assimilate_experiment(directory, donor, seed, data):
    raw = directory / "fit.raw"
    raw.write_bytes(combine(data["fit"]).to(torch.uint8).numpy().tobytes())
    cache = directory / "cache"
    exporter, export_seconds = cli(
        ROOT / "scripts/export_donor_cache.py", "--backend", "nexus_donor", "--model", donor,
        "--input", raw, "--out-dir", cache,
    )
    outputs = {}
    for name, kl_weight in (("assimilated", KL_WEIGHT), ("scratch", 0.0)):
        out = directory / f"{name}.pt"
        receipt, elapsed = cli(
            "-m", "nexus_expanse.cli", "assimilate", "--cache-dir", cache, "--out", out,
            "--device", "cpu", "--dtype", "fp32", "--steps", STUDENT_STEPS,
            "--kl-weight", kl_weight, "--seed", seed,
        )
        with forbid_donor_forward() as guard:
            restored = restore_skill(out)
            check = data["check"]
            scores = {domain: bpb(restored, check[domain]) for domain in DOMAINS}
            outputs[name] = dict(
                receipt=receipt, seconds=elapsed, bpb=bpb(restored, combine(check)), by_domain=scores,
                restore_donor_forward_calls=guard.donor_forward_calls,
            )
    return dict(cache=exporter, export_seconds=export_seconds, **outputs)


def score(evidence_rows):
    superiority = retention = recovery = accepted = distillation = cold = 0
    for row in evidence_rows:
        edit, students = row["edit"], row["assimilation"]
        arms = edit["arms"]
        svd = arms["svd_subspace_ablate"]
        random_arm = arms["random_subspace_ablate"]
        fault = arms["unrepaired_faulted"]
        superiority += svd["fault_bpb"] < min(random_arm["fault_bpb"], fault["fault_bpb"])
        retention += svd["retain_drift_bpb"] <= 0.02
        recovery += svd["fraction_damage_recovered"] is not None and svd["fraction_damage_recovered"] >= 0.5
        accepted += svd["accepted"]
        assimilated = students["assimilated"]
        distillation += assimilated["bpb"] < students["scratch"]["bpb"] and assimilated["receipt"]["donor_forward_calls"] == 0
        cold += all(exact_skill_receipt(receipt) for receipt in (
            svd["verification"], random_arm["verification"],
            assimilated["receipt"], students["scratch"]["receipt"],
        ))
    if superiority == retention == recovery == accepted == distillation == 5:
        verdict = "PASS"
    elif superiority >= 4 and retention == 5:
        verdict = "MIXED"
    else:
        verdict = "FAIL"
    return dict(
        scientific=verdict, svd_beats_both_seeds=superiority, svd_within_held_retain_bound_seeds=retention,
        svd_half_damage_recovery_seeds=recovery, svd_accepted_seeds=accepted,
        assimilated_beats_scratch_seeds=distillation, exact_cold_restore_seeds=cold,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("the scored receipt already exists")
    head = phase3.git("rev-parse", "HEAD").decode().strip()
    if (phase3.git("rev-parse", "HEAD^").decode().strip() != BASE
            or phase3.git("status", "--porcelain")
            or not phase3.git("ls-tree", head, PREREG)):
        parser.error("run scored confirmation on the clean first Phase 9 commit")
    tree = phase3.git("rev-parse", "HEAD^{tree}").decode().strip()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix="nexus-phase9-") as scratch:
        for seed in SEEDS:
            seed_started = time.perf_counter()
            directory = Path(scratch) / str(seed)
            directory.mkdir()
            splits, hashes = frozen_data(seed)
            checkpoint = directory / "trained.pt"
            donor = produce_donor(checkpoint, seed, combine(splits["fit"]))
            edit = edit_experiment(checkpoint, seed, splits)
            assimilation = assimilate_experiment(directory, checkpoint, seed, splits)
            rows.append(dict(seed=seed, split_provenance=hashes, donor=donor, edit=edit,
                             assimilation=assimilation, seconds=time.perf_counter() - seed_started))
            print(f"Phase 9 scored seed {seed} recorded", flush=True)
    if phase3.git("rev-parse", "HEAD").decode().strip() != head or phase3.git("status", "--porcelain"):
        parser.error("checkout changed during scored confirmation")
    summary = score(rows)
    result = dict(
        schema="nexus.phase9-weight-edit-and-assimilate.v1", source_commit=head, source_tree=tree,
        baseline=BASE, fixture_source_sha=phase3.SOURCE, prereg_sha256=digest(ROOT / PREREG),
        platform=platform.platform(), python=platform.python_version(), torch=str(torch.__version__),
        device="cpu", threads=torch.get_num_threads(), autograd_calls=sum(row["edit"]["autograd_calls"] for row in rows),
        guarded_donor_forward_calls=sum(
            row["edit"]["donor_forward_calls"] + sum(
                row["assimilation"][arm]["receipt"]["donor_forward_calls"]
                + row["assimilation"][arm]["restore_donor_forward_calls"]
                for arm in ("assimilated", "scratch")
            ) for row in rows
        ),
        frozen_edit_parameters_all_seeds=all(row["edit"]["all_parameters_frozen"] for row in rows),
        fit_val_check_32byte_overlap=0, external_downloads=0, paid_api_calls=0,
        wall_seconds=time.perf_counter() - started,
        max_rss_kib=max(resource.getrusage(mode).ru_maxrss for mode in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)),
        seeds=rows, summary=summary,
    )
    atomic(args.out, result)
    print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
