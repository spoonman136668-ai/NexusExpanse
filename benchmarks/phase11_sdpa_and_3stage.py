"""Frozen CPU attention profile and shard-assimilation, repair, and birth composition."""

import argparse
import json
import math
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import phase3_multi_skill_routing as phase3
from benchmarks import phase4_skill_birth as phase4
from benchmarks.dev_artifacts import atomic, digest
from benchmarks.phase9_weight_edit_and_assimilate import combine, edit_skill_experiment, frozen_data
from benchmarks.phase10b_shard_stream_and_edit import checked_score, make_matched_scratch, produce_cross_width_donor
from nexus_expanse.checkpoint_transport import load_skill
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.jepa import state_hash
from nexus_expanse.multi_skill_router import MultiSkillBank
from nexus_expanse.shard_stream_assimilation import adapt_skill, assimilate_zero_exec
from nexus_expanse.skill_birth import EPISODIC_BIRTH, OpenWorldSkillBank
from nexus_expanse.weight_editing import exact_skill_receipt, restore_skill, verify_skill

BASE = "a01f4386715fd59c75535961194677847d862622"
PREREG = "docs/PHASE11_SDPA_AND_3STAGE_PREREG.md"
SEEDS = tuple(range(2501, 2506))
SDPA_SEED = 2511
CONTEXTS = (512, 1024)
MATH_TOLERANCE = 1e-5
RETENTION_LIMIT = 0.015
ADAPT_STEPS = 64
RESIDENTS = phase3.DOMAINS
NOVEL = phase4.NOVEL


def make_sdpa_model(context, *, use_sdpa, dim=64):
    return NexusExpanseByteCore(
        dim=dim, n_layers=2, n_heads=4, ffn_mult=2, max_seq_len=context,
        dropout=0.0, enable_skip=False, use_sdpa=use_sdpa,
    )


@torch.inference_mode()
def core_prefix_diff(model, raw, cut):
    before = model.forward_neural(raw)[:2]
    altered = raw.clone()
    altered[:, cut:] = (altered[:, cut:] + 17) % 256
    after = model.forward_neural(altered)[:2]
    short = model.forward_neural(raw[:, :cut])[:2]
    return max(
        max((a[:, :cut] - b[:, :cut]).abs().max().item(), (a[:, :cut] - c).abs().max().item())
        for a, b, c in zip(before, after, short, strict=True)
    )


@torch.inference_mode()
def bank_prefix_diff(bank, raw, cut):
    route = bank.forward_hmm_bma_hierarchical
    baseline = route(raw, include_next=True)
    alternate = raw.clone()
    alternate[:, cut:] = (alternate[:, cut:] + 17) % 256
    changed = route(alternate, include_next=True)
    short = route(raw[:, :cut], include_next=True)
    size = cut + 1
    return max(
        max((a[:, :size] - b[:, :size]).abs().max().item(), (a[:, :size] - c).abs().max().item())
        for a, b, c in zip(baseline, changed, short, strict=True)
    )


def training_pass(model, inputs, targets):
    model.zero_grad(set_to_none=True)
    start = time.perf_counter()
    logits, hidden = model.forward_neural(inputs)[:2]
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    mid = time.perf_counter()
    loss.backward()
    return logits.detach(), hidden.detach(), (mid - start, time.perf_counter() - mid)


def profile_arm(model, inputs, targets, memory_path):
    model.train()
    for _ in range(2):
        training_pass(model, inputs, targets)
    timings = [training_pass(model, inputs, targets)[2] for _ in range(7)]
    model.zero_grad(set_to_none=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True,
    ) as trace:
        training_pass(model, inputs, targets)
    trace.export_chrome_trace(str(memory_path))
    events = json.loads(memory_path.read_bytes())["traceEvents"]
    allocations = [event["args"]["Total Allocated"] for event in events
                   if event.get("name") == "[memory]" and event["args"]["Device Id"] == -1]
    return dict(
        samples_seconds=[dict(forward=f, backward=b) for f, b in timings],
        forward_median_seconds=statistics.median(f for f, _ in timings),
        backward_median_seconds=statistics.median(b for _, b in timings),
        forward_backward_median_seconds=statistics.median(f + b for f, b in timings),
        peak_cpu_tensor_allocated_bytes=max(allocations),
    )


def attention_experiment(directory):
    rows = []
    for context in CONTEXTS:
        torch.manual_seed(SDPA_SEED)
        legacy = make_sdpa_model(context, use_sdpa=False)
        sdpa = make_sdpa_model(context, use_sdpa=True)
        sdpa.load_state_dict(legacy.state_dict())
        raw = torch.randint(256, (2, context + 1), generator=torch.Generator().manual_seed(SDPA_SEED + context))
        inputs, targets = raw[:, :-1].contiguous(), raw[:, 1:].contiguous()
        legacy_logits, legacy_hidden, _ = training_pass(legacy, inputs, targets)
        sdpa_logits, sdpa_hidden, _ = training_pass(sdpa, inputs, targets)
        gradients = []
        for a, b in zip(legacy.parameters(), sdpa.parameters(), strict=True):
            if a.grad is None and b.grad is None:
                continue
            if a.grad is None or b.grad is None:
                raise RuntimeError("attention paths disagree about which weights receive gradients")
            gradients.append((a.grad - b.grad).abs().max().item())
        parity = dict(
            training_logits_max_abs_diff=(legacy_logits - sdpa_logits).abs().max().item(),
            training_hidden_max_abs_diff=(legacy_hidden - sdpa_hidden).abs().max().item(),
            training_gradient_max_abs_diff=max(gradients),
        )
        del legacy_logits, legacy_hidden, sdpa_logits, sdpa_hidden
        arms = {
            "legacy": profile_arm(legacy, inputs, targets, directory / f"legacy-{context}.json"),
            "sdpa": profile_arm(sdpa, inputs, targets, directory / f"sdpa-{context}.json"),
        }
        legacy.eval()
        sdpa.eval()
        with torch.inference_mode():
            a, a_hidden = legacy.forward_neural(inputs)[:2]
            b, b_hidden = sdpa.forward_neural(inputs)[:2]
            eval_diff = dict(native_eval_logits_max_abs_diff=(a - b).abs().max().item(),
                             native_eval_hidden_max_abs_diff=(a_hidden - b_hidden).abs().max().item())
        prefix = core_prefix_diff(sdpa, inputs, context // 2)
        rows.append(dict(context=context, dim=64, batch=2, seed=SDPA_SEED, parity=parity,
                         native_eval_diagnostics=eval_diff, prefix_max_abs_diff=prefix, arms=arms,
                         passes=max(parity.values()) <= MATH_TOLERANCE and prefix == 0.0))
    return rows


def assimilation_stage(donor, seed, data):
    imported, first = assimilate_zero_exec(donor, mock_pt=True, seed=seed)
    scratch, scratch_initial = verify_skill(make_matched_scratch(seed))
    results = {}
    last = None
    for name, archive, initial in (("assimilated", imported, first), ("scratch", scratch, scratch_initial)):
        work = restore_skill(archive)
        adapted, final = adapt_skill(work, combine(data["fit"]), seed=seed + 60000, steps=ADAPT_STEPS)
        zero, calls_zero = checked_score(archive, data["check"])
        test, calls_after = checked_score(adapted, data["check"])
        if name == "assimilated":
            last = adapted
        results[name] = dict(initial=initial, final=final, zero_update_bpb=zero, held_bpb=test,
                             cold_scoring_donor_calls=calls_zero + calls_after)
    return last, results


def resident_data(seed, text):
    # Use the Phase 3 rules and selection budgets. Text uses precisely the Phase 11 splits.
    original_fit, suites, _ = phase3.data_for(seed)
    fit, resident = {}, {}
    for index, name in enumerate(RESIDENTS):
        if name in text["fit"]:
            fit[name] = text["fit"][name]
            resident[name] = text["check"][name]
        else:
            fit[name] = original_fit[name]
            resident[name] = suites["single"]["raw"][index * phase3.CHECK_WINDOWS:(index + 1) * phase3.CHECK_WINDOWS]
    return fit, resident, {k: phase3.digest(value.to(torch.uint8).numpy().tobytes()) for k, value in resident.items()}


def domain_bpb(raw, logp):
    return -logp.gather(-1, raw.unsqueeze(-1)).squeeze(-1).double().mean(1) / math.log(2)


def evaluation(bank, resident, novel):
    size = sum(len(windows) for windows in resident.values())
    raw = torch.cat([*resident.values(), novel])
    logp, weights = bank.forward_hmm_bma_hierarchical(raw)
    components = bank.component_log_probs(resident[RESIDENTS[0]], selected=torch.zeros_like(resident[RESIDENTS[0]]))
    losses = domain_bpb(raw, logp)
    rows = [len(resident[name]) for name in RESIDENTS] + [len(novel)]
    domains = {name: score.tolist() for name, score in zip((*RESIDENTS, NOVEL), losses.split(rows), strict=True)}
    answer = logp[size:, 3::4].argmax(-1) == novel[:, 3::4]
    active = weights[size:].argmax(-1)
    index = bank.skill_names.index(NOVEL) if NOVEL in bank.skills else -1
    return dict(
        domain_window_bpb=domains,
        novel_answer_accuracy_pct=phase3.accuracy(answer), novel_cold_routing_pct=phase3.accuracy(active == index),
        output_sha256=phase3.output_digest(logp, weights, 0),
        resident_direct_output_sha256=phase3.digest(components.contiguous().numpy().tobytes()),
        bank_identity=dict(phase3.bank_identity(bank), resident_names=list(bank.resident_names)),
    )


def cold_restore(directory):
    inputs = torch.load(directory / "holdout.pt", map_location="cpu", weights_only=True)
    with forbid_donor_forward() as guard:
        restored = OpenWorldSkillBank.load_bank(directory / "expanded.pt")
        scores = evaluation(restored.bank, **inputs)
        probe = inputs["novel"][:1, :24]
        scores["prefix_max_abs_diff"] = bank_prefix_diff(restored.bank, probe, 12)
        scores["donor_forward_calls"] = guard.donor_forward_calls
    return scores


def birth_stage(directory, seed, edited, data, producer):
    fit, resident, digests = resident_data(seed, data)
    novel_fit, novel, novel_provenance = phase4.rule_c_data(seed, fit, torch.cat(list(resident.values())))
    for name, record in producer.items():
        if digest(directory / f"{name}.pt") != record["source_sha256"]:
            raise ValueError("resident checkpoint custody mismatch")
    with forbid_donor_forward() as guard:
        others = {name: load_skill(directory / f"{name}.pt", "residual8") for name in RESIDENTS[1:]}
        bank = MultiSkillBank(edited, others)
        world = OpenWorldSkillBank(bank)
        before_hashes = phase4.skill_hashes(bank)
        before = evaluation(bank, resident, novel)
        false = {name: int(world.detect_novelty(block.reshape(-1, 2 * phase3.BYTES)).sum())
                 for name, block in resident.items()}
        probes = []
        born = None
        for blocks in range(phase4.FIRST_PROBE_BLOCKS, phase4.MAX_ENCOUNTER_BLOCKS + 1):
            observed = novel_fit[:blocks].reshape(1, -1)
            born = world.observe(NOVEL, observed, mode=EPISODIC_BIRTH)
            probes.append(dict(bytes=observed.numel(), triggered=born is not None))
            if born is not None:
                break

        after_hashes = phase4.skill_hashes(bank)
        warm = evaluation(bank, resident, novel)
        world.save_bank(directory / "expanded.pt")
        calls = guard.donor_forward_calls
    torch.save(dict(resident=resident, novel=novel), directory / "holdout.pt")
    cold = json.loads(subprocess.check_output(
        [sys.executable, __file__, "--cold", str(directory)], cwd=ROOT, text=True,
    ))
    prefix = cold.pop("prefix_max_abs_diff")
    cold_calls = cold.pop("donor_forward_calls")
    drift = {
        name: statistics.mean(after - before for after, before in zip(
            cold["domain_window_bpb"][name], before["domain_window_bpb"][name], strict=True))
        for name in RESIDENTS
    }
    base = RESIDENTS[0]
    base_window_drift = max(
        after - before for after, before in zip(
            cold["domain_window_bpb"][base], before["domain_window_bpb"][base], strict=True,
        )
    )
    direct_exact = cold["resident_direct_output_sha256"] == before["resident_direct_output_sha256"]
    weights_exact = all(after_hashes[name] == before_hashes[name] for name in RESIDENTS)
    return dict(
        born=born is not None, bank_size=len(bank.skills), birth_probes=probes, false_births_by_domain=false,
        false_births=sum(false.values()), resident_negative_probes=sum(len(block) // 2 for block in resident.values()),
        resident_weights_exact=weights_exact, repaired_resident_forecasts_exact=direct_exact,
        pre_birth=before, cold=cold, warm_output_sha256=warm["output_sha256"],
        cold_restore_exact=cold == warm, donor_forward_calls=calls + cold_calls, prefix_max_abs_diff=prefix,
        resident_domain_drift_bpb=drift, repaired_resident_max_window_drift_bpb=base_window_drift,
        max_retention_drift_bpb=max(0.0, base_window_drift, *drift.values()), resident_check_sha256=digests,
        novel_provenance=novel_provenance, resident_snapshot_sha256=digest(directory / "expanded.pt"),
    )


def seed_run(directory, seed):
    started = time.perf_counter()
    data, source = frozen_data(seed)
    fit, _, _ = resident_data(seed, data)
    torch.save(fit, directory / "fit.pt")
    generated = json.loads(subprocess.check_output(
        [sys.executable, __file__, "--produce", str(directory), "--seed", str(seed)], cwd=ROOT, text=True,
    ))
    archive, first = assimilation_stage(directory / "donor", seed, data)
    donor = first["assimilated"]["initial"]
    expected = {key: value for key, value in generated["shards"]["files"].items() if key.endswith(".pt")}
    if donor["loaded_shard_count"] != 3 or donor["shards"] != expected:
        raise ValueError("assimilated donor shard provenance does not match the producer")
    clean = restore_skill(archive)
    before_edit = state_hash(clean)
    edit, edited = edit_skill_experiment(clean, seed, data)
    edited_archive, edit_verification = verify_skill(edited)
    resident = restore_skill(edited_archive)
    birth = birth_stage(directory, seed, resident, data, generated["residents"])
    return dict(
        seed=seed, split_provenance=source, producer=generated, assimilation=first, edit=edit, birth=birth,
        edit_cold_verification=edit_verification, lifecycle_hashes=dict(
            assimilated_sha256=before_edit, repaired_sha256=state_hash(resident),
        ), seconds=time.perf_counter() - started,
    )


def write_donors(directory, seed):
    fit = torch.load(directory / "fit.pt", map_location="cpu", weights_only=True)
    produced = produce_cross_width_donor(directory / "donor", combine(fit), seed)
    phase3.produce(directory, seed, domains=RESIDENTS[1:])
    return dict(shards=produced, residents=json.loads((directory / "producer.json").read_bytes()))


def verdict(rows):
    first = second = third = all_pass = cold = false_births = donor_calls = edit_autograd = 0
    violations = 0
    for row in rows:
        student, scratch = (row["assimilation"][name] for name in ("assimilated", "scratch"))
        svd, random, fault = (row["edit"]["arms"][name] for name in (
            "svd_subspace_ablate", "random_subspace_ablate", "unrepaired_faulted",
        ))
        birth = row["birth"]
        calls = sum(arm["initial"].get("donor_forward_calls", 0) + arm["final"]["donor_forward_calls"]
                    + arm["cold_scoring_donor_calls"] for arm in row["assimilation"].values())
        calls += row["edit"]["donor_forward_calls"] + birth["donor_forward_calls"]
        donor_calls += calls
        edit_autograd += row["edit"]["autograd_calls"]
        false_births += birth["false_births"]
        stage1 = student["held_bpb"]["combined"] < scratch["held_bpb"]["combined"] and calls == 0
        retain_delta = svd["retain_bpb"] - fault["retain_bpb"]
        stage2 = (svd["accepted"] and svd["fraction_damage_recovered"] is not None
                  and svd["fraction_damage_recovered"] >= 0.5 and svd["fault_bpb"] < random["fault_bpb"]
                  and retain_delta <= 0.02 and row["edit"]["autograd_calls"] == 0)
        exact = (exact_skill_receipt(student["initial"]) and exact_skill_receipt(student["final"])
                 and exact_skill_receipt(row["edit_cold_verification"]) and birth["cold_restore_exact"])
        cold += exact
        stage3 = (birth["born"] and birth["bank_size"] == 5
                  and birth["cold"]["novel_answer_accuracy_pct"] >= 85
                  and birth["cold"]["novel_cold_routing_pct"] >= 90
                  and birth["max_retention_drift_bpb"] <= RETENTION_LIMIT and birth["false_births"] == 0
                  and birth["resident_weights_exact"] and birth["repaired_resident_forecasts_exact"] and exact)
        violations += (retain_delta > 0.02 or birth["max_retention_drift_bpb"] > RETENTION_LIMIT
                       or birth["prefix_max_abs_diff"] != 0.0 or not exact or not birth["born"]
                       or not svd["accepted"] or not birth["resident_weights_exact"]
                       or not birth["repaired_resident_forecasts_exact"])
        first += stage1
        second += stage2
        third += stage3
        all_pass += stage1 and stage2 and stage3
    scientific = "PASS" if not violations and all_pass == 5 else "MIXED" if not violations and all_pass >= 4 else "FAIL"
    return dict(scientific=scientific, stage1_passes=first, stage2_passes=second, stage3_passes=third,
                all_stages_passes=all_pass, integrity_or_retention_violations=violations, exact_cold_restores=cold,
                false_births=false_births, guarded_donor_forward_calls=donor_calls, edit_autograd_calls=edit_autograd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path)
    mode.add_argument("--produce", type=Path, help=argparse.SUPPRESS)
    mode.add_argument("--cold", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.produce:
        print(json.dumps(write_donors(args.produce, args.seed), allow_nan=False))
        return
    if args.cold:
        print(json.dumps(cold_restore(args.cold), allow_nan=False))
        return

    head = phase3.git("rev-parse", "HEAD").decode().strip()
    if (args.out.exists() or phase3.git("rev-parse", "HEAD^").decode().strip() != BASE
            or phase3.git("status", "--porcelain") or not phase3.git("ls-tree", head, PREREG)):
        parser.error("run scored confirmation once from the clean first Phase 11 commit")
    tree = phase3.git("rev-parse", "HEAD^{tree}").decode().strip()
    start = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix="nexus-phase11-") as temp:
        directory = Path(temp)
        suite_a = attention_experiment(directory)
        print("Phase 11 attention profiles recorded", flush=True)
        for seed in SEEDS:
            scratch = directory / str(seed)
            scratch.mkdir()
            rows.append(seed_run(scratch, seed))
            print(f"Phase 11 composition seed {seed} recorded", flush=True)
    if phase3.git("rev-parse", "HEAD").decode().strip() != head or phase3.git("status", "--porcelain"):
        parser.error("checkout changed during scored confirmation")
    summary = verdict(rows)
    result = dict(
        schema="nexus.phase11-sdpa-and-3stage.v1", baseline=BASE, source_commit=head, source_tree=tree,
        prereg_sha256=digest(ROOT / PREREG), fixture_source_sha=phase3.SOURCE,
        platform=platform.platform(), device="cpu", torch=str(torch.__version__), python=platform.python_version(),
        sdpa_contexts=suite_a, sdpa_profile_pass=all(row["passes"] for row in suite_a),
        summary=summary, seeds=rows, external_downloads=0, paid_api_calls=0,
        peak_loaded_shards=max(row["assimilation"]["assimilated"]["initial"]["peak_loaded_shards"] for row in rows),
        resident_negative_probe_count=sum(row["birth"]["resident_negative_probes"] for row in rows),
        wall_seconds=time.perf_counter() - start,
        max_rss_kib=max(resource.getrusage(kind).ru_maxrss for kind in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)),
    )
    atomic(args.out, result)
    print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
