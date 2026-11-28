"""Five-seed CPU comparison of verifier-gated and matched greedy fine-tuning."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import tempfile
import time
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import gate17_gpu_scaling_harness as scaling
from benchmarks.dev_artifacts import atomic, atomic_bytes, digest
from benchmarks.gate_custody import peak_rss, update
from benchmarks.phase20_verified_search import BATCH, CONTEXT, STEPS, train
from benchmarks.phase21_livebench_scoreboard import fit_corpus, prefix_probe
from benchmarks.phase3_multi_skill_routing import git
from nexus_expanse.cli import snapshot_checks
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.jepa import state_hash
from nexus_expanse.livebench_eval import CATEGORIES, evaluate_budgets, load_tasks_jsonl
from nexus_expanse.self_distill import DELIMITER, distillation_partitions, harvest_verified_traces, train_task_rows, write_harvest
from nexus_expanse.unified import NexusExpanseUnified
from nexus_expanse.verified_search import Verification, VerifiedByteSearch
from scripts.prepare_gpu_corpus import assert_disjoint

BASE = "1f0210611506d21725a5b23ee8f714a2e05ec4e5"
PREREG = "docs/PHASE22_VERIFIED_SELF_DISTILLATION_PREREG.md"
SEEDS = tuple(range(3601, 3606))
BUDGETS = (1, 16)
ADAPT_STEPS = 24
LR = 0.001
CPU = torch.device("cpu")


def sha(raw):
    return sha256(raw).hexdigest()


def task_file(path, rows):
    atomic_bytes(path, b"".join((json.dumps(row, allow_nan=False) + "\n").encode() for row in rows))
    return load_tasks_jsonl(path)


def inputs(seed):
    fit, val, held = distillation_partitions(seed)
    training, calibration, _ = fit_corpus(fit, val, seed)
    held_raw = b"".join(row["turns"][0].encode()+answer+DELIMITER for row, answer in held)
    pools = [{row["turns"][0] for row, _ in rows} for rows in (fit, val, held)]
    assert len(set.union(*pools)) == sum(map(len, pools))
    assert_disjoint((training, calibration), {"held prompts and answers":held_raw})
    return train_task_rows(fit), held, training, calibration, held_raw


def recheck(payload, rows, tasks):
    """Re-verify written offsets against freshly parsed tasks and exact bytes."""
    cursor = 0
    for row in rows:
        task = tasks[row["question_id"]]
        trace = payload[cursor:cursor+row["length"]]
        split = row["prompt_bytes"]
        n = row["solution_bytes"]
        answer = trace[split:split+n]
        result = task.verify(answer)
        if (row["offset"] != cursor or trace != task.prompt + answer + DELIMITER
                or row["trace_sha256"] != sha(trace) or row["solution_sha256"] != sha(answer)
                or row["output_sha256"] != sha(payload) or not result.passed or result.passed_tests != result.total_tests
                or row["passed_tests"] != result.passed_tests or row["total_tests"] != result.total_tests
                or (task.mode == "arithmetic" and answer.partition(b"\n")[2].strip(b"\r\n"))):
            raise ValueError("written harvest failed independent replay")
        cursor += row["length"]
    if cursor != len(payload):
        raise ValueError("unexpected bytes outside harvest audit")
    return len(rows)


def greedy_control(engine, tasks, accepted, seed):
    payload, rows, calls = bytearray(), [], 0
    for i, task in enumerate(tasks):
        if task.id not in accepted:
            continue
        # Search's greedy byte choices are independent from this status. The
        # actual verifier is used only for reporting after the raw selection.
        search = engine.run(task.prompt, lambda _: Verification(False, 0, 0, None), candidates=1,
                            max_bytes=task.max_bytes, stop_bytes=task.stop, seed=seed+64*i)
        raw = search.selected.raw
        payload.extend(task.prompt + raw + DELIMITER)
        rows.append(dict(question_id=task.id, category=task.category, continuation_hex=raw.hex(),
                         independently_verified=task.verify(raw).passed))
        calls += search.donor_forward_calls
    return bytes(payload), rows, calls


def adapt(base_archive, payload, geometry, evaluation):
    assert_disjoint((payload,), {"held":evaluation})
    runtime = NexusExpanseUnified.restore(base_archive)
    model = runtime.backbone
    initial = state_hash(model)
    model.train().requires_grad_(True)
    for layer in model.layers:
        layer.ffn.enable_skip = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=.01)
    targets = 0
    started = time.perf_counter()
    with forbid_donor_forward() as guard:
        for step in range(ADAPT_STEPS):
            rows = scaling.training_rows({scaling.STREAM:payload}, geometry, step)
            byte_ids, keep, count = scaling.make_batch(rows, CONTEXT, CPU)
            optimizer.zero_grad(set_to_none=True)
            update(model, optimizer, scaling.loss_for(model, byte_ids, keep, "fp32", CPU))
            targets += count
        model.eval().requires_grad_(False)
        for layer in model.layers:
            layer.ffn.enable_skip = True
        runtime.snapshot_quant = "fp32"
        archive, report = snapshot_checks(runtime, snapshot_quant="int8")
    return archive, dict(initial_state_sha256=initial, start_archive_sha256=sha(base_archive), end_archive_sha256=sha(archive),
                         training_sha256=sha(payload), training_bytes=len(payload), seconds=time.perf_counter()-started,
                         targets=targets, updates=ADAPT_STEPS, donor_forward_calls=guard.donor_forward_calls, **report)


def train_bytes(payload, length):
    return (payload * ((length + len(payload)-1)//len(payload)))[:length]


def seed_run(root, seed):
    train_rows, held, fit, calibration, held_raw = inputs(seed)
    eval_path = root / "held.bin"
    atomic_bytes(eval_path, held_raw)
    pool_path, check_path = (root / name for name in ("train-tasks.jsonl", "check-tasks.jsonl"))
    pool = task_file(pool_path, train_rows)
    suite = task_file(check_path, [row for row, _ in held])
    labels = {row["question_id"]:answer for row, answer in held}
    if (pool.unsupported or suite.unsupported or any(sum(t.category == c for t in suite.tasks) != 6 for c in CATEGORIES)
            or any(not task.verify(labels[task.id]).passed for task in suite.tasks)):
        raise ValueError("invalid frozen evaluation records")
    _, base_archive, training = train(root, seed, fit, calibration)
    base = VerifiedByteSearch(base_archive, expected_sha256=training["report"]["snapshot_sha256"], always_wake=True)
    search_seed, eval_seed = seed * 1000, seed * 1000 + 100_000
    base_scores, base_wall = evaluate_budgets(base, suite, BUDGETS, seed=eval_seed)
    traces = harvest_verified_traces(base, pool, seed=search_seed)
    out_bin, out_json = root / "verified.bin", root / "verified.jsonl"
    write_harvest(traces, out_bin, out_jsonl=out_json, evaluation=held_raw)
    written = out_bin.read_bytes()
    audited = [json.loads(line) for line in out_json.read_bytes().splitlines()]
    clean_pool = load_tasks_jsonl(pool_path)
    verified = recheck(written, audited, {task.id:task for task in clean_pool.tasks})
    replay_exact, replay_calls = None, 0
    if seed == SEEDS[0]:
        repeat = harvest_verified_traces(VerifiedByteSearch(base_archive, always_wake=True), clean_pool, seed=search_seed)
        replay_exact = traces == repeat
        replay_calls = repeat.donor_forward_calls
    control, baseline_outputs, control_calls = greedy_control(base, pool.tasks, {r["question_id"] for r in audited}, search_seed)
    control_path = root / "greedy.bin"
    atomic_bytes(control_path, control)
    assert_disjoint((control,), {"held":held_raw})
    trainable = bool(audited)
    arms = {"base":dict(scoreboards=base_scores, wall_seconds=base_wall)}
    if trainable:
        length = max(CONTEXT * BATCH+1, len(written), len(control))
        length += (1-length) % CONTEXT
        geometry = scaling.Geometry(seed, 64, CONTEXT, 1, ADAPT_STEPS, BATCH, 4, 2)
        for name, path, raw in (("unverified",control_path,control),("verified",out_bin,written)):
            loaded, evaluation, _ = scaling.load_corpora(SimpleNamespace(smoke=False, train_file=path, eval_file=eval_path))
            if loaded[scaling.STREAM] != raw or evaluation[scaling.STREAM] != held_raw:
                raise ValueError("training entry path did not preserve the exact raw bytes")
            adapted, receipt = adapt(base_archive, train_bytes(raw, length), geometry, held_raw)
            engine = VerifiedByteSearch(adapted, expected_sha256=receipt["end_archive_sha256"], always_wake=True)
            scores, wall = evaluate_budgets(engine, suite, BUDGETS, seed=eval_seed)
            arms[name] = dict(scoreboards=scores, wall_seconds=wall, adaptation=receipt)
        prefix_engine = engine
    else:
        arms.update({name:dict(scoreboards=base_scores, wall_seconds=0.0, adaptation=None)
                     for name in ("unverified", "verified")})
        prefix_engine = base
    prefix = prefix_probe(prefix_engine, suite.tasks[0]) if seed == SEEDS[0] else None
    return dict(seed=seed, training=training, arms=arms, prefix=prefix, harvest=traces.report(), audit=audited,
                independent_verified=verified, replay_exact=replay_exact, trainable=trainable,
                held_sha256=sha(held_raw), held_tasks_sha256=digest(check_path), harvest_tasks_sha256=digest(pool_path),
                shared_32_byte_windows=0, greedy_comparison=baseline_outputs, greedy_sha256=sha(control),
                extra_donor_forward_calls=control_calls+replay_calls)


def exact_snapshot(receipt):
    return (receipt["snapshot_quant"] == "int8" and receipt["exact_state"] and receipt["round_trip_probabilities"] == 0.0
            and len(receipt["parity"]) == 4 and not any(receipt["parity"].values()))


def verdict(rows):
    prefix = rows[0]["prefix"]
    good = {k:0 for k in BUDGETS}
    joint = forwards = 0
    exact = []
    for row in rows:
        arms = row["arms"]
        wins = {k:arms["verified"]["scoreboards"][k]["overall_score"] > max(
            arms[c]["scoreboards"][k]["overall_score"] for c in ("base", "unverified")) for k in BUDGETS}
        joint += all(wins.values())
        for k in BUDGETS:
            good[k] += wins[k]
        cold = row["training"]
        parity = (cold["cold_equal"] and cold["targets"] == CONTEXT*STEPS*BATCH and cold["updates"] == STEPS
                  and exact_snapshot(cold["report"]))
        forwards += row["extra_donor_forward_calls"] + cold["donor_forward_calls"] + row["harvest"]["donor_forward_calls"]
        for arm in arms.values():
            forwards += arm["scoreboards"][16]["donor_forward_calls"]
        if row["trainable"]:
            a, b = (arms[name]["adaptation"] for name in ("unverified", "verified"))
            parity &= all(exact_snapshot(x) and x["updates"] == ADAPT_STEPS and x["targets"] == ADAPT_STEPS*CONTEXT*BATCH
                          and x["start_archive_sha256"] == cold["report"]["snapshot_sha256"] for x in (a,b))
            parity &= a["initial_state_sha256"] == b["initial_state_sha256"] and a["training_bytes"] == b["training_bytes"]
            forwards += a["donor_forward_calls"]+b["donor_forward_calls"]
        exact.append(parity and row["independent_verified"] == row["harvest"]["accepted"] and not row["shared_32_byte_windows"])
    engineering = all(exact) and not forwards and rows[0]["replay_exact"] and not prefix["failures"] and prefix["max_diff"] == 0.0
    science = "INVALID" if not engineering else "PASS" if joint==len(SEEDS) else "MIXED" if any(good.values()) else "FAIL"
    return dict(engineering="PASS" if engineering else "FAIL", scientific=science, strict_wins=good, joint_wins=joint,
                donor_forward_calls=forwards, trainable_seeds=sum(r["trainable"] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out.resolve()
    intent = out.with_suffix(".intent.json")
    head = git("rev-parse", "HEAD").decode().strip()
    if (out.is_relative_to(ROOT) or out.exists() or intent.exists() or git("status", "--porcelain")
            or git("rev-parse", "HEAD^").decode().strip() != BASE or not git("ls-tree", head, PREREG)):
        parser.error("run from the clean Phase 22 preregistration commit with a fresh external output")
    binding = dict(base=BASE, source_commit=head, source_tree=git("rev-parse", "HEAD^{tree}").decode().strip(),
                   prereg_sha256=digest(ROOT / PREREG))
    atomic(intent, binding)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix="phase22-verified-self-distillation-") as tmp:
        for seed in SEEDS:
            root = Path(tmp) / str(seed)
            root.mkdir()
            rows.append(seed_run(root, seed))
            print(f"Phase 22 seed {seed} complete", flush=True)
    if git("rev-parse", "HEAD").decode().strip() != head or git("status", "--porcelain"):
        parser.error("the scored source changed")
    result = dict(schema="nexus.phase22-verified-self-distillation.v1", **binding, offline=True,
                  platform=platform.platform(), torch=str(torch.__version__), seconds=time.perf_counter()-started,
                  peak_rss_bytes=peak_rss(), seeds=rows, summary=verdict(rows))
    atomic(out, result)
    print(json.dumps(result["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
