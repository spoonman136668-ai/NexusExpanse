"""Offline DEV comparison of direct-answer and scratchpad byte curricula."""

import argparse
import hashlib
from itertools import permutations
import json
from pathlib import Path
import platform
import random
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import gate17_gpu_scaling_harness as scaling
from benchmarks.dev_artifacts import atomic, digest
from benchmarks.gate_custody import peak_rss
from benchmarks.phase13_warmstart_train import cli
from benchmarks.phase21_livebench_scoreboard import prefix_probe
from benchmarks.phase23_snapshot_finetune import exact_snapshot
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.livebench_eval import evaluate_budgets, suite_from_rows
from nexus_expanse.unified import NexusExpanseUnified
from nexus_expanse.verified_search import VerifiedByteSearch, make_verifier
from scripts.prepare_gpu_corpus import assert_disjoint
from scripts.prepare_reasoning_curriculum import arithmetic_steps, builtin_references, prepare, scratchpad

BASE = "6229a892922a47dd5c8939dae6ad5f64bc318e78"
PREREG = "docs/PHASE24_COT_AND_SCALING_PREREG.md"
SEEDS = tuple(range(3801, 3806))
BUDGETS = (1, 16)
CATEGORIES = ("math", "reasoning")
CONTEXT, BATCH, MICRO_BATCHES = 96, 1, 2
EPOCHS, STEPS = 2, 48


def records(seed, split, rounds):
    rng = random.Random(f"phase24:{seed}:{split}")
    for i in range(rounds):
        mark = f"[{seed}.{i:x}.{split}]\n"
        base = {"T": 15, "E": 46, "H": 74}[split]
        a, b, c = rng.randrange(base, base+24), rng.randrange(2, 10), rng.randrange(2, 22)
        prompt = f"{mark}({a}*{b})+{c}="
        answer = f"{a*b+c}\n"
        work = f"{arithmetic_steps(a, '*', b)} {a*b}+{c}={a*b+c}."
        yield dict(id=f"{split}-math-{i}", category="math", verifier="arithmetic", prompt=prompt, max_bytes=112), answer, work

        slots = [base*100+4*i+j for j in range(3)]
        first, middle, last = rng.choice(tuple(permutations("ABC")))
        # Each pair of before constraints forces a distinct consecutive slot.
        before = [(first,middle), (middle,last)]
        spec = dict(variables={key:slots for key in "ABC"},
                    constraints=[dict(type="before", left=x, right=y) for x,y in before])
        assignments = dict(zip((first,middle,last), slots, strict=True))
        prompt = f"{mark}{first}<{middle}<{last}. A=n,B=n,C=n. Slots {','.join(map(str,slots))}.\n"
        answer = ",".join(f"{key}={assignments[key]}" for key in "ABC") + "\n"
        work = f"{first} needs 2 later: {slots[0]}. {middle} needs 1 later: {slots[1]}. {last}={slots[2]}."
        yield dict(id=f"{split}-logic-{i}", category="reasoning", spec=spec, prompt=prompt, max_bytes=128), answer, work


def encode(rows, use_think):
    chunks = []
    for row, answer, work in rows:
        raw = (scratchpad(work, answer) if use_think else answer).encode()
        category = row["category"]
        kind = "arithmetic" if category == "math" else "logic_csp"
        prompt = row["prompt"].encode()
        check = make_verifier(kind, prompt, spec=row.get("spec"), allow_think=True)(raw)
        if not check.passed:
            raise ValueError(f"invalid frozen {category} answer: {row['id']}")
        chunks.append(prompt + raw + b"\n")
    return b"".join(chunks)


def corpora(root, seed):
    fit, calibration, held = (tuple(records(seed, split, n)) for split,n in (("T",96), ("E",12), ("H",6)))
    held_bytes = {f"held-{think}":encode(held, think) for think in (False, True)}
    suite = suite_from_rows((row for row, _, _ in held), allow_think=True)
    evidence = {}
    for arm, think in (("direct", False), ("scratchpad", True)):
        path = root / arm
        path.mkdir()
        refs = {**held_bytes, **builtin_references()}
        train, val = encode(fit, think), encode(calibration, think)
        # Keep every sampled block full so both arms score the same target bytes.
        train = train[:((len(train)-1)//CONTEXT)*CONTEXT+1]
        assert_disjoint((train, val), refs)
        assert_disjoint((train,), {"calibration": val})
        files = {}
        for name, raw in (("train", train), ("eval", val)):
            file = path / f"stream-{name}.bin"
            file.write_bytes(raw)
            files[name] = dict(bytes=len(raw), sha256=digest(file))
        evidence[arm] = dict(**files, fit_records=len(fit), calibration_records=len(calibration),
                             shared_32_byte_windows=0)
    return suite, evidence, {name:hashlib.sha256(raw).hexdigest() for name, raw in held_bytes.items()}


def train(root, corpus, seed, *, resume=False):
    run, archive = root / "training", root / "unified.pt"
    args = ("train", "--train-file", corpus/"stream-train.bin", "--eval-file", corpus/"stream-eval.bin",
            "--directory", run, "--device", "cpu", "--dtype", "fp32", "--seed", seed, "--use-sdpa",
            "--dim", 32, "--layers", 2, "--n-heads", 8, "--grad-accum", MICRO_BATCHES, "--grad-checkpoint",
            "--context", CONTEXT, "--epochs", EPOCHS, "--steps-per-epoch", STEPS, "--batch-per-domain", BATCH,
            "--eval-blocks", 4, "--epoch-limit", 1 if resume else EPOCHS, "--lr", 0.001, "--lr-schedule", "cosine")
    if resume:
        cli(*args)
    export = cli(*args, "--save-snapshot", archive, "--snapshot-quant", "int8")
    receipt = json.loads((run / "gate17.json").read_bytes())
    return archive.read_bytes(), dict(export=export, receipt=receipt)


def exact_resume(a, b):
    ar, br = a["receipt"], b["receipt"]
    return (ar["cold_equal"] and br["cold_equal"] and ar["initial_hash"] == br["initial_hash"]
            and ar["final_hash"] == br["final_hash"] and ar["checkpoint_sha256"] == br["checkpoint_sha256"]
            and len({row["telemetry"]["worker_pid"] for row in br["epochs"]}) == EPOCHS
            and ar["fresh_evaluation"]["parity"] == br["fresh_evaluation"]["parity"]
            and not any(ar["fresh_evaluation"]["parity"].values())
            and all(x["losses"] == y["losses"] and x["sha256"] == y["sha256"]
                    for x,y in zip(ar["epochs"], br["epochs"], strict=True))
            and exact_snapshot(a["export"]) and exact_snapshot(b["export"])
            and a["export"]["snapshot_sha256"] == b["export"]["snapshot_sha256"])


def large_geometry():
    geometry = scaling.Geometry(3801, 1024, 4096, 1, 1, 1, 1, 16, 16)
    with forbid_donor_forward() as guard, torch.inference_mode():
        model = scaling.model_for(geometry, "causal_attention_skill", torch.device("cpu")).eval()
        tokens = torch.tensor([[0, 127, 255]])
        # Limit this CPU forward to the three input positions while preserving full snapshot geometry.
        logits = model.forward_neural(tokens)[0]
        runtime = scaling.build_unified(model, {scaling.STREAM: b"native bytes"})
        kind, config = runtime._config()
        fp32_bytes = sum(t.numel()*t.element_size() for t in model.state_dict().values())
        size_limit = NexusExpanseUnified._snapshot_limit(kind, config)
        raw = runtime.snapshot(snapshot_quant="int8")
        with tempfile.TemporaryFile() as file:
            torch.save(model.state_dict(), file)
            fp32_checkpoint_bytes = file.tell()
        return dict(config=config, parameters=sum(t.numel() for t in model.parameters()),
                    fp32_tensor_bytes=fp32_bytes, fp32_checkpoint_bytes=fp32_checkpoint_bytes,
                    int8_snapshot_bytes=len(raw), snapshot_limit=size_limit,
                    logits_shape=list(logits.shape), finite=bool(torch.isfinite(logits).all()),
                    donor_forward_calls=guard.donor_forward_calls)


def seed_run(root, seed):
    suite, curriculum, held_digests = corpora(root, seed)
    training, scoreboards, times = {}, {}, {}
    prefix, resume = None, None
    for arm in curriculum:
        raw, training[arm] = train(root / f"run-{arm}", root / arm, seed)
        sha = training[arm]["export"]["snapshot_sha256"]
        engine = VerifiedByteSearch(raw, expected_sha256=sha, always_wake=True, no_episodic=True)
        scoreboards[arm], times[arm] = evaluate_budgets(engine, suite, BUDGETS, seed=seed*1000, allow_think=True)
        if seed == SEEDS[0] and arm == "direct":
            other, receipt = train(root / "resume", root / arm, seed, resume=True)
            resume = dict(exact=raw == other and exact_resume(training[arm], receipt), **receipt)
        if seed == SEEDS[0] and arm == "scratchpad":
            prefix = prefix_probe(engine, suite.tasks[0])
    return dict(seed=seed, curriculum=curriculum, held_sha256=held_digests, training=training, cold_resume=resume,
                scoreboards=scoreboards, evaluation_wall_seconds=times, prefix=prefix)


def summarize(rows, large, pack):
    arms = ("direct", "scratchpad")
    totals = {k:{arm:{c:sum(r["scoreboards"][arm][k]["counts"][c]["passed"] for r in rows)
                      for c in CATEGORIES} for arm in arms} for k in BUDGETS}
    wins = {k:sum(r["scoreboards"]["scratchpad"][k]["overall_score"] > r["scoreboards"]["direct"][k]["overall_score"]
                  for r in rows) for k in BUDGETS}
    deltas = {k:{c:totals[k]["scratchpad"][c]-totals[k]["direct"][c] for c in CATEGORIES} for k in BUDGETS}
    training = [t for r in rows for t in r["training"].values()] + [rows[0]["cold_resume"]]
    calls = large["donor_forward_calls"] + pack["donor_forward_calls"]
    calls += sum(t["export"]["donor_forward_calls"] + t["receipt"]["fresh_evaluation"]["donor_forward_calls"] for t in training)
    calls += sum(r["scoreboards"][arm][16]["donor_forward_calls"] for r in rows for arm in arms)
    prefix = rows[0]["prefix"]
    engineering = (rows[0]["cold_resume"]["exact"] and large["finite"] and large["logits_shape"] == [1,3,256]
                   and 100_000_000 <= large["parameters"] and 0 < large["int8_snapshot_bytes"] < large["fp32_tensor_bytes"]
                   <= large["fp32_checkpoint_bytes"] < large["snapshot_limit"]
                   and pack["verifier_passed"] == sum(f["examples"] for f in pack["files"].values())
                   and not any(pack["shared_32_byte_windows"].values()) and not calls
                   and not prefix["failures"] and prefix["max_diff"] == 0.0
                   and all(t["receipt"]["cold_equal"] and exact_snapshot(t["export"])
                           and sum(e["telemetry"]["target_bytes"] for e in t["receipt"]["epochs"])
                           == CONTEXT*BATCH*MICRO_BATCHES*EPOCHS*STEPS
                           and not any(t["receipt"]["fresh_evaluation"]["parity"].values()) for t in training))
    science = ("INVALID" if not engineering else
               "PASS" if min(wins.values()) == len(SEEDS) and all(v > 0 for group in deltas.values() for v in group.values()) else
               "MIXED" if any(wins.values()) or any(v > 0 for group in deltas.values() for v in group.values()) else "FAIL")
    return dict(engineering="PASS" if engineering else "FAIL", scientific=science, passed=totals,
                scratchpad_minus_direct=deltas, strict_win_seeds=wins, donor_forward_calls=calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out.resolve()
    intent = out.with_suffix(".intent.json")
    git = scaling.git
    head = git("rev-parse", "HEAD").decode().strip()
    if (out.is_relative_to(ROOT) or out.exists() or intent.exists() or git("status", "--porcelain")
            or git("rev-parse", "HEAD^").decode().strip() != BASE or not git("ls-tree", head, PREREG)):
        parser.error("run from the clean Phase 24 preregistration commit with a fresh external output")
    binding = dict(base=BASE, source_commit=head, source_tree=git("rev-parse", "HEAD^{tree}").decode().strip(),
                   prereg_sha256=digest(ROOT / PREREG))
    atomic(intent, binding)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    large = large_geometry()
    with tempfile.TemporaryDirectory(prefix="phase24-cot-scaling-") as tmp:
        root = Path(tmp)
        pack = prepare(root / "curriculum", size_bytes=48_000, seed=SEEDS[0], include_think=True)
        rows = []
        for seed in SEEDS:
            dest = root / str(seed)
            dest.mkdir()
            rows.append(seed_run(dest, seed))
            print(f"Phase 24 seed {seed} complete", flush=True)
    if git("rev-parse", "HEAD").decode().strip() != head or git("status", "--porcelain"):
        parser.error("the scored source changed")
    result = dict(schema="nexus.phase24-cot-and-scaling.v1", **binding, offline=True, platform=platform.platform(),
                  torch=str(torch.__version__), seconds=time.perf_counter()-started, peak_rss_bytes=peak_rss(),
                  flags=dict(allow_think=True, always_wake=True, no_episodic=True), large_geometry=large,
                  curriculum_pack=pack, seeds=rows, summary=summarize(rows, large, pack))
    atomic(out, result)
    print(json.dumps(result["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
