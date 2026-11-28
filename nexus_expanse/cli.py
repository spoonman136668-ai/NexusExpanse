"""Nexus Expanse raw-byte developer CLI."""
import argparse
from collections import deque
from contextlib import nullcontext
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from benchmarks import gate17_gpu_scaling_harness as scaling
from benchmarks.dev_artifacts import atomic_bytes, digest
from benchmarks.gate_custody import EpochRun
from benchmarks.unified_checks import exactness
from nexus_expanse.byte_decode import engram_forecast, nucleus_byte
from nexus_expanse.expr_verifier import EXPRESSIONS, MAX_LENGTH, run_pipeline, verify as verify_expression
from nexus_expanse.jepa import state_hash
from nexus_expanse.livebench_eval import VERIFIERS, builtin_dev_suite, evaluate_budgets, load_tasks_jsonl
from nexus_expanse.patching import CompactBigram
from nexus_expanse.snapshot_quant import SNAPSHOT_QUANTS
from nexus_expanse.self_distill import builtin_train_suite, harvest_verified_traces, write_harvest
from nexus_expanse.ssd_engram import SSDEngramStore
from nexus_expanse.structured_verifiers import strict_json
from nexus_expanse.unified import NexusExpanseUnified
from nexus_expanse.verifier_agent import VerifierGatedNexusAgent
from nexus_expanse.checkpoint_transport import CONFIG as SKILL_CONFIG, CorrectedLinear, ImportedSkill, METHODS as SKILL_METHODS, load_skill
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.multi_skill_router import MultiSkillBank, load_bank
from nexus_expanse.cache_assimilation import assimilate
from nexus_expanse.shard_stream_assimilation import STUDENT_DIMS, STUDENT_LAYERS, assimilate_zero_exec
from nexus_expanse.weight_editing import (
    apply_surgical_edit, exact_skill_receipt, inspect_fault_direction, restore_skill, verify_skill,
)
from nexus_expanse.verified_search import MAX_CASE_BYTES, MAX_OUTPUT, VerifiedByteSearch, make_verifier

VERSION = "2.3.0-dev"
TRAIN_OPTIONS = ("device", "dtype", "epoch_limit", "seed", "dim", "context",
                 "epochs", "steps_per_epoch", "batch_per_domain", "eval_blocks", "layers", "lr", "lr_schedule",
                 "n_heads", "grad_accum")
EXPRESSION_LIBRARY = dict(zip(b"amdrvX", EXPRESSIONS, strict=True))
MAX_STREAM_CHUNK = 4096
MAX_REPEAT_WINDOW = 512


def emit(obj):
    print(json.dumps(obj, allow_nan=False), flush=True)


def read_snapshot(path, limit, expected=None):
    with path.open("rb") as src:
        raw = src.read(limit + 1)
    if not 0 < len(raw) <= limit:
        raise ValueError("snapshot is empty or exceeds its size limit")
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected.lower():
        raise ValueError("snapshot SHA-256 mismatch")
    return raw


def parsed_hex(text, limit):
    data = bytes.fromhex(text)
    if not 1 <= len(data) <= limit:
        raise ValueError(f"expected between 1 and {limit} raw bytes")
    return data


def snapshot_checks(model, *, snapshot_quant=None):
    if snapshot_quant is None:
        snapshot_quant = model.snapshot_quant
    original = model.snapshot(snapshot_quant=snapshot_quant)
    restored = NexusExpanseUnified.restore(original)
    if snapshot_quant != "fp32" and snapshot_quant != model.snapshot_quant:
        # Check the weights actually delivered by a lossy export, including
        # whether its dequantized state survives a second pack and restore.
        model = restored
        restored = NexusExpanseUnified.restore(model.snapshot(snapshot_quant=snapshot_quant))
    # Exercise the actual saved cursor before resetting for the prefix check.
    round_trip = 0.
    for byte in b"\x1fnexus\x00":
        a = model.step(byte, adapt=False).mixed
        b = restored.step(byte, adapt=False).mixed
        round_trip = max(round_trip, (a - b).abs().max().item())
    state_equal = all((
        state_hash(model.backbone) == state_hash(restored.backbone),
        model.episodic.snapshot(max_bytes=model.MEMORY_BUDGET) ==
        restored.episodic.snapshot(max_bytes=restored.MEMORY_BUDGET),
        model.reasoning.snapshot() == restored.reasoning.snapshot(),
        (model.arbitrator.updates, model.arbitrator.share, model.arbitrator.weight(0.)) ==
        (restored.arbitrator.updates, restored.arbitrator.share, restored.arbitrator.weight(0.)),
    ))
    model.reset_context()
    restored.reset_context()
    window = bytes(range(min(model.max_seq_len, 32)))
    parity = exactness(model, restored, window)
    if not state_equal or round_trip != 0. or any(v != 0. for v in parity.values()):
        raise ValueError(f"unified runtime parity failed: {parity}")
    return original, dict(snapshot_quant=snapshot_quant, parity=parity,
                          round_trip_probabilities=round_trip, exact_state=state_equal)


def default_model():
    geometry = scaling.Geometry(42, 32, 32, 1, 1, 1, 1)
    model = scaling.model_for(geometry, "causal_attention_monarch", torch.device("cpu"))
    counts = CompactBigram.from_dense(np.zeros((256, 256), dtype="<u4"))
    return NexusExpanseUnified(model, counts)


def verify_command(args):
    if args.snapshot_sha256 is not None and not args.snapshot:
        raise ValueError("--snapshot-sha256 requires --snapshot")
    if args.snapshot_quant is not None and not args.save_snapshot:
        raise ValueError("--snapshot-quant requires --save-snapshot")
    with forbid_donor_forward() as guard:
        if args.snapshot:
            raw = read_snapshot(args.snapshot, NexusExpanseUnified.SNAPSHOT_LIMIT, args.snapshot_sha256)
            model = NexusExpanseUnified.restore(raw)
        else:
            raw = None
            model = default_model()
        snapshot, report = snapshot_checks(model, snapshot_quant=args.snapshot_quant)
        if raw is not None and report["snapshot_quant"] == model.snapshot_quant:
            snapshot = raw
    if args.save_snapshot:
        atomic_bytes(args.save_snapshot, snapshot)
    emit(dict(command="verify", status="PASS", snapshot_bytes=len(snapshot),
              snapshot_sha256=hashlib.sha256(snapshot).hexdigest(), donor_forward_calls=guard.donor_forward_calls,
              **report))


def raw_chunks(args):
    if args.input_hex is not None:
        for line in args.input_hex:
            yield parsed_hex(line, MAX_STREAM_CHUNK)
        return
    while line := sys.stdin.readline(8194):
        if len(line) >= 8194:
            raise ValueError("hex input line too long")
        if line.strip():
            yield parsed_hex(line, MAX_STREAM_CHUNK)


def parse_repetition_penalty(text):
    penalty = float(text)
    if not math.isfinite(penalty) or penalty < 1.0:
        raise argparse.ArgumentTypeError("repetition penalty must be finite and at least 1.0")
    return penalty


def repetition_adjusted_logs(probabilities, recent, penalty):
    logs = probabilities.log()
    if recent:
        indices = sorted(set(recent))
        repeated = logs[indices]
        logs[indices] = repeated * penalty
        if torch.isneginf(logs.max()):
            # All positive-mass bytes were penalized and multiplication
            # overflowed. A common shift preserves their relative scores.
            logs[indices] = (repeated - repeated.max()) * penalty
    return logs


def continuation(model, forecast, length, *, adapt=False, temperature=0.0, top_p=1.0, seed=42,
                 repetition_penalty=1.0, repeat_window=64, prompt=b"", forecast_filter=None):
    if not math.isfinite(repetition_penalty) or repetition_penalty < 1.0:
        raise ValueError("repetition penalty must be finite and at least 1.0")
    if type(repeat_window) is not int or not 1 <= repeat_window <= MAX_REPEAT_WINDOW:
        raise ValueError("repeat window must be between 1 and 512 bytes")
    recent = deque(prompt[-repeat_window:], maxlen=repeat_window) if repetition_penalty > 1.0 else None
    generator = torch.Generator().manual_seed(seed) if temperature else None
    output = bytearray()
    for _ in range(length):
        forecast_probs = forecast.mixed
        if forecast_filter is not None:
            forecast_probs = forecast_filter(forecast_probs, output)
        if generator is None:
            scores = (forecast_probs if recent is None else
                      repetition_adjusted_logs(forecast_probs, recent, repetition_penalty))
            byte = int(scores.argmax())
        else:
            logs = (forecast_probs.log() if recent is None else
                    repetition_adjusted_logs(forecast_probs, recent, repetition_penalty))
            byte = nucleus_byte(logs, temperature, top_p, generator)
        output.append(byte)
        if recent is not None:
            recent.append(byte)
        forecast = model.step(byte, adapt=adapt)
    return bytes(output)


def stream_command(args):
    data = read_snapshot(args.snapshot, NexusExpanseUnified.SNAPSHOT_LIMIT, args.snapshot_sha256)
    snapshot_checks(NexusExpanseUnified.restore(data))
    model = NexusExpanseUnified.restore(data)
    observed = 0
    for chunk in raw_chunks(args):
        for byte in chunk:
            forecast = model.step(byte, adapt=not args.no_adapt)
        observed += len(chunk)
        indices = torch.argsort(forecast.mixed, descending=True, stable=True)[:5]
        predicted = [dict(byte=int(b), probability=float(forecast.mixed[b])) for b in indices]
        speculative = b""
        if args.continuation:
            shadow = NexusExpanseUnified.restore(model.snapshot())
            speculative = continuation(shadow, forecast, args.continuation)
        emit(dict(command="stream", observed_hex=chunk.hex(), next=predicted,
                  continuation_hex=speculative.hex(), arbitration_updates=model.arbitrator.updates))
    if not observed:
        raise ValueError("provide at least one observed hex byte")
    if args.save_snapshot:
        snapshot, report = snapshot_checks(model)
        atomic_bytes(args.save_snapshot, snapshot)
        emit(dict(command="stream", saved_snapshot=str(args.save_snapshot),
                  snapshot_sha256=hashlib.sha256(snapshot).hexdigest(), **report))


def build_engram_command(args):
    report = SSDEngramStore.build(args.input, args.out, min_order=args.min_order,
                                  max_order=args.max_order, min_support=args.min_support)
    emit(dict(command="build-engram", **report, donor_forward_calls=0))


def parse_engram_alpha(text):
    value = float(text)
    if not math.isfinite(value) or not 0 <= value <= 0.9:
        raise argparse.ArgumentTypeError("engram alpha must be finite and in [0.0, 0.9]")
    return value


def open_engram(args):
    if args.engram_index_sha256 is not None and args.engram_index is None:
        raise ValueError("--engram-index-sha256 requires --engram-index")
    if args.engram_index is None:
        return nullcontext()
    return SSDEngramStore(args.engram_index, expected_sha256=args.engram_index_sha256)


def add_engram_arguments(parser):
    parser.add_argument("--engram-index", type=Path, help="read-only byte suffix index")
    parser.add_argument("--engram-index-sha256", help="verify the index against a pinned digest")
    parser.add_argument("--engram-alpha", type=parse_engram_alpha, default=0.35)


def generate_command(args):
    if (not math.isfinite(args.temperature) or args.temperature < 0
            or not math.isfinite(args.top_p) or not 0 < args.top_p <= 1):
        raise ValueError("temperature must be finite and nonnegative and top-p must be in (0, 1]")
    if args.prompt_text is not None:
        prompt = args.prompt_text.encode("utf-8")
        if not 1 <= len(prompt) <= MAX_STREAM_CHUNK:
            raise ValueError(f"expected between 1 and {MAX_STREAM_CHUNK} prompt bytes")
    else:
        prompt = parsed_hex(args.prompt_hex, MAX_STREAM_CHUNK)
    with open_engram(args) as engram, forbid_donor_forward() as guard:
        hits = 0
        forecast_filter = None
        if engram is not None:
            tail = prompt[-engram.max_order:]

            def forecast_filter(probabilities, output):
                nonlocal hits
                observed = tail + bytes(output[-engram.max_order:])
                mixed, gain = engram_forecast(probabilities, observed, engram, args.engram_alpha)
                hits += int(gain > 0)
                return mixed

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        device = scaling.device_for(args.device, "fp32")
        data = read_snapshot(args.snapshot, NexusExpanseUnified.SNAPSHOT_LIMIT, args.snapshot_sha256)
        model = NexusExpanseUnified.restore(data, fast_decode=True, decode_device=device)
        model.no_episodic = args.no_episodic
        if args.always_wake:
            model.reader.minimum_probability = 2.0
        start = time.perf_counter()
        forecast = model.prefill(prompt, adapt=args.adapt)
        prefill_elapsed = time.perf_counter() - start
        start = time.perf_counter()
        generated = continuation(model, forecast, args.max_bytes, adapt=args.adapt,
                                 temperature=args.temperature, top_p=args.top_p, seed=args.seed, prompt=prompt,
                                 repetition_penalty=args.repetition_penalty, repeat_window=args.repeat_window,
                                 forecast_filter=forecast_filter)
        elapsed = time.perf_counter() - start
        engram_report = ({} if engram is None else dict(engram_hits=hits, engram_hit_rate=hits / len(generated),
                                                       engram_index_sha256=engram.index_sha256))
        emit(dict(
            command="generate", prompt_bytes=len(prompt), generated_bytes=len(generated),
            generated_hex=generated.hex(), generated_text=generated.decode("utf-8", errors="replace"),
            device=device.type, prefill_ms=prefill_elapsed * 1000,
            ms_per_byte=elapsed * 1000 / len(generated), snapshot_sha256=hashlib.sha256(data).hexdigest(),
            repetition_penalty=args.repetition_penalty, repeat_window=args.repeat_window,
            always_wake=bool(args.always_wake), no_episodic=bool(args.no_episodic),
            donor_forward_calls=guard.donor_forward_calls,
            **engram_report,
        ))


def read_verifier_json(value):
    if value is None:
        return None
    if value.lstrip().startswith(("[", "{")):
        raw = value.encode("utf-8")
    else:
        with Path(value).open("rb") as source:
            raw = source.read(MAX_CASE_BYTES + 1)
    if len(raw) > MAX_CASE_BYTES:
        raise ValueError("verifier JSON exceeds its size limit")
    try:
        return strict_json(raw)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("invalid verifier JSON") from exc


def search_verified_command(args):
    prompt = args.prompt_text.encode("utf-8")
    if args.verifier == "arithmetic" and args.tests_json is not None:
        raise ValueError("arithmetic does not take test cases")
    verifier = make_verifier(args.verifier, prompt, read_verifier_json(args.tests_json),
                             spec=read_verifier_json(args.spec_json), allow_think=args.allow_think)
    stop = (args.stop_text.encode("utf-8") if args.stop_text is not None else
            b"\n" if args.verifier == "arithmetic" else b"\n\n")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = scaling.device_for(args.device, "fp32")
    data = read_snapshot(args.snapshot, NexusExpanseUnified.SNAPSHOT_LIMIT)
    start = time.perf_counter()
    with open_engram(args) as engram:
        engine = VerifiedByteSearch(data, expected_sha256=args.snapshot_sha256, device=device,
                                    always_wake=args.always_wake, no_episodic=args.no_episodic,
                                    engram=engram, engram_alpha=args.engram_alpha)
        result = engine.run(prompt, verifier, candidates=args.candidates, max_bytes=args.max_bytes,
                            temperature=args.temperature, top_p=args.top_p, seed=args.seed, stop_bytes=stop, allow_think=args.allow_think)
        selected = result.selected
        report = dict(command="search-verified", verified=selected.verification.passed,
                      candidates_tried=len(result.candidates), verified_count=sum(c.verification.passed for c in result.candidates),
                      selected_text=selected.raw.decode("utf-8", errors="replace"), selected_hex=selected.raw.hex(),
                      selected_logprob=selected.logprob, passed_tests=selected.verification.passed_tests,
                      total_tests=selected.verification.total_tests, error=selected.verification.error,
                      snapshot_sha256=engine.snapshot_sha256, device=device.type, always_wake=args.always_wake,
                      no_episodic=bool(args.no_episodic), allow_think=args.allow_think,
                      donor_forward_calls=result.donor_forward_calls, wall_ms=(time.perf_counter() - start) * 1000)
        if engram is not None:
            report["engram_index_sha256"] = engram.index_sha256
        emit(report)


def eval_livebench_command(args):
    started = time.perf_counter()
    suite = (builtin_dev_suite(allow_think=args.allow_think) if args.builtin_dev_suite
             else load_tasks_jsonl(args.tasks_jsonl, allow_think=args.allow_think))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = scaling.device_for(args.device, "fp32")
    data = read_snapshot(args.snapshot, NexusExpanseUnified.SNAPSHOT_LIMIT)
    with open_engram(args) as engram:
        engine = VerifiedByteSearch(data, expected_sha256=args.snapshot_sha256, device=device,
                                    always_wake=args.always_wake, no_episodic=args.no_episodic,
                                    engram=engram, engram_alpha=args.engram_alpha)
        boards, _ = evaluate_budgets(engine, suite, (args.candidates,), seed=args.seed, max_bytes=args.max_bytes,
                                     temperature=args.temperature, top_p=args.top_p, allow_think=args.allow_think)
        report = boards[args.candidates]
        if engram is not None:
            report["engram_index_sha256"] = engram.index_sha256
        emit(dict(command="eval-livebench", scope="local_objective", device=device.type,
                  source="builtin_dev_suite" if args.builtin_dev_suite else "tasks_jsonl",
                  no_episodic=bool(args.no_episodic), allow_think=args.allow_think,
                  wall_seconds=time.perf_counter() - started, **report))


def harvest_verified_command(args):
    started = time.perf_counter()
    suite = builtin_train_suite() if args.builtin_train_suite else load_tasks_jsonl(args.tasks_jsonl)
    evaluation = args.eval_file.read_bytes() if args.eval_file is not None else None
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = scaling.device_for(args.device, "fp32")
    raw = read_snapshot(args.snapshot, NexusExpanseUnified.SNAPSHOT_LIMIT)
    with open_engram(args) as engram:
        engine = VerifiedByteSearch(raw, expected_sha256=args.snapshot_sha256, device=device,
                                    always_wake=args.always_wake, no_episodic=args.no_episodic,
                                    engram=engram, engram_alpha=args.engram_alpha)
        harvest = harvest_verified_traces(engine, suite, candidates=args.candidates, max_bytes=args.max_bytes,
                                          temperature=args.temperature, top_p=args.top_p, seed=args.seed)
        write_harvest(harvest, args.out_bin, out_jsonl=args.out_jsonl, evaluation=evaluation)
        report = harvest.report()
        if engram is not None:
            report["engram_index_sha256"] = engram.index_sha256
        if evaluation is not None:
            report.update(heldout_sha256=hashlib.sha256(evaluation).hexdigest(), shared_32_byte_windows=0)
        emit(dict(command="harvest-verified", device=device.type, candidates=args.candidates,
                  no_episodic=bool(args.no_episodic), wall_seconds=time.perf_counter() - started, **report))


def add_search_runtime_arguments(parser, *, max_bytes):
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--snapshot-sha256")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    add_engram_arguments(parser)
    parser.add_argument("--always-wake", action="store_true")
    parser.add_argument("--no-episodic", action="store_true", help="bypass episodic memory during decode")
    parser.add_argument("--candidates", type=int, choices=range(1, 65), default=16, metavar="1..64")
    parser.add_argument("--max-bytes", type=int, choices=range(1, MAX_OUTPUT + 1), default=max_bytes, metavar="1..512")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)


def corpus_paths(directory):
    return scaling.corpus_paths(directory)


def export_trained_snapshot(directory, paths, target, *, snapshot_quant="fp32", focus_domain="all", init_snapshot=None):
    report = json.loads((directory / "gate17.json").read_text())
    identity = report["identity"]
    geometry = scaling.replace(scaling.Geometry(**identity["geometry"]), grad_checkpoint=False)
    architecture = identity.get("snapshot_architecture")
    expected_init = identity.get("init_snapshot_sha256")
    if (digest(init_snapshot) if init_snapshot is not None else None) != expected_init:
        raise ValueError("initial snapshot SHA-256 changed before export")
    if (scaling.git("status", "--porcelain")
            or identity["source_sha"] != scaling.git("rev-parse", "HEAD").decode().strip()
            or identity["tree"] != scaling.git("rev-parse", "HEAD^{tree}").decode().strip()
            or (architecture is None and identity["backbone"] != scaling.selected_backbone(
                use_sdpa=identity.get("use_sdpa", False), skill_backbone=identity.get("skill_backbone", False),
            ))
            or identity["gate15_sha256"] != digest(scaling.GATE15)):
        raise ValueError("training result does not bind the current committed source")
    train, _, manifest = scaling.load_corpora(SimpleNamespace(smoke=False, focus_domain=focus_domain, **paths))
    if identity["corpora"] != manifest or identity.get("focus_domain", "all") != focus_domain:
        raise ValueError("training corpus changed before export")
    spec = EpochRun(directory, "gate17", identity, geometry.epochs, geometry.steps_per_epoch)
    verified = spec.read()
    if any(verified["fresh_evaluation"]["parity"].values()):
        raise ValueError("training checkpoint has no exact parity evidence")
    options = {} if architecture is None else dict(snapshot_architecture=architecture)
    model = scaling.model_for(geometry, identity["backbone"], torch.device("cpu"), **options)
    model.load_state_dict(torch.load(spec.checkpoint, weights_only=True, map_location="cpu"))
    if state_hash(model) != verified["final_hash"]:
        raise ValueError("restored neural weights disagree with the committed training receipt")
    torch.set_num_threads(1)
    with forbid_donor_forward() as guard:
        snapshot, checks = snapshot_checks(scaling.build_unified(model, train, focus_domain=focus_domain),
                                           snapshot_quant=snapshot_quant)
    atomic_bytes(target, snapshot)
    init_receipt = {} if expected_init is None else dict(init_snapshot_sha256=expected_init)
    emit(dict(command="train", saved_snapshot=str(target), neural_state_sha256=verified["final_hash"],
              snapshot_sha256=hashlib.sha256(snapshot).hexdigest(),
              donor_forward_calls=verified.get("donor_forward_calls", 0) + guard.donor_forward_calls,
              **checks, **init_receipt))


def train_command(args):
    if args.snapshot_quant != "fp32" and not args.save_snapshot:
        raise ValueError("--snapshot-quant requires --save-snapshot")
    if args.train_file:
        if not args.eval_file or args.focus_domain != "all":
            raise ValueError("--train-file requires --eval-file and cannot be combined with a focused domain")
        paths = dict(train_file=args.train_file, eval_file=args.eval_file)
    else:
        if args.eval_file:
            raise ValueError("--eval-file requires --train-file")
        paths = corpus_paths(args.corpus_dir)
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"missing raw corpus file: {path}")
    if args.save_snapshot and args.save_snapshot.exists():
        raise FileExistsError(args.save_snapshot)
    command = ["--directory", str(args.directory)]
    if args.focus_domain != "all":
        command.extend(("--focus-domain", args.focus_domain))
    for name in TRAIN_OPTIONS:
        value = getattr(args, name)
        if value is not None:
            command.extend(("--" + name.replace("_", "-"), str(value)))
    for name, path in paths.items():
        command.extend(("--" + name.replace("_", "-"), str(path)))
    if args.use_sdpa:
        command.append("--use-sdpa")
    if args.init_skill:
        command.extend(("--init-skill", str(args.init_skill)))
    if args.init_snapshot:
        command.extend(("--init-snapshot", str(args.init_snapshot)))
    if args.skill_backbone:
        command.append("--skill-backbone")
    if args.grad_checkpoint:
        command.append("--grad-checkpoint")
    if args.until_complete:
        command.append("--until-complete")
    scaling.main(command)
    if args.save_snapshot and (args.directory / "gate17.json").is_file():
        options = {} if args.init_snapshot is None else dict(init_snapshot=args.init_snapshot)
        export_trained_snapshot(args.directory.resolve(), paths, args.save_snapshot,
                                snapshot_quant=args.snapshot_quant, focus_domain=args.focus_domain, **options)


def solve_command(args):
    source = parsed_hex(args.source_hex, MAX_LENGTH)
    goal = parsed_hex(args.target_hex, MAX_LENGTH)
    if len(source) != len(goal) or any(b > 7 for b in source + goal):
        raise ValueError("solve needs equal-length byte strings in the 00–07 alphabet")
    if args.agent_snapshot:
        data = read_snapshot(args.agent_snapshot, VerifierGatedNexusAgent.BUDGET)
        agent = VerifierGatedNexusAgent.restore(data)
        if not agent.gated or not agent.structured:
            raise ValueError("solve needs a gated expression agent snapshot")
        calibration = 0
    else:
        agent = VerifierGatedNexusAgent(structured=True)
        calibration = 0
        for i in range(8):
            row = bytes((i, (i + 1) % 8, (i + 3) % 8))
            for op in EXPRESSION_LIBRARY:
                primitive = bytes((op,))
                target = run_pipeline(row, primitive, EXPRESSION_LIBRARY)[-1]
                agent.observe(verify_expression(row, target, primitive, EXPRESSION_LIBRARY))
                calibration += 1
    programs = [bytes(p) for p in itertools.product(EXPRESSION_LIBRARY, repeat=args.depth)]
    random.Random(42).shuffle(programs)
    attempted = set()
    result = None
    for _ in range(min(args.budget, len(programs))):
        pipeline = agent.propose(source, goal, programs, attempted)
        attempted.add(pipeline)
        result = verify_expression(source, goal, pipeline, EXPRESSION_LIBRARY)
        agent.observe(result)
        if result.passed:
            break
    if args.save_agent_snapshot:
        atomic_bytes(args.save_agent_snapshot, agent.snapshot())
    emit(dict(command="solve", solved=result.passed, calls=len(attempted), calibration_calls=calibration,
              output_hex=result.states[-1].hex(), residual_hex=result.residual.hex(),
              program=[EXPRESSION_LIBRARY[b].decode("ascii") for b in result.program]))
    return 0 if result.passed else 1



def import_skill_command(args):
    with forbid_donor_forward() as guard, torch.inference_mode():
        skill = load_skill(args.donor, args.method, width_factor=args.width_factor).eval()
        probe = torch.arange(24, dtype=torch.long).unsqueeze(0)
        alt = probe.clone()
        alt[:, 12:] = (alt[:, 12:] + 17) % 256
        y1 = skill(probe)
        y2 = skill(alt)
        prefix_diff = float((y1[:, :12] - y2[:, :12]).abs().max().item())
        payload = dict(
            state=skill.state_dict(),
            method=args.method,
            width_factor=args.width_factor,
            width=SKILL_CONFIG["dim"] * args.width_factor,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.out)
        reloaded_pkg = torch.load(args.out, weights_only=True, map_location="cpu")
        rebuilt = ImportedSkill(width=reloaded_pkg["width"])
        for layer in rebuilt.core.layers:
            for name in ("fc1", "fc2"):
                base = getattr(layer.ffn, name)
                if args.method != "monarch":
                    setattr(layer.ffn, name, CorrectedLinear(base, torch.zeros(base.out_dim, base.in_dim), args.method))
        rebuilt.load_state_dict(reloaded_pkg["state"])
        y_rt = rebuilt.eval()(probe)
        rt_diff = float((y1 - y_rt).abs().max().item())
        if guard.donor_forward_calls != 0 or prefix_diff != 0.0 or rt_diff != 0.0:
            raise ValueError(f"import-skill verification failed: calls={guard.donor_forward_calls} prefix={prefix_diff} rt={rt_diff}")
        raw_bytes = args.out.read_bytes()
        emit(dict(
            command="import-skill",
            status="PASS",
            method=args.method,
            width_factor=args.width_factor,
            donor_forward_calls=guard.donor_forward_calls,
            prefix_max_abs_diff=prefix_diff,
            round_trip_max_abs_diff=rt_diff,
            skill_bytes=len(raw_bytes),
            skill_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        ))
    return 0


def assimilate_command(args):
    archive, receipt = assimilate(
        args.cache_dir, device=args.device, dtype=args.dtype, steps=args.steps,
        kl_weight=args.kl_weight, seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_bytes(args.out, archive)
    emit(dict(command="assimilate", status="PASS", skill_bytes=len(archive),
              skill_sha256=hashlib.sha256(archive).hexdigest(), **receipt))
    return 0


def assimilate_zero_exec_command(args):
    archive, receipt = assimilate_zero_exec(
        args.donor_dir, hf_repo=args.hf_repo, mock_pt=args.mock_pt, seed=args.seed,
        student_dim=args.student_dim, student_layers=args.student_layers, use_engram=args.use_engram,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_bytes(args.out, archive)
    emit(dict(command="assimilate-zero-exec", status="PASS", skill_bytes=len(archive),
              skill_sha256=hashlib.sha256(archive).hexdigest(), **receipt))
    return 0


@torch.inference_mode()
def weight_edit_command(args):
    if not math.isfinite(args.alpha):
        raise ValueError("alpha must be finite")
    skill = restore_skill(args.skill)
    skill.core.fixed_eval_geometry = True
    fault = parsed_hex(args.fault_hex, skill.core.max_seq_len)
    retain = parsed_hex(args.retain_hex, skill.core.max_seq_len)
    diagnostics = inspect_fault_direction(
        skill, torch.tensor([list(fault)]), torch.tensor([list(retain)]), args.layer,
    )
    direction = diagnostics["direction_v"]
    working = skill
    if args.op == "subspace_ablate":
        working = apply_surgical_edit(skill, args.layer, "attn.out_proj", args.op, direction, args.alpha)
    archive, receipt = verify_skill(working)
    if not exact_skill_receipt(receipt):
        raise ValueError("weight-edit skill verification failed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_bytes(args.out, archive)
    emit(dict(command="weight-edit", status="PASS", op=args.op, layer=args.layer, alpha=args.alpha,
              direction_v=direction.tolist(), layers=diagnostics["layers"],
              skill_bytes=len(archive), skill_sha256=hashlib.sha256(archive).hexdigest(), **receipt))
    return 0


def route_prefix_check(route, raw, original):
    cut = raw.size(1) // 2
    alternate = raw.clone()
    alternate[:, cut:] = (alternate[:, cut:] + 17) % 256
    perturbed = route(alternate, include_next=True)
    shortened = route(raw[:, :cut], include_next=True)
    logp, routing = original[:2]
    size = cut + 1
    difference = max(
        (logp[:, :size] - perturbed[0][:, :size]).abs().max().item(),
        (logp[:, :size] - shortened[0]).abs().max().item(),
    )
    if (difference != 0.0 or not torch.equal(routing[:, :size], perturbed[1][:, :size])
            or not torch.equal(routing[:, :size], shortened[1])):
        raise ValueError(f"route-stream exact prefix check failed: {difference}")
    return difference


def route_stream_command(args):
    data = bytearray()
    offsets = []
    for chunk in raw_chunks(args):
        data.extend(chunk)
        if len(data) > MAX_STREAM_CHUNK:
            raise ValueError(f"route-stream allows up to {MAX_STREAM_CHUNK} bytes per stream")
        offsets.append(len(data))
    if not data:
        raise ValueError("provide at least one observed hex byte")

    with forbid_donor_forward() as guard:
        if args.bank:
            bank = load_bank(args.bank)
        else:
            paths = dict(args.skill_snapshot)
            if len(paths) != len(args.skill_snapshot):
                raise ValueError("snapshot skill names must be unique")
            bank = MultiSkillBank.from_unified_snapshots(paths)
        route = getattr(bank, f"forward_{args.router}")
        raw = torch.tensor(list(data), dtype=torch.long).unsqueeze(0)
        result = route(raw, include_next=True)
        if len(data) == 1:
            probe = torch.arange(24, dtype=torch.long).unsqueeze(0)
            difference = route_prefix_check(route, probe, route(probe, include_next=True))
        else:
            difference = route_prefix_check(route, raw, result)
        logp, routing, *measured = result
        names = bank.skill_names
        if routing.ndim == 3:
            weights = routing[0]
            active = weights.argmax(-1)
        else:
            active = routing[0]
            weights = torch.nn.functional.one_hot(active, len(names)).float()
        calls = measured[0] if measured else len(names) * len(data) / (len(data) + 1)
        report = dict(
            command="route-stream", router=args.router, prefix_max_abs_diff=difference,
            avg_active_skills_per_byte=calls, donor_forward_calls=guard.donor_forward_calls,
            chunks=[
                dict(end_offset=offset, active_skill=names[int(active[offset])],
                     routing_weights=dict(zip(names, weights[offset].tolist(), strict=True)),
                     next_byte=int(logp[0, offset].argmax()))
                for offset in offsets
            ],
        )
    emit(report)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", action="version", version=f"Nexus Expanse {VERSION}")
    sub = p.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="run Gate 17 corpus training with immutable resume")
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus-dir", type=Path)
    source.add_argument("--train-file", type=Path)
    train.add_argument("--eval-file", type=Path)
    train.add_argument("--focus-domain", choices=scaling.FOCUS_DOMAINS, default="all")
    train.add_argument("--directory", type=Path, required=True)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    train.add_argument("--epoch-limit", type=int, choices=range(1, 9), default=1)
    for name in ("seed", "context", "epochs", "steps_per_epoch", "batch_per_domain", "eval_blocks"):
        train.add_argument("--" + name.replace("_", "-"), type=int)
    train.add_argument("--dim", type=int, choices=scaling.TRAIN_DIMS, metavar="{32,64,...,1024}")
    train.add_argument("--layers", type=int, choices=scaling.TRAIN_LAYERS)
    scaling.add_lr_arguments(train)
    scaling.add_scaling_arguments(train)
    initial = train.add_mutually_exclusive_group()
    initial.add_argument("--init-skill", type=Path, help="start from a verified local standalone byte skill")
    initial.add_argument("--init-snapshot", type=Path, help="initialize neural weights from a verified unified snapshot")
    train.add_argument("--skill-backbone", action="store_true", help="match a warm-start model from scratch")
    train.add_argument("--use-sdpa", action="store_true", help="causal SDPA training up to 4096 bytes")
    train.add_argument("--until-complete", action="store_true")
    train.add_argument("--save-snapshot", type=Path, help="export an exact checked unified snapshot on completion")
    train.add_argument("--snapshot-quant", choices=SNAPSHOT_QUANTS, default="fp32",
                       help="storage precision for the exported neural weights")
    train.set_defaults(handler=train_command)

    stream = sub.add_parser("stream", help="forecast after observing raw hex byte chunks")
    stream.add_argument("--snapshot", type=Path, required=True)
    stream.add_argument("--snapshot-sha256")
    stream.add_argument("--input-hex", action="append", help="repeat for multiple chunks, or send hex lines on stdin")
    stream.add_argument("--continuation", default=0, type=int, choices=range(33), metavar="0..32")
    stream.add_argument("--no-adapt", action="store_true", help="freeze online memory while observing the prompt")
    stream.add_argument("--save-snapshot", type=Path)
    stream.set_defaults(handler=stream_command)

    generate = sub.add_parser("generate", help="generate UTF-8 text and raw bytes from a unified snapshot")
    generate.add_argument("--snapshot", type=Path, required=True)
    generate.add_argument("--snapshot-sha256")
    generate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    prompt = generate.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt-text")
    prompt.add_argument("--prompt-hex")
    generate.add_argument("--max-bytes", type=int, choices=range(1, MAX_STREAM_CHUNK + 1), default=128,
                          metavar="1..4096")
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-p", type=float, default=0.9)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--repetition-penalty", type=parse_repetition_penalty, default=1.0)
    generate.add_argument("--repeat-window", type=int, choices=range(1, MAX_REPEAT_WINDOW + 1), default=64,
                          metavar="1..512")
    generate.add_argument("--always-wake", action="store_true", help="keep every observed byte on the neural path")
    generate.add_argument("--no-episodic", action="store_true", help="bypass episodic memory during decode")
    generate.add_argument("--adapt", action="store_true", help="adapt online memory in the restored session")
    add_engram_arguments(generate)
    generate.set_defaults(handler=generate_command)

    search = sub.add_parser("search-verified", help="sample byte continuations and verify bounded tasks")
    add_search_runtime_arguments(search, max_bytes=128)
    search.add_argument("--allow-think", action="store_true", help="grade the answer after a leading scratchpad")
    search.add_argument("--verifier", choices=tuple(VERIFIERS.values()), required=True)
    search.add_argument("--prompt-text", required=True)
    search.add_argument("--tests-json", help="bounded Python test list as JSON or a local file")
    search.add_argument("--spec-json", help="structured verifier specification as JSON or a local file")
    search.add_argument("--stop-text", help="stop after emitting this exact UTF-8 delimiter")
    search.set_defaults(handler=search_verified_command)

    livebench = sub.add_parser("eval-livebench", help="score local tasks with deterministic verifiers")
    add_search_runtime_arguments(livebench, max_bytes=MAX_OUTPUT)
    livebench.add_argument("--allow-think", action="store_true", help="grade the answer after a leading scratchpad")
    tasks = livebench.add_mutually_exclusive_group(required=True)
    tasks.add_argument("--tasks-jsonl", type=Path, help="local LiveBench-shaped task rows")
    tasks.add_argument("--builtin-dev-suite", action="store_true", help="use the deterministic offline suite")
    livebench.set_defaults(handler=eval_livebench_command)

    harvest = sub.add_parser("harvest-verified", help="write independently re-verified byte traces")
    add_search_runtime_arguments(harvest, max_bytes=MAX_OUTPUT)
    train_tasks = harvest.add_mutually_exclusive_group(required=True)
    train_tasks.add_argument("--tasks-jsonl", type=Path, help="local training-only task rows")
    train_tasks.add_argument("--builtin-train-suite", action="store_true")
    harvest.add_argument("--out-bin", type=Path, required=True)
    harvest.add_argument("--out-jsonl", type=Path, help="write accepted trace offsets and verification counts")
    harvest.add_argument("--eval-file", type=Path, help="reject any shared 32-byte window with this held stream")
    harvest.set_defaults(handler=harvest_verified_command)

    engram = sub.add_parser("build-engram", help="index raw-byte suffix counts in a new immutable SQLite file")
    engram.add_argument("--input", type=Path, required=True)
    engram.add_argument("--out", type=Path, required=True)
    engram.add_argument("--min-order", type=int, default=4)
    engram.add_argument("--max-order", type=int, default=12)
    engram.add_argument("--min-support", type=int, default=2)
    engram.set_defaults(handler=build_engram_command)

    solve = sub.add_parser("solve", help="search the bounded expression library")
    solve.add_argument("--source-hex", required=True)
    solve.add_argument("--target-hex", required=True)
    solve.add_argument("--depth", required=True, type=int, choices=(1, 2, 3))
    solve.add_argument("--budget", default=6, type=int, choices=range(1, 7))
    solve.add_argument("--agent-snapshot", type=Path)
    solve.add_argument("--save-agent-snapshot", type=Path)
    solve.set_defaults(handler=solve_command)

    verify = sub.add_parser("verify", help="check exact prefix, streaming and snapshot round trip")
    verify.add_argument("--snapshot", type=Path)
    verify.add_argument("--snapshot-sha256")
    verify.add_argument("--save-snapshot", type=Path)
    verify.add_argument("--snapshot-quant", choices=SNAPSHOT_QUANTS,
                        help="re-export the verified snapshot with this neural weight precision")
    verify.set_defaults(handler=verify_command)

    imp = sub.add_parser("import-skill", help="convert a static donor checkpoint into a verified Nexus skill")
    imp.add_argument("--donor", type=Path, required=True)
    imp.add_argument("--method", choices=SKILL_METHODS + ("residual16",), default="residual8")
    imp.add_argument("--width-factor", type=int, choices=(1, 2), default=1)
    imp.add_argument("--out", type=Path, required=True)
    imp.set_defaults(handler=import_skill_command)

    assimilation = sub.add_parser("assimilate", help="train a byte skill from static donor-cache shards")
    assimilation.add_argument("--cache-dir", type=Path, required=True)
    assimilation.add_argument("--out", type=Path, required=True)
    assimilation.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    assimilation.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    assimilation.add_argument("--steps", type=int, default=192)
    assimilation.add_argument("--kl-weight", type=float, default=0.9)
    assimilation.add_argument("--seed", type=int, default=0)
    assimilation.set_defaults(handler=assimilate_command)

    zero_exec = sub.add_parser("assimilate-zero-exec", help="initialize a native skill from static donor shards")
    donor_source = zero_exec.add_mutually_exclusive_group(required=True)
    donor_source.add_argument("--donor-dir", type=Path)
    donor_source.add_argument("--hf-repo", metavar="ORG/MODEL", help="stream indexed HF safetensors from a pinned revision")
    zero_exec.add_argument("--out", type=Path, required=True)
    zero_exec.add_argument("--mock-pt", action="store_true", help="read an offline weights-only test fixture")
    zero_exec.add_argument("--seed", type=int, default=0)
    zero_exec.add_argument("--student-dim", type=int, choices=STUDENT_DIMS, default=32)
    zero_exec.add_argument("--student-layers", type=int, choices=STUDENT_LAYERS, default=2)
    zero_exec.add_argument("--use-engram", action="store_true", default=None,
                           help="include static DeepSeek Engram lookup shards when indexed")
    zero_exec.set_defaults(handler=assimilate_zero_exec_command)

    edit = sub.add_parser("weight-edit", help="inspect or ablate one attention projection")
    edit.add_argument("--skill", type=Path, required=True)
    edit.add_argument("--op", choices=("inspect", "subspace_ablate"), required=True)
    edit.add_argument("--layer", type=int, required=True)
    edit.add_argument("--alpha", type=float, default=1.0)
    edit.add_argument("--fault-hex", default=bytes(range(24)).hex())
    edit.add_argument("--retain-hex", default=bytes(range(128, 152)).hex())
    edit.add_argument("--out", type=Path, required=True)
    edit.set_defaults(handler=weight_edit_command)

    route = sub.add_parser("route-stream", help="forecast raw bytes with a saved multi-skill bank")
    bank_source = route.add_mutually_exclusive_group(required=True)
    bank_source.add_argument("--bank", type=Path)
    bank_source.add_argument("--skill-snapshot", action="append", nargs=2, metavar=("NAME", "PATH"),
                             help="repeat to route named unified snapshots directly")
    route.add_argument(
        "--router", choices=("bayes_bma", "bayes_argmax", "surprise_sparse", "hmm_bma",
                             "hmm_sparse", "hmm_bma_hierarchical", "hmm_sparse_hierarchical",
                             "hmm_conf_sparse_hierarchical"),
        required=True,
    )
    route.add_argument("--input-hex", action="append", help="repeat for chunks, or send hex lines on stdin")
    route.set_defaults(handler=route_stream_command)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command != "train":
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
    try:
        return args.handler(args)
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    except (ValueError, KeyError, OSError, RuntimeError, pickle.UnpicklingError) as exc:
        p.exit(2, f"nexus {args.command}: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
