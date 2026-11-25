"""Nexus Expanse v1.0-rc1 raw-byte developer CLI."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from benchmarks import gate17_gpu_scaling_harness as scaling
from benchmarks.dev_artifacts import atomic_bytes, digest
from benchmarks.gate_custody import EpochRun
from benchmarks.unified_checks import exactness
from nexus_expanse.expr_verifier import EXPRESSIONS, MAX_LENGTH, run_pipeline, verify as verify_expression
from nexus_expanse.jepa import state_hash
from nexus_expanse.patching import CompactBigram
from nexus_expanse.unified import NexusExpanseUnified
from nexus_expanse.verifier_agent import VerifierGatedNexusAgent

VERSION = "1.0-rc1"
TRAIN_OPTIONS = ("device", "dtype", "epoch_limit", "seed", "dim", "context",
                 "epochs", "steps_per_epoch", "batch_per_domain", "eval_blocks")
EXPRESSION_LIBRARY = dict(zip(b"amdrvX", EXPRESSIONS, strict=True))
MAX_STREAM_CHUNK = 4096


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


def snapshot_checks(model):
    original = model.snapshot()
    restored = NexusExpanseUnified.restore(original)
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
    return original, dict(parity=parity, round_trip_probabilities=round_trip, exact_state=state_equal)


def default_model():
    geometry = scaling.Geometry(42, 32, 32, 1, 1, 1, 1)
    model = scaling.model_for(geometry, "causal_attention_monarch", torch.device("cpu"))
    counts = CompactBigram.from_dense(np.zeros((256, 256), dtype="<u4"))
    return NexusExpanseUnified(model, counts)


def verify_command(args):
    if args.snapshot_sha256 is not None and not args.snapshot:
        raise ValueError("--snapshot-sha256 requires --snapshot")
    if args.snapshot:
        raw = read_snapshot(args.snapshot, NexusExpanseUnified.SNAPSHOT_LIMIT, args.snapshot_sha256)
        model = NexusExpanseUnified.restore(raw)
    else:
        raw = None
        model = default_model()
    snapshot, report = snapshot_checks(model)
    if raw is not None:
        snapshot = raw
    if args.save_snapshot:
        atomic_bytes(args.save_snapshot, snapshot)
    emit(dict(command="verify", status="PASS", snapshot_bytes=len(snapshot),
              snapshot_sha256=hashlib.sha256(snapshot).hexdigest(), **report))


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


def stream_command(args):
    data = read_snapshot(args.snapshot, NexusExpanseUnified.SNAPSHOT_LIMIT, args.snapshot_sha256)
    snapshot_checks(NexusExpanseUnified.restore(data))
    model = NexusExpanseUnified.restore(data)
    observed = 0
    for chunk in raw_chunks(args):
        for byte in chunk:
            forecast = model.step(byte)
        observed += len(chunk)
        indices = torch.argsort(forecast.mixed, descending=True, stable=True)[:5]
        predicted = [dict(byte=int(b), probability=float(forecast.mixed[b])) for b in indices]
        speculative = bytearray()
        if args.continuation:
            shadow = NexusExpanseUnified.restore(model.snapshot())
            output = forecast
            for _ in range(args.continuation):
                selected = int(output.mixed.argmax())
                speculative.append(selected)
                output = shadow.step(selected, adapt=False)
        emit(dict(command="stream", observed_hex=chunk.hex(), next=predicted,
                  continuation_hex=speculative.hex(), arbitration_updates=model.arbitrator.updates))
    if not observed:
        raise ValueError("provide at least one observed hex byte")
    if args.save_snapshot:
        snapshot, report = snapshot_checks(model)
        atomic_bytes(args.save_snapshot, snapshot)
        emit(dict(command="stream", saved_snapshot=str(args.save_snapshot),
                  snapshot_sha256=hashlib.sha256(snapshot).hexdigest(), **report))


def corpus_paths(directory):
    return {f"{split}_{domain}": directory / f"{domain}-{split}.bin"
            for split in ("train", "eval") for domain in scaling.DOMAINS}


def export_trained_snapshot(directory, paths, target):
    report = json.loads((directory / "gate17.json").read_text())
    identity = report["identity"]
    geometry = scaling.Geometry(**identity["geometry"])
    if (scaling.git("status", "--porcelain")
            or identity["source_sha"] != scaling.git("rev-parse", "HEAD").decode().strip()
            or identity["tree"] != scaling.git("rev-parse", "HEAD^{tree}").decode().strip()
            or identity["backbone"] != scaling.selected_backbone()
            or identity["gate15_sha256"] != digest(scaling.GATE15)):
        raise ValueError("training result does not bind the current committed source")
    train, _, manifest = scaling.load_corpora(SimpleNamespace(smoke=False, **paths))
    if identity["corpora"] != manifest:
        raise ValueError("training corpus changed before export")
    spec = EpochRun(directory, "gate17", identity, geometry.epochs, geometry.steps_per_epoch)
    verified = spec.read()
    if any(verified["fresh_evaluation"]["parity"].values()):
        raise ValueError("training checkpoint has no exact parity evidence")
    model = scaling.model_for(geometry, identity["backbone"], torch.device("cpu"))
    model.load_state_dict(torch.load(spec.checkpoint, weights_only=True, map_location="cpu"))
    if state_hash(model) != verified["final_hash"]:
        raise ValueError("restored neural weights disagree with the committed training receipt")
    torch.set_num_threads(1)
    snapshot, checks = snapshot_checks(scaling.build_unified(model, train))
    atomic_bytes(target, snapshot)
    emit(dict(command="train", saved_snapshot=str(target), neural_state_sha256=verified["final_hash"],
              snapshot_sha256=hashlib.sha256(snapshot).hexdigest(), **checks))


def train_command(args):
    paths = corpus_paths(args.corpus_dir)
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"missing raw corpus file: {path}")
    if args.save_snapshot and args.save_snapshot.exists():
        raise FileExistsError(args.save_snapshot)
    command = ["--directory", str(args.directory)]
    for name in TRAIN_OPTIONS:
        value = getattr(args, name)
        if value is not None:
            command.extend(("--" + name.replace("_", "-"), str(value)))
    for name, path in paths.items():
        command.extend(("--" + name.replace("_", "-"), str(path)))
    if args.until_complete:
        command.append("--until-complete")
    scaling.main(command)
    if args.save_snapshot and (args.directory / "gate17.json").is_file():
        export_trained_snapshot(args.directory.resolve(), paths, args.save_snapshot)


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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", action="version", version=f"Nexus Expanse {VERSION}")
    sub = p.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="run Gate 17 corpus training with immutable resume")
    train.add_argument("--corpus-dir", type=Path, required=True)
    train.add_argument("--directory", type=Path, required=True)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    train.add_argument("--epoch-limit", type=int, choices=range(1, 9), default=1)
    for name in TRAIN_OPTIONS[3:]:
        train.add_argument("--" + name.replace("_", "-"), type=int)
    train.add_argument("--until-complete", action="store_true")
    train.add_argument("--save-snapshot", type=Path, help="export an exact checked unified snapshot on completion")
    train.set_defaults(handler=train_command)

    stream = sub.add_parser("stream", help="forecast after observing raw hex byte chunks")
    stream.add_argument("--snapshot", type=Path, required=True)
    stream.add_argument("--snapshot-sha256")
    stream.add_argument("--input-hex", action="append", help="repeat for multiple chunks, or send hex lines on stdin")
    stream.add_argument("--continuation", default=0, type=int, choices=range(33), metavar="0..32")
    stream.add_argument("--save-snapshot", type=Path)
    stream.set_defaults(handler=stream_command)

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
    verify.set_defaults(handler=verify_command)
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
    except (ValueError, OSError, RuntimeError, pickle.UnpicklingError) as exc:
        p.exit(2, f"nexus {args.command}: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
