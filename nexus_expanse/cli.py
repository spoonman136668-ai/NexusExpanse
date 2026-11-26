"""Nexus Expanse v1.0-rc1 raw-byte developer CLI."""
import argparse
import hashlib
import itertools
import json
import math
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
from nexus_expanse.checkpoint_transport import CONFIG as SKILL_CONFIG, CorrectedLinear, ImportedSkill, METHODS as SKILL_METHODS, load_skill
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.multi_skill_router import load_bank
from nexus_expanse.cache_assimilation import assimilate
from nexus_expanse.shard_stream_assimilation import assimilate_zero_exec
from nexus_expanse.weight_editing import (
    apply_surgical_edit, exact_skill_receipt, inspect_fault_direction, restore_skill, verify_skill,
)

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
            or identity["backbone"] != scaling.selected_backbone(use_sdpa=identity.get("use_sdpa", False))
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
    if args.use_sdpa:
        command.append("--use-sdpa")
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
    archive, receipt = assimilate_zero_exec(args.donor_dir, hf_repo=args.hf_repo, mock_pt=args.mock_pt, seed=args.seed)
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
        bank = load_bank(args.bank)
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
    train.add_argument("--corpus-dir", type=Path, required=True)
    train.add_argument("--directory", type=Path, required=True)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    train.add_argument("--epoch-limit", type=int, choices=range(1, 9), default=1)
    for name in TRAIN_OPTIONS[3:]:
        train.add_argument("--" + name.replace("_", "-"), type=int)
    train.add_argument("--use-sdpa", action="store_true", help="causal SDPA training up to 4096 bytes")
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
    route.add_argument("--bank", type=Path, required=True)
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
