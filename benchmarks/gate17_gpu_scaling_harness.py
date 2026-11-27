"""Bounded raw-byte corpus training and unified-runtime qualification."""
import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from benchmarks.dev_artifacts import atomic, digest
from benchmarks.gate_custody import EpochRun, peak_rss, prepare_directory, update
from benchmarks.unified_checks import exactness
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.hybrid_backbone import HybridDeltaAttentionCore
from nexus_expanse.jepa import state_hash
from nexus_expanse.patching import CompactBigram
from nexus_expanse.unified import NexusExpanseUnified
from nexus_expanse.weight_editing import _affine_parameters, restore_skill

DOMAINS = ("math", "code", "prose")
GATE15 = ROOT / "evidence/gate15-hybrid-dev-001.json"
PHASE1 = "57eded620df922f250495dbb7eeaa280630b1881"


@dataclass(frozen=True)
class Geometry:
    seed: int
    dim: int
    context: int
    epochs: int
    steps_per_epoch: int
    batch_per_domain: int
    eval_blocks: int
    layers: int = 2


SMOKE = Geometry(seed=42, dim=32, context=32, epochs=2, steps_per_epoch=2,
                 batch_per_domain=1, eval_blocks=2)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def load_corpora(args):
    fields = {(split, domain): getattr(args, f"{split}_{domain}")
              for split in ("train", "eval") for domain in DOMAINS}
    if args.smoke:
        if any(fields.values()):
            raise ValueError("the frozen smoke supplies all six input spans")
        code = git("show", f"{PHASE1}:nexus_expanse/substrate.py")
        prose = git("show", f"{PHASE1}:README.md")
        math_data = lambda lo, hi: b"".join(f"{i}+2={i + 2}\n".encode() for i in range(lo, hi))
        train = dict(math=math_data(0, 50), code=code[:384], prose=prose[:384])
        evaluation = dict(math=math_data(70, 90), code=code[-128:], prose=prose[-128:])
    else:
        if not all(fields.values()):
            raise ValueError("provide separate --train-* and --eval-* files for math, code, and prose")
        train, evaluation = ({d: fields[split, d].read_bytes() for d in DOMAINS}
                             for split in ("train", "eval"))
    manifest = {}
    for split, data in (("train", train), ("eval", evaluation)):
        if any(len(data[d]) < 2 for d in DOMAINS):
            raise ValueError("every corpus needs at least two raw bytes")
        manifest[split] = {d: dict(bytes=len(data[d]), sha256=hashlib.sha256(data[d]).hexdigest())
                           for d in DOMAINS}
    train_hashes = {r["sha256"] for r in manifest["train"].values()}
    eval_hashes = {r["sha256"] for r in manifest["eval"].values()}
    if train_hashes & eval_hashes:
        raise ValueError("train and evaluation corpora contain identical full files")
    return train, evaluation, manifest


def device_for(requested, dtype):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this PyTorch/device environment")
        if dtype == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError("the CUDA device lacks native BF16 support")
    return torch.device(requested)


def selected_backbone(*, use_sdpa=False, skill_backbone=False):
    prior = json.loads(GATE15.read_text())
    if prior["schema"] != "nexus-gate15-hybrid-dev-v1" or prior["status"] not in ("PASS", "MIXED", "FAIL"):
        raise ValueError("invalid Gate 15 evidence")
    if prior["status"] == "PASS":
        if not all((prior["parity"]["passed"], prior["reasoning_bar"], prior["speed_cache_bar"])):
            raise ValueError("Gate 15 PASS is inconsistent with its raw criteria")
        if not use_sdpa and not skill_backbone:
            return "hybrid"
    return "causal_attention_skill" if skill_backbone else "causal_attention_monarch"


def model_for(geometry, backbone, device, *, use_sdpa=False):
    torch.manual_seed(geometry.seed)
    common = dict(dim=geometry.dim, n_heads=4, ffn_mult=2, max_seq_len=geometry.context)
    if backbone == "hybrid":
        model = HybridDeltaAttentionCore(**common)
    else:
        skill_shape = backbone == "causal_attention_skill"
        model = NexusExpanseByteCore(
            **common, n_layers=geometry.layers, dropout=0.0, enable_skip=False, use_sdpa=use_sdpa,
            ffn_kind="dense" if skill_shape else "monarch", untied_readout=skill_shape,
        )
    return model.to(device)


def read_init_skill(path):
    with path.open("rb") as source:
        payload = source.read(NexusExpanseUnified.SNAPSHOT_LIMIT + 1)
    if not 0 < len(payload) <= NexusExpanseUnified.SNAPSHOT_LIMIT:
        raise ValueError("initial skill is empty or exceeds the snapshot size limit")
    return restore_skill(payload), hashlib.sha256(payload).hexdigest()


@torch.no_grad()
def initialize_from_skill(model, skill):
    if model.dim != skill.readout.in_features or len(model.layers) != len(skill.core.layers):
        raise ValueError("initial skill width or depth does not match training geometry")
    # Materialize each correction once into the native affine layout.
    target = model.state_dict()
    for name, tensor in skill.core.state_dict().items():
        if ".ffn.fc1." in name or ".ffn.fc2." in name:
            continue
        if name == "pos_emb.weight":
            length = min(len(tensor), len(target[name]))
            target[name][:length].copy_(tensor[:length])
        else:
            target[name].copy_(tensor)
    model.readout.weight.copy_(skill.readout.weight)
    for layer, imported in zip(model.layers, skill.core.layers, strict=True):
        for name in ("fc1", "fc2"):
            weight, bias = _affine_parameters(getattr(imported.ffn, name))
            native = getattr(layer.ffn, name)
            native.weight.copy_(weight)
            if bias is None:
                native.bias.zero_()
            else:
                native.bias.copy_(bias)


def block_count(data, context):
    return (len(data) - 2) // context + 1


def shuffled_index(index, count, seed):
    # An affine bijection uses bounded memory even for very large input files.
    cycle, item = divmod(index, count)
    rng = random.Random(f"{seed}:{cycle}")
    offset = rng.randrange(count)
    stride = rng.randrange(1, count + 1)
    while math.gcd(stride, count) != 1:
        stride = stride % count + 1
    return (offset + stride * item) % count


def make_batch(rows, context, device):
    chunks = [data[index * context:index * context + context + 1] for data, index in rows]
    batch = torch.zeros((len(rows), context + 1), dtype=torch.long)
    lengths = torch.tensor([len(chunk) - 1 for chunk in chunks])
    for i, chunk in enumerate(chunks):
        batch[i, :len(chunk)] = torch.tensor(list(chunk))
    keep = torch.arange(context)[None, :] < lengths[:, None]
    return batch.to(device), keep.to(device), int(lengths.sum())


def training_rows(data, geometry, step):
    rows = []
    for i, domain in enumerate(DOMAINS):
        raw = data[domain]
        count = block_count(raw, geometry.context)
        start = step * geometry.batch_per_domain
        rows.extend((raw, shuffled_index(j, count, geometry.seed + i))
                    for j in range(start, start + geometry.batch_per_domain))
    return rows


def loss_for(model, batch, keep, dtype, device, *, reduction="mean", prior=None, alpha=0.0):
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=dtype == "bf16"):
        out = model.forward_neural(batch[:, :-1])[0]
        if alpha:
            log_native = out[keep].float().log_softmax(-1) + math.log1p(-alpha)
            log_prior = prior[batch[:, :-1][keep]] + math.log(alpha)
            return F.nll_loss(torch.logaddexp(log_native, log_prior), batch[:, 1:][keep], reduction=reduction)
        return F.cross_entropy(out[keep], batch[:, 1:][keep], reduction=reduction)


def vram(device):
    if device.type != "cuda":
        return dict(peak_cuda_allocated_bytes=None, peak_cuda_reserved_bytes=None)
    return dict(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))


def build_unified(model, train):
    """Build the CPU runtime with the Gate 17 fitted compact and episodic inputs."""
    counts = np.zeros((256, 256), dtype="<u4")
    examples = [train[d][:512] for d in DOMAINS]
    for raw in examples:
        b = np.frombuffer(raw, dtype=np.uint8)
        np.add.at(counts, (b[:-1], b[1:]), 1)
    # Transfer the same trained neural state into the CPU FP32 runtime.
    unified = NexusExpanseUnified(model.cpu(), CompactBigram.from_dense(counts))
    for raw in examples:
        for i in range(1, len(raw)):
            unified.episodic.observe_transition(raw[max(0, i - 8):i], raw[i])
    return unified


@torch.inference_mode()
def cold_evaluate(path, args, geometry, backbone, train, evaluation, device):
    start = time.perf_counter()
    model = model_for(geometry, backbone, device, use_sdpa=args.use_sdpa).eval()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if not args.use_sdpa:
        del state
    restored = state_hash(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - start
    if args.dtype == "bf16":
        torch.backends.mha.set_fastpath_enabled(False)
    per_domain = {}
    for domain in DOMAINS:
        data = evaluation[domain]
        n = min(geometry.eval_blocks, block_count(data, geometry.context))
        elapsed = time.perf_counter()
        total_bytes = 0
        total_loss = 0.0
        for i in range(0, n, geometry.batch_per_domain):
            batch, keep, valid = make_batch([(data, j) for j in range(i, min(n, i + geometry.batch_per_domain))],
                                            geometry.context, device)
            total_loss += float(loss_for(model, batch, keep, args.dtype, device, reduction="sum"))
            total_bytes += valid
        elapsed = time.perf_counter() - elapsed
        per_domain[domain] = dict(blocks=n, target_bytes=total_bytes, bpb=total_loss / (math.log(2) * total_bytes),
                                  wall_seconds=elapsed, bytes_per_second=total_bytes / elapsed)
    gpu = vram(device)
    torch.backends.mha.set_fastpath_enabled(True)

    if args.use_sdpa:
        # The CPU unified runtime uses the legacy cache's exact reference geometry.
        del model
        model = model_for(geometry, backbone, torch.device("cpu")).eval()
        model.load_state_dict(state)
    unified = build_unified(model, train)
    raw = evaluation["prose"][:min(32, geometry.context)]
    checkpoint = unified.snapshot()
    parity = exactness(unified, NexusExpanseUnified.restore(checkpoint), raw)
    if any(parity.values()):
        raise ValueError(f"CPU unified inference failed exact prefix/streaming checks: {parity}")
    return dict(state_hash=restored, restore_seconds=load_seconds, peak_rss_bytes=peak_rss(),
                worker_pid=os.getpid(), validation=per_domain, parity=parity,
                unified_initial_serialized_bytes=len(checkpoint), parity_bytes=len(raw), **gpu)


def experiment_identity(args, geometry, backbone, manifest, device, *, init_skill_sha256=None):
    if git("status", "--porcelain"):
        raise ValueError("train from a clean committed checkout")
    cuda = None if device.type == "cpu" else dict(
        name=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)),
        runtime=torch.version.cuda,
    )
    identity = dict(source_sha=git("rev-parse", "HEAD").decode().strip(),
                tree=git("rev-parse", "HEAD^{tree}").decode().strip(), gate15_sha256=digest(GATE15),
                backbone=backbone, corpora=manifest, geometry=asdict(geometry), smoke=args.smoke,
                dtype=args.dtype, device=device.type, cuda=cuda, threads=torch.get_num_threads(),
                platform=platform.platform(), torch=str(torch.__version__), numpy=np.__version__,
                python=sys.version)
    if args.use_sdpa:
        identity["use_sdpa"] = True
    if backbone == "causal_attention_skill":
        identity["skill_backbone"] = True
    if init_skill_sha256 is not None:
        identity["init_skill_sha256"] = init_skill_sha256
    return identity


def run_slice(spec, args, geometry, backbone, train, device, argv, *, skill=None, guard):
    model = model_for(geometry, backbone, torch.device("cpu") if skill is not None else device,
                      use_sdpa=args.use_sdpa)
    prior = None
    prior_alpha = 0.0
    if skill is not None:
        initialize_from_skill(model, skill)
        model.to(device)
        if skill.prior_alpha:
            prior = skill.static_byte_prior.to(device)
            prior_alpha = skill.prior_alpha
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    epoch_trace = {}
    cold_report = {}

    def epoch(number):
        epoch_bytes = 0
        start_hash = state_hash(model)
        first = (number - 1) * geometry.steps_per_epoch
        steps = sorted({int(state["step"]) for state in optimizer.state.values()})
        if steps != ([first] if first else []):
            raise ValueError("optimizer step counter does not match the epoch cursor")
        losses = []
        alphas = []
        for index in range(first, first + geometry.steps_per_epoch):
            rows = training_rows(train, geometry, index)
            batch, keep, valid = make_batch(rows, geometry.context, device)
            optimizer.zero_grad(set_to_none=True)
            remaining = geometry.steps_per_epoch - 1 - index
            alpha = prior_alpha * remaining / (geometry.steps_per_epoch - 1) if number == 1 and remaining > 0 else 0.0
            alphas.append(alpha)
            losses.append(update(model, optimizer, loss_for(model, batch, keep, args.dtype, device,
                                                           prior=prior, alpha=alpha)))
            epoch_bytes += valid
        epoch_trace.update(target_bytes=epoch_bytes, start_model_hash=start_hash,
                           end_model_hash=state_hash(model), optimizer_steps_at_start=first)
        if skill is not None:
            epoch_trace["prior_alpha_schedule"] = alphas
        return losses

    def metrics(number, seconds):
        return dict(epoch_trace, bytes_per_second=epoch_trace["target_bytes"] / seconds,
                    worker_pid=os.getpid(), **vram(device))

    def cold(path):
        if device.type == "cuda":
            # Release training tensors before the fresh evaluator uses the device.
            model.cpu()
            optimizer.state.clear()
            optimizer.param_groups.clear()
            torch.cuda.empty_cache()
        result = subprocess.check_output([sys.executable, __file__, *argv, "--cold", str(path)],
                                         text=True)
        cold_report.update(json.loads(result))
        if cold_report["worker_pid"] == os.getpid():
            raise ValueError("restoration did not use a fresh process")
        return cold_report["state_hash"], cold_report["restore_seconds"], cold_report["peak_rss_bytes"]

    record = spec.advance(model, optimizer, args.epoch_limit, epoch, cold,
                          lambda: dict(fresh_evaluation=cold_report, donor_forward_calls=guard.donor_forward_calls),
                          state_hash(model),
                          epoch_metrics=metrics)
    if record:
        print(json.dumps(dict(completed_epochs=len(record["epochs"]), target_bytes=sum(
            r["telemetry"]["target_bytes"] for r in record["epochs"]), cold_equal=record["cold_equal"],
            parity=record["fresh_evaluation"]["parity"])))


def emit_smoke(spec, args, device, dest):
    if not args.smoke or device.type != "cpu" or args.dtype != "fp32":
        raise ValueError("smoke evidence requires the frozen CPU FP32 profile")
    report = spec.read()
    epochs = report["epochs"]
    workers = [r["telemetry"]["worker_pid"] for r in epochs]
    if len(epochs) != 2 or len(set(workers)) != 2:
        raise ValueError("qualification requires two separate one-epoch training invocations")
    first, second = [r["telemetry"] for r in epochs]
    if (first["end_model_hash"] != second["start_model_hash"]
            or second["optimizer_steps_at_start"] != SMOKE.steps_per_epoch
            or second["end_model_hash"] != report["final_hash"]):
        raise ValueError("restarted weights or optimizer could not be verified")
    if report["fresh_evaluation"]["worker_pid"] in workers or any(report["fresh_evaluation"]["parity"].values()):
        raise ValueError("fresh-process evaluation or exact parity did not qualify")
    atomic(dest, dict(schema="nexus-gate17-harness-smoke-dev-v1", qualification=f"{platform.system().upper()}_CPU_FP32_SMOKE",
                      status="PASS", train_workers=workers, exact_resume=True, **report))
    print(json.dumps(dict(status="PASS", cold_equal=report["cold_equal"],
                          parity=report["fresh_evaluation"]["parity"])))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--use-sdpa", action="store_true")
    p.add_argument("--init-skill", type=Path, help="initialize from a local standalone byte skill")
    p.add_argument("--layers", type=int, choices=(2, 4, 6))
    p.add_argument("--skill-backbone", action="store_true", help="use dense FFNs and an untied readout from scratch")
    p.add_argument("--epoch-limit", type=int, default=1, choices=range(1, 9))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dim", type=int, default=64, choices=range(32, 257, 32), metavar="{32,64,...,256}")
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=32)
    p.add_argument("--batch-per-domain", type=int, default=2)
    p.add_argument("--eval-blocks", type=int, default=16)
    for split in ("train", "eval"):
        for domain in DOMAINS:
            p.add_argument(f"--{split}-{domain}", type=Path)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--until-complete", action="store_true",
                        help="run bounded slices in the foreground until the final verified receipt")
    action.add_argument("--smoke-evidence", type=Path)
    action.add_argument("--cold", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    # Set CUDA's deterministic cuBLAS configuration before initializing CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    geometry = SMOKE if args.smoke else Geometry(
        args.seed, args.dim, args.context, args.epochs, args.steps_per_epoch, args.batch_per_domain,
        args.eval_blocks, args.layers or 2,
    )
    max_context = 4096 if args.use_sdpa else 512
    if not (8 <= geometry.context <= max_context and 1 <= geometry.epochs <= 5000
            and 1 <= geometry.steps_per_epoch <= 64 and 1 <= geometry.batch_per_domain <= 8
            and 1 <= geometry.eval_blocks <= 128):
        p.error("invalid context, epoch, step, batch or evaluation size")
    if args.until_complete:
        # Keep the foreground parent free of any CUDA context or corpus buffers.
        directory = prepare_directory(args.directory, ROOT)
        child = [sys.executable, __file__, *(arg for arg in argv if arg != "--until-complete")]
        for _ in range(geometry.epochs):
            subprocess.run(child, check=True)
            if (directory / "gate17.json").exists():
                return
        raise RuntimeError("bounded slices ended without a final receipt")
    with forbid_donor_forward() as guard:
        try:
            if args.smoke and (args.init_skill or args.layers or args.skill_backbone):
                raise ValueError("the frozen smoke profile uses its original training architecture")
            device = device_for(args.device, args.dtype)
            train, evaluation, manifest = load_corpora(args)
            skill, skill_sha = read_init_skill(args.init_skill) if args.init_skill else (None, None)
            if skill is not None:
                if args.layers is None:
                    geometry = replace(geometry, layers=len(skill.core.layers))
                if geometry.dim != skill.readout.in_features or geometry.layers != len(skill.core.layers):
                    raise ValueError("initial skill width or depth does not match training geometry")
        except ValueError as exc:
            p.error(str(exc))

        backbone = selected_backbone(use_sdpa=args.use_sdpa, skill_backbone=bool(skill or args.skill_backbone))
        if backbone == "hybrid" and geometry.layers != 2:
            p.error("select --skill-backbone or --use-sdpa for a different layer depth")
        if args.cold:
            result = cold_evaluate(args.cold, args, geometry, backbone, train, evaluation, device)
            print(json.dumps(dict(result, donor_forward_calls=guard.donor_forward_calls), allow_nan=False))
            return
        directory = prepare_directory(args.directory, ROOT)
        ident = experiment_identity(args, geometry, backbone, manifest, device, init_skill_sha256=skill_sha)
        spec = EpochRun(directory, "gate17", ident, geometry.epochs, geometry.steps_per_epoch, device=device)
        if args.smoke_evidence:
            emit_smoke(spec, args, device, args.smoke_evidence)
        else:
            run_slice(spec, args, geometry, backbone, train, device, argv, skill=skill, guard=guard)


if __name__ == "__main__":
    main()
