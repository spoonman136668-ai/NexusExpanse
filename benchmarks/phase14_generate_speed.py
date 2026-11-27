"""Compare fixed-geometry legacy and batched raw-byte decode on one CPU thread."""

import argparse
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import tarfile
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.dev_artifacts import atomic, atomic_bytes, digest
from benchmarks.gate17_gpu_scaling_harness import Geometry, initialize_from_skill, model_for
from benchmarks.gate_custody import peak_rss
from benchmarks.phase3_multi_skill_routing import git
from nexus_expanse.cli import continuation, main as cli_main
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.patching import CompactBigram
from nexus_expanse.shard_stream_assimilation import make_student
from nexus_expanse.unified import NexusExpanseUnified
from scripts.prepare_gpu_corpus import assert_disjoint
from scripts.prepare_real_corpus import MIN_BYTES, prepare

BASE = "bf39629efb44397c24d5d7a10a9b6fd853c0f3c5"
PREREG = "docs/PHASE14_FAST_GENERATE_AND_REAL_CORPUS_PREREG.md"
PROFILES = ((2801, False, 32, 8), (2802, True, 32, 26), (2803, False, 40, 65),
            (2804, True, 64, 148), (2805, False, 48, 111))
DECODE_BYTES = 32
REPETITIONS = 3


def snapshot(seed, warm, window):
    width, layers = (64, 4) if warm else (32, 2)
    model = model_for(Geometry(seed, width, window, 1, 1, 1, 1, layers),
                      "causal_attention_skill" if warm else "causal_attention_monarch", torch.device("cpu"))
    if warm:
        initialize_from_skill(model, make_student(seed + 1, student_dim=width, student_layers=layers))
    counts = CompactBigram.from_dense(np.zeros((256, 256), dtype="<u4"))
    return NexusExpanseUnified(model, counts).snapshot()


def call_cli(*args):
    with redirect_stdout(StringIO()) as output:
        cli_main(list(map(str, args)))
    return json.loads(output.getvalue())


def legacy(payload, prompt):
    with forbid_donor_forward() as guard:
        model = NexusExpanseUnified.restore(payload)
        start = time.perf_counter()
        for value in prompt:
            forecast = model.step(value, adapt=False)
        prefill_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        data = continuation(model, forecast, DECODE_BYTES)
        return dict(generated_hex=data.hex(), prefill_ms=prefill_ms,
                    ms_per_byte=(time.perf_counter() - start) * 1000 / DECODE_BYTES,
                    donor_forward_calls=guard.donor_forward_calls)


def row(root, seed, warm, window, length):
    payload = snapshot(seed, warm, window)
    path = root / f"{seed}.pt"
    atomic_bytes(path, payload)
    prompt = random.Random(seed).randbytes(length)
    commands = {
        "baseline": lambda: legacy(payload, prompt),
        "optimized": lambda: call_cli("generate", "--snapshot", path, "--device", "cpu",
                                      "--prompt-hex", prompt.hex(), "--max-bytes", DECODE_BYTES),
    }
    for run in commands.values():
        run()
    trials = {name: [] for name in commands}
    names = list(commands)
    for index in range(REPETITIONS):
        for name in (names if index % 2 == 0 else names[::-1]):
            trials[name].append(commands[name]())
    expected = call_cli("stream", "--snapshot", path, "--no-adapt", "--input-hex", prompt.hex(),
                        "--continuation", DECODE_BYTES)["continuation_hex"]
    mismatches = sum(
        a != b for reports in trials.values() for report in reports
        for a, b in zip(bytes.fromhex(report["generated_hex"]), bytes.fromhex(expected), strict=True)
    )
    baseline, optimized = (statistics.median(row["ms_per_byte"] for row in trials[name]) for name in names)
    return dict(seed=seed, layout="warm-start" if warm else "standard", window=window, prompt_hex=prompt.hex(),
                prompt_bytes=length, generated_bytes=DECODE_BYTES, stream_hex=expected, snapshot_sha256=digest(path),
                byte_mismatches=mismatches, donor_forward_calls=sum(r["donor_forward_calls"] for t in trials.values()
                                                                  for r in t),
                baseline_ms_per_byte=baseline, optimized_ms_per_byte=optimized,
                decode_speedup=baseline / optimized, trials=trials)


def package_repo(root):
    source = root / "source"
    source.mkdir()
    archived = git("archive", BASE, "docs", "nexus_expanse", "benchmarks", "scripts", "tests")
    with tarfile.open(fileobj=BytesIO(archived), mode="r:") as tar:
        tar.extractall(source, filter="data")
    start = time.perf_counter()
    directory = root / "corpus"
    report = prepare(directory, local={"prose": source / "docs", "code": source}, hub={},
                     seed=42, fraction=0.10, limit=1_500_000, math_fallback=True)
    outputs = {}
    for name, item in report["files"].items():
        path = directory / item["file"]
        raw = path.read_bytes()
        if len(raw) != item["bytes"] or digest(path) != item["sha256"] or item["documents"] < 1:
            raise ValueError(f"corpus artifact disagrees with its manifest: {name}")
        outputs[name] = raw
    assert_disjoint([raw for name, raw in outputs.items() if name.startswith("train_")],
                    {name: raw for name, raw in outputs.items() if name.startswith("eval_")})
    return dict(source_commit=BASE, wall_seconds=time.perf_counter() - start, manifest=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    head = git("rev-parse", "HEAD").decode().strip()
    tree = git("rev-parse", "HEAD^{tree}").decode().strip()
    out = args.out.resolve()
    intent = out.with_suffix(".intent.json")
    if (out.is_relative_to(ROOT) or out.exists() or intent.exists() or git("status", "--porcelain")
            or git("rev-parse", "HEAD^").decode().strip() != BASE or not git("ls-tree", head, PREREG)):
        parser.error("run once from the clean preregistration commit with a new external output")
    identity = dict(base=BASE, source_commit=head, source_tree=tree, prereg_sha256=digest(ROOT / PREREG))
    atomic(intent, identity)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="nexus-phase14-") as temp:
        root = Path(temp)
        rows = [row(root, *profile) for profile in PROFILES]
        corpus = package_repo(root)
    if git("rev-parse", "HEAD").decode().strip() != head or git("status", "--porcelain"):
        parser.error("checkout changed during scoring")
    median_ratio = statistics.median(r["decode_speedup"] for r in rows)
    exact = sum(r["byte_mismatches"] == 0 for r in rows)
    calls = sum(r["donor_forward_calls"] for r in rows)
    corpus_ok = corpus["manifest"]["global_overlap_32b"] == 0 and all(
        item["bytes"] >= MIN_BYTES for item in corpus["manifest"]["files"].values()
    )
    summary = dict(exact_seeds=exact, byte_mismatches=sum(r["byte_mismatches"] for r in rows),
                   donor_forward_calls=calls, median_decode_speedup=median_ratio, corpus_ok=corpus_ok,
                   scored_engineering="PASS" if exact == 5 and not calls and corpus_ok and median_ratio >= 2 else "FAIL")
    result = dict(schema="nexus.phase14-fast-generate.v1", **identity, platform=platform.platform(),
                  torch=str(torch.__version__), threads=torch.get_num_threads(), device="cpu",
                  external_downloads=0, wall_seconds=time.perf_counter() - start, peak_rss_bytes=peak_rss(),
                  seeds=rows, corpus=corpus, summary=summary)
    atomic(out, result)
    print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
