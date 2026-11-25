"""Gate 14 scorecard for one integrated Nexus Expanse DEV kernel per seed."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.gate4_corpus import MANIFEST as CODE_MANIFEST, read_split
from benchmarks.gate4a_dev import digest as probability_digest, fixture
from benchmarks.gate4b_dev import prepare
from benchmarks.gate9_jepa_dev import atomic, digest
from benchmarks.gate11_selective_kuramoto_dev import capacity_eval, identity, manifest, setup
from benchmarks.gate12_deltanet_dev import ATTENTION, DELTA, SEEDS, model_for, run_spec
from benchmarks.gate_custody import peak_rss, prepare_directory, writer
from benchmarks.unified_checks import EXPERTS, exactness
from nexus_expanse.consolidation import ConsolidatingMemory
from nexus_expanse.jepa import state_hash
from nexus_expanse.patching import CompactBigram
from nexus_expanse.unified import NexusExpanseUnified

GATE12 = ROOT / "evidence/gate12-deltanet-dev-001.json"
GATE13 = ROOT / "evidence/gate13-jepa-proven-backbone-dev-001.json"
GATE8 = ROOT / "evidence/gate8a-hopfield-vsa-dev-001.json"
CONTINUATION = b"\n    return result\n"


def experiment_identity():
    gate12 = json.loads(GATE12.read_text())
    gate13 = json.loads(GATE13.read_text())
    if (gate12["schema"] != "nexus-gate12-deltanet-dev-v1"
            or gate13["schema"] != "nexus-gate13-jepa-proven-backbone-dev-v1"
            or gate13["identity"]["gate12_sha256"] != digest(GATE12)):
        raise ValueError("upstream gate evidence or linking digest drifted")
    ident = dict(identity(), gate12_sha256=digest(GATE12), gate13_sha256=digest(GATE13),
                 gate8_sha256=digest(GATE8), real_corpus_manifest_sha256=digest(CODE_MANIFEST))
    return ident, gate12


def subrun(*args):
    return json.loads(subprocess.check_output([sys.executable, __file__, *map(str, args)], text=True))


def evaluate_corpus(unified, blocks):
    start, cpu = time.perf_counter(), time.process_time()
    raw_bytes = transitions = 0
    totals = dict.fromkeys(EXPERTS, 0.0)
    block_rows = []
    initial_updates, initial_wakes = unified.arbitrator.updates, unified.reader.deep_calls
    for block in blocks:
        unified.reset_context()
        cursor = unified.step(block[0])
        accum = dict.fromkeys(EXPERTS, 0.0)
        for byte in block[1:]:
            for name in EXPERTS:
                accum[name] -= math.log2(max(float(getattr(cursor, name)[byte]), 1e-300))
            cursor = unified.step(byte)
        count = len(block) - 1
        block_rows.append(dict(sha256=hashlib.sha256(block).hexdigest(), transitions=count,
                               bpb={k: v / count for k, v in accum.items()}))
        raw_bytes += len(block)
        transitions += count
        for name, loss in accum.items():
            totals[name] += loss
    elapsed = time.perf_counter() - start
    updates = unified.arbitrator.updates - initial_updates
    if updates != transitions or unified.episodic.serialized_size() > unified.MEMORY_BUDGET:
        raise ValueError("prequential or episodic budget invariant failed")
    return dict(raw_bytes=raw_bytes, transitions=transitions, seconds=elapsed,
                cpu_seconds=time.process_time() - cpu,
                ms_per_byte=1000 * elapsed / raw_bytes, arbitration_updates=updates,
                wake_calls=unified.reader.deep_calls - initial_wakes,
                bpb={k: v / transitions for k, v in totals.items()}, blocks=block_rows)


def score_composition(unified, seed):
    singles, _, novel, shifted, phases = fixture(seed)
    for phase in phases:
        for prefix, _, _, observed in phase:
            unified.observe_relation(prefix, observed)

    def score(rows):
        predictions, labels = [], []
        loss = 0.0
        for prefix, _, target in rows:
            probabilities = unified.predict_relation(prefix)
            predictions.append(int(probabilities.argmax()))
            labels.append(target)
            loss -= math.log2(max(float(probabilities[target]), 1e-300))
        return dict(accuracy=sum(a == b for a, b in zip(predictions, labels, strict=True)) / len(rows),
                    predictions=predictions, labels=labels, bpb=loss / len(rows))

    return dict(
        unseen=score(novel), primitives=score(singles), shifted_primitives=score(shifted),
        fixture_digest=hashlib.sha256(repr((phases, novel)).encode()).hexdigest(),
        observed_records=sum(map(len, phases)),
        corrupted=sum(y != observed for phase in phases for _, _, y, observed in phase),
    )


def restored_outputs(unified, seed):
    novel = fixture(seed)[2]
    return dict(model=state_hash(unified.backbone),
                composition=probability_digest(unified.predict_relation(p) for p, _, _ in novel),
                continuation=probability_digest(unified.step(byte).mixed for byte in CONTINUATION))


def run_seed(directory, checkpoint_dir, ident, gate12, seed):
    arm = DELTA if gate12["status"] == "PASS" else ATTENTION
    upstream = run_spec(checkpoint_dir, gate12["identity"], seed, arm)
    record = next(r for r in gate12["runs"] if r["seed"] == seed)["capacity"][arm]
    if upstream.read() != record:
        raise ValueError("local checkpoint custody disagrees with published Gate 12")
    supervised = model_for(seed, arm)
    supervised.load_state_dict(torch.load(upstream.checkpoint, weights_only=True))
    if state_hash(supervised) != record["final_hash"]:
        raise ValueError("restored reasoning checkpoint digest mismatch")

    prepared = json.loads((directory / "prepared.json").read_text())
    if (prepared["identity"] != ident or prepared["counts_sha256"] != digest(directory / "counts.bin")
            or prepared["memory_sha256"] != digest(directory / "memory")):
        raise ValueError("fitted real corpus inputs changed")
    counts = CompactBigram.restore((directory / "counts.bin").read_bytes())
    memory = ConsolidatingMemory.restore((directory / "memory").read_bytes(), max_bytes=40000)
    unified = NexusExpanseUnified(supervised.encoder, counts, memory=memory)
    blocks = read_split()["dev"]
    weights = state_hash(unified.backbone)
    torch.manual_seed(seed)
    sample = bytes(torch.randint(256, (32,)).tolist())
    # Use a throwaway clone for parity so these diagnostics cannot write corpus state.
    initial = unified.snapshot()
    parity = exactness(NexusExpanseUnified.restore(initial), NexusExpanseUnified.restore(initial), sample)
    corpus = evaluate_corpus(unified, blocks)
    comp = score_composition(unified, seed)
    historical = next(r for r in json.loads(GATE8.read_text())["runs"] if r["seed"] == seed)
    if comp["fixture_digest"] != historical["candidate"]["fixture_digest"]:
        raise ValueError("Gate 8 composition fixture assignment changed")
    cap = capacity_eval(supervised, manifest()["splits"]["eval"])
    checkpoint = directory / f"{seed}-unified.pt"
    snapshot = unified.snapshot()
    with checkpoint.open("xb") as output:
        output.write(snapshot)
    expected = restored_outputs(unified, seed)
    cold = subrun("--cold", checkpoint, "--seed", seed)
    if state_hash(unified.backbone) != weights:
        raise ValueError("verification altered the frozen encoder")
    exact = all(v == 0.0 for v in parity.values())
    matching_tasks = cap == record["heldout"]
    historical_unseen = historical["candidate"]["arms"]["latent_wta"]["scores"]["unseen"]["accuracy"]
    result = dict(
        identity=ident, seed=seed, backbone=arm, source_model_sha256=record["checkpoint_sha256"],
        parity=parity, exact=exact, corpus=corpus, composition=comp, tasks=cap,
        scores_preserved_from_gate12=matching_tasks, gate8_historical_unseen_accuracy=historical_unseen,
        memory=dict(whole_system_serialized_bytes=len(snapshot), checkpoint_sha256=digest(checkpoint),
                    episodic_bytes=unified.episodic.serialized_size(), compact_bytes=len(counts.snapshot()),
                    hopfield_codebook_bytes=unified.reasoning.live_codebook_bytes,
                    hopfield_slot_bytes=unified.reasoning.live_slot_bytes,
                    stream_tensor_bytes=unified.stream.tensor_bytes,
                    fresh_process_peak_rss_bytes=peak_rss()),
        cold=cold, expected_digests=expected, cold_equal=cold["digests"] == expected,
    )
    atomic(directory / f"{seed}-unified.json", result)
    print(json.dumps(dict(seed=seed, exact=exact, cold_equal=result["cold_equal"],
                          bpb=corpus["bpb"], composition=comp["unseen"]["accuracy"])))


def merge(directory, ident, dest):
    runs = [json.loads((directory / f"{s}-unified.json").read_text()) for s in SEEDS]
    for seed, r in zip(SEEDS, runs, strict=True):
        if (r["identity"] != ident or r["seed"] != seed
                or digest(directory / f"{seed}-unified.pt") != r["memory"]["checkpoint_sha256"]):
            raise ValueError("invalid unified receipt")
    engineering = all(r["exact"] and r["cold_equal"] and r["scores_preserved_from_gate12"] for r in runs)
    status = "PASS" if engineering else "FAIL"
    atomic(dest, dict(schema="nexus-gate14-unified-v1-dev-v1", qualification="REUSED_DEV_FIXTURES",
                      identity=ident, version="1.0-dev", engineering=status,
                      scientific="MEASURED_DEV", backbone=runs[0]["backbone"], runs=runs))
    print(json.dumps(dict(engineering=status, backbone=runs[0]["backbone"])))


def main():
    setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path)
    p.add_argument("--checkpoint-directory", type=Path)
    p.add_argument("--seed", type=int, choices=SEEDS)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--merge", type=Path)
    action.add_argument("--cold", type=Path)
    args = p.parse_args()
    if args.cold:
        start = time.monotonic()
        restored = NexusExpanseUnified.restore(args.cold.read_bytes())
        loaded = time.monotonic() - start
        checks = restored_outputs(restored, args.seed)
        print(json.dumps(dict(digests=checks, restore_seconds=loaded,
                              verification_seconds=time.monotonic() - start - loaded,
                              fresh_process_peak_rss_bytes=peak_rss())))
        return
    if args.directory is None:
        p.error("--directory required")
    directory = prepare_directory(args.directory, ROOT)
    ident, gate12 = experiment_identity()
    if args.prepare:
        with writer(directory, "prepare"):
            if any((directory / name).exists() for name in ("prepared.json", "counts.bin", "memory")):
                raise ValueError("real corpus preparation output already exists")
            prepared = prepare(directory, read_split()["train"], compact=True)
            atomic(directory / "prepared.json", dict(identity=ident, **prepared,
                   counts_sha256=digest(directory / "counts.bin"), memory_sha256=digest(directory / "memory")))
            print(json.dumps(prepared))
    elif args.merge:
        merge(directory, ident, args.merge)
    elif args.seed is not None and args.checkpoint_directory is not None:
        with writer(directory, f"{args.seed}-unified"):
            if (directory / f"{args.seed}-unified.pt").exists():
                raise ValueError("unified checkpoint already exists")
            run_seed(directory, args.checkpoint_directory.resolve(), ident, gate12, args.seed)
    else:
        p.error("--seed and --checkpoint-directory required")


if __name__ == "__main__":
    main()
