"""Offline raw-byte routing, native snapshot custody, and telemetry birth."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.dev_artifacts import atomic, atomic_bytes, digest, save_pt
from benchmarks.gate_custody import peak_rss
from benchmarks.phase13_warmstart_train import cli
from benchmarks.phase3_multi_skill_routing import SOURCE, git
from benchmarks.phase9_weight_edit_and_assimilate import frozen_data
from nexus_expanse.cli import route_prefix_check
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.jepa import state_hash
from nexus_expanse.multi_skill_router import MultiSkillBank, load_bank, save_bank
from nexus_expanse.multimodal_bytes import generate_stream
from nexus_expanse.skill_birth import EPISODIC_BIRTH, OpenWorldSkillBank
from nexus_expanse.unified import NexusExpanseUnified
from scripts.prepare_gpu_corpus import CorpusOverlapError, assert_disjoint

BASE = "90ba28f1eb4268a66b535b24ef5736b2ac79441d"
PREREG = "docs/PHASE18_MULTIMODAL_BYTE_SKILLS_PREREG.md"
SEEDS = tuple(range(3201, 3206))
KNOWN = ("prose", "code", "audio", "image")
DOMAINS = KNOWN + ("telemetry",)
SWITCHES = ("prose", "audio", "image", "code", "audio", "prose")
CONTEXT, STEPS = 64, 64
FIT_BYTES, CHECK_BYTES, CAL_BYTES = 6081, 1536, 384
SEGMENT, SETTLE = 128, 16
WINDOW, PROBE_STRIDE, BIRTH_STEPS = 64, 32, 64


def tensor(raw):
    return torch.tensor([list(raw)], dtype=torch.long)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def text_data(seed):
    rejected = []
    for attempt in range(8):
        corpus_seed = seed + 10_000 * attempt
        rows, provenance = frozen_data(corpus_seed)
        data = {}
        for split, size in (("fit", FIT_BYTES), ("val", CAL_BYTES), ("check", CHECK_BYTES)):
            data[split] = {name: rows[split][domain].to(torch.uint8).numpy().tobytes()[:size]
                           for name, domain in (("prose", "markdown_prose"), ("code", "base_python"))}
        try:
            assert_disjoint(data["fit"].values(), data["val"])
            assert_disjoint(data["fit"].values(), data["check"])
            assert_disjoint(data["val"].values(), data["check"])
        except CorpusOverlapError as error:
            rejected.append(dict(seed=corpus_seed, reason=str(error)))
        else:
            return data, dict(seed=corpus_seed, rejected=rejected, windows=provenance)
    raise ValueError("no leakage-free text fixture within the frozen data candidates")


def corpus(root, seed):
    data, provenance = text_data(seed)
    fit, cal, check = (data[key] for key in ("fit", "val", "check"))
    for domain in ("audio", "image"):
        fit[domain] = generate_stream(domain, f"{seed}:fit", 96)[:FIT_BYTES]
        cal[domain] = generate_stream(domain, f"{seed}:val", 6, exclude=(fit[domain],))[:CAL_BYTES]
        check[domain] = generate_stream(domain, f"{seed}:check", 24, exclude=(fit[domain], cal[domain]))[:CHECK_BYTES]
    encounter = generate_stream("telemetry", f"{seed}:encounter", 80)
    check["telemetry"] = generate_stream("telemetry", f"{seed}:check", 48, exclude=(encounter,))
    for name in KNOWN:
        assert_disjoint((fit[name],), {"val": cal[name], "check": check[name]})
        assert_disjoint((cal[name],), {"check": check[name]})
    assert_disjoint((encounter,), {"telemetry": check["telemetry"]})
    data["encounter"] = {"telemetry": encounter}
    custody = {split: {d: dict(bytes=len(raw), sha256=sha(raw)) for d, raw in rows.items()}
               for split, rows in data.items()}
    files = {}
    for name in KNOWN + ("monolith",):
        train = root / f"{name}-train.bin"
        evaluation = root / f"{name}-eval.bin"
        if name == "monolith":
            joined = b"".join(fit[d] for d in KNOWN)
            selected = joined[:((len(joined) - 1) // CONTEXT) * CONTEXT + 1]
            held = b"".join(check[d] for d in KNOWN)
            assert_disjoint((selected,), {name: held})
        else:
            selected, held = fit[name], check[name]
        atomic_bytes(train, selected)
        atomic_bytes(evaluation, held)
        files[name] = train, evaluation
    return files, cal, check, encounter, dict(text=provenance, splits=custody, train_eval_overlap_32b=0)


def train_skill(root, name, seed, files):
    train, evaluation = files[name]
    run, snapshot = root / name, root / f"{name}.pt"
    steps = STEPS * (len(KNOWN) if name == "monolith" else 1)
    report = cli("train", "--directory", run, "--train-file", train, "--eval-file", evaluation,
                 "--device", "cpu", "--dtype", "fp32", "--skill-backbone", "--use-sdpa", "--seed", seed,
                 "--dim", 128, "--layers", 4, "--context", CONTEXT, "--epochs", 1, "--steps-per-epoch", steps,
                 "--batch-per-domain", 1, "--eval-blocks", 4, "--save-snapshot", snapshot, "--snapshot-quant", "int8")
    raw = snapshot.read_bytes()
    restored = NexusExpanseUnified.restore(raw)
    fp32 = len(restored.snapshot())
    receipt = json.loads((run / "gate17.json").read_bytes())
    return snapshot, dict(export=report, int8_bytes=len(raw), fp32_bytes=fp32, compression=fp32 / len(raw),
                          target_bytes=receipt["epochs"][0]["telemetry"]["target_bytes"], updates=receipt["steps"],
                          cold_equal=receipt["cold_equal"], archived_state_sha256=state_hash(restored.backbone),
                          donor_forward_calls=receipt["donor_forward_calls"] +
                          receipt["fresh_evaluation"]["donor_forward_calls"] + report["donor_forward_calls"])


def bpb(raw, logp):
    return -float(logp.gather(-1, raw.unsqueeze(-1)).double().mean()) / math.log(2)


@torch.inference_mode()
def switched(bank, monolith, held):
    chunks = []
    occurrences = dict.fromkeys(KNOWN, 0)
    for name in SWITCHES:
        start = occurrences[name] * SEGMENT
        chunks.append(held[name][start:start + SEGMENT])
        occurrences[name] += 1
    raw = tensor(b"".join(chunks))
    components = bank.component_log_probs(raw)
    routed, weights = bank.forward_hmm_bma(raw)
    indices = torch.tensor([[KNOWN.index(d) for d in SWITCHES]]).repeat_interleave(SEGMENT, dim=1)
    oracle = components.gather(-2, indices[..., None, None].expand(-1, -1, 1, 256)).squeeze(-2)
    single = {name: bpb(raw, components[:, :, i]) for i, name in enumerate(KNOWN)}
    single["monolith"] = bpb(raw, monolith.component_log_probs(raw).squeeze(2))
    active = weights.argmax(-1)
    settled = torch.arange(raw.shape[1]) % SEGMENT >= SETTLE
    return dict(bytes=raw.numel(), bpb=dict(hmm=bpb(raw, routed), uniform=bpb(raw, components.logsumexp(-2) - math.log(len(KNOWN))),
                                           oracle=bpb(raw, oracle), best_single=min(single.values()), single=single),
                routing_accuracy=float((active == indices).double().mean()),
                settled_routing_accuracy=float((active[:, settled] == indices[:, settled]).double().mean()),
                settled_by_domain={d: float((active == indices)[(indices == i) & settled].double().mean())
                                   for i, d in enumerate(KNOWN)})


@torch.inference_mode()
def domain_metrics(bank, raw, domains):
    logp, routing = bank.forward_hmm_bma(raw)
    active = routing.argmax(-1)
    scores = {d: dict(bpb=bpb(raw[i:i + 1], logp[i:i + 1]),
                      accuracy=float((active[i, SETTLE:] == bank.skill_names.index(d)).double().mean())
                      if d in bank.skill_names else 0.0) for i, d in enumerate(domains)}
    output = sha(logp.numpy().tobytes() + routing.numpy().tobytes())
    return dict(scores=scores, output_sha256=output, states={name: state_hash(skill) for name, skill in bank.skills.items()})


def cold_receipt(root):
    raw = torch.load(root / "probe.pt", weights_only=True, map_location="cpu")
    with forbid_donor_forward() as guard:
        before = domain_metrics(load_bank(root / "before.pt"), raw, DOMAINS)
        after = domain_metrics(load_bank(root / "after.pt"), raw, DOMAINS)
    return dict(before=before, after=after, worker_pid=os.getpid(), donor_forward_calls=guard.donor_forward_calls)


def birth_check(root, bank, cal, check, encounter):
    raw = torch.cat([tensor(check[d][:SEGMENT]) for d in DOMAINS])
    save_pt(root / "probe.pt", raw)
    save_bank(bank, root / "before.pt")
    before = domain_metrics(bank, raw, DOMAINS)
    components = bank.component_log_probs(raw[:-1])
    world = OpenWorldSkillBank(bank)
    tail = world.calibrate_tail_surprise({d: tensor(cal[d][:160]) for d in KNOWN}, window=WINDOW,
                                         quantile=0.90, margin_bpb=0.5)
    false_flags = world.detect_novelty(raw[:-1], tail=tail).tolist()
    prefix = cal["prose"][:128]
    mixed = prefix + encounter + cal["audio"][:128]
    candidate = None
    observed = None
    probes = []
    birth_at = None
    for end in range(96, len(mixed) + 1, PROBE_STRIDE):
        observed = mixed[max(0, end - 192):end]
        candidate = world.observe("telemetry", tensor(observed), tail=tail, mode=EPISODIC_BIRTH, steps=BIRTH_STEPS)
        probes.append(dict(end=end, triggered=candidate is not None))
        if candidate is not None:
            birth_at = end
            break
    after = domain_metrics(bank, raw, DOMAINS)
    resident_exact = all(after["states"][name] == before["states"][name] for name in KNOWN)
    resident_exact = resident_exact and torch.equal(components, bank.component_log_probs(raw[:-1])[:, :, :len(KNOWN)])
    world.save_bank(root / "after.pt")
    cold = json.loads(subprocess.check_output([sys.executable, __file__, "--cold", str(root)], cwd=ROOT))
    exact = cold["worker_pid"] != os.getpid() and cold["before"] == before and cold["after"] == after
    delta = {name: after["scores"][name]["bpb"] - before["scores"][name]["bpb"] for name in DOMAINS}
    return dict(birth_at_byte=birth_at, telemetry_start=len(prefix), telemetry_end=len(prefix) + len(encounter),
                observed_bytes=len(observed) if candidate is not None else 0,
                observed_sha256=sha(observed) if candidate is not None else None,
                birth_steps=BIRTH_STEPS if candidate is not None else 0,
                born_quant=candidate.neural.snapshot_quant if candidate is not None else None,
                tail_limits_bpb=list(tail.limits_bpb), false_novelty=dict(zip(KNOWN, false_flags, strict=True)),
                probes=probes, cold=cold, exact_cold=exact, resident_exact=resident_exact, bpb_drift=delta,
                max_resident_drift=max(delta[d] for d in KNOWN), before_bank_bytes=(root / "before.pt").stat().st_size,
                after_bank_bytes=(root / "after.pt").stat().st_size)


def seed_run(root, seed):
    files, cal, held, encounter, custody = corpus(root, seed)
    paths, artifacts = {}, {}
    with forbid_donor_forward() as guard:
        for name in files:
            paths[name], artifacts[name] = train_skill(root, name, seed, files)
        bank = MultiSkillBank.from_unified_snapshots({name: paths[name] for name in KNOWN})
        monolith = MultiSkillBank.from_unified_snapshots({"monolith": paths["monolith"]})
        def route(raw, **kwargs):
            return bank.forward_hmm_bma(raw, **kwargs)

        prefix = {}
        for name, data in held.items():
            raw = tensor(data[:64])
            prefix[name] = route_prefix_check(route, raw, route(raw, include_next=True))
        switch = switched(bank, monolith, held)
        birth = birth_check(root, bank, cal, held, encounter)
    calls = guard.donor_forward_calls + birth["cold"]["donor_forward_calls"] + sum(
        item["donor_forward_calls"] for item in artifacts.values())
    exact_snapshot = all(item["cold_equal"] and item["compression"] >= 3.0 and item["export"]["exact_state"]
                         and item["export"]["snapshot_quant"] == "int8" and len(item["export"]["parity"]) == 4
                         and not any(item["export"]["parity"].values())
                         and item["export"]["round_trip_probabilities"] == 0.0 for item in artifacts.values())
    matched = (all(artifacts[n]["updates"] == STEPS and artifacts[n]["target_bytes"] == CONTEXT * STEPS for n in KNOWN)
               and artifacts["monolith"]["updates"] == STEPS * len(KNOWN)
               and artifacts["monolith"]["target_bytes"] == CONTEXT * STEPS * len(KNOWN))
    return dict(seed=seed, corpus=custody, artifacts=artifacts, prefix=prefix, switch=switch, birth=birth,
                exact=exact_snapshot and matched and birth["exact_cold"] and birth["resident_exact"] and not any(prefix.values()),
                donor_forward_calls=calls)


def verdict(rows):
    def routing(row):
        scores = row["switch"]["bpb"]
        return scores["hmm"] < min(scores["uniform"], scores["best_single"]) and row["switch"]["settled_routing_accuracy"] >= 0.80

    def birth(row):
        item = row["birth"]
        at = item["birth_at_byte"]
        return (at is not None and item["telemetry_start"] < at <= item["telemetry_end"]
                and not any(item["false_novelty"].values()) and item["bpb_drift"]["telemetry"] < 0
                and item["max_resident_drift"] <= 0.05 and item["born_quant"] == "int8"
                and min(item["cold"]["after"]["scores"][d]["accuracy"] for d in DOMAINS) >= 0.80)

    routes = sum(routing(row) for row in rows)
    births = sum(birth(row) for row in rows)
    exact = sum(row["exact"] for row in rows)
    calls = sum(row["donor_forward_calls"] for row in rows)
    engineering = exact == len(SEEDS) and not calls
    science = ("INVALID" if not engineering else "PASS" if routes == births == len(SEEDS) else
               "MIXED" if max(routes, births) == len(SEEDS) else "FAIL")
    return dict(scored_engineering="PASS" if engineering else "FAIL", scientific=science, exact_seeds=exact,
                routing_pass_seeds=routes, birth_pass_seeds=births, donor_forward_calls=calls,
                telemetry_births=sum(r["birth"]["birth_at_byte"] is not None and
                                     r["birth"]["telemetry_start"] < r["birth"]["birth_at_byte"] <= r["birth"]["telemetry_end"]
                                     for r in rows),
                minimum_compression=min(a["compression"] for r in rows for a in r["artifacts"].values()),
                mean_settled_routing_accuracy=statistics.mean(r["switch"]["settled_routing_accuracy"] for r in rows),
                maximum_resident_drift=max(r["birth"]["max_resident_drift"] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out", type=Path)
    action.add_argument("--cold", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.cold:
        print(json.dumps(cold_receipt(args.cold), allow_nan=False))
        return
    out = args.out.resolve()
    intent = out.with_suffix(".intent.json")
    head = git("rev-parse", "HEAD").decode().strip()
    tree = git("rev-parse", "HEAD^{tree}").decode().strip()
    if (out.is_relative_to(ROOT) or out.exists() or intent.exists() or git("status", "--porcelain")
            or git("rev-parse", "HEAD^").decode().strip() != BASE or not git("ls-tree", head, PREREG)):
        parser.error("run once from the clean Phase 18 preregistration commit with a new external output")
    identity = dict(base=BASE, source_commit=head, source_tree=tree, prereg_sha256=digest(ROOT / PREREG))
    atomic(intent, identity)
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix="nexus-phase18-") as scratch:
        for seed in SEEDS:
            root = Path(scratch) / str(seed)
            root.mkdir()
            rows.append(seed_run(root, seed))
            print(f"Phase 18 seed {seed} complete", flush=True)
    if git("rev-parse", "HEAD").decode().strip() != head or git("status", "--porcelain"):
        parser.error("the committed source changed during scoring")
    result = dict(schema="nexus.phase18-multimodal-byte-skills.v1", **identity, corpus_source_sha=SOURCE,
                  platform=platform.platform(), torch=str(torch.__version__), device="cpu", threads=1,
                  external_downloads=0, wall_seconds=time.perf_counter() - started,
                  parent_peak_rss_bytes=peak_rss(), seeds=rows, summary=verdict(rows))
    atomic(out, result)
    print(json.dumps(result["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
