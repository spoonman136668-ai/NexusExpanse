"""Gate 13 matched objectives on the selected raw-byte reasoning backbones."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from benchmarks.gate10_jepa_corpus import DOMAINS
from benchmarks.gate10_jepa_dev import ce_loss, hist, probe, retrieval
from benchmarks.gate11_selective_kuramoto_dev import expanded_train, identity, manifest, setup
from benchmarks.gate12_deltanet_dev import ARMS as BACKBONES, ATTENTION, DELTA, SEEDS, attention
from benchmarks.gate9_jepa_dev import atomic, digest, fit, noncollapse, pack, predict, raw
from benchmarks.gate_custody import EpochRun, peak_rss, prepare_directory, update
from nexus_expanse.deltanet import DeltaNetMonarchByteCore
from nexus_expanse.jepa import state_hash
from nexus_expanse.jepa_backbone import BackboneJEPA, last_hidden

UNTRAINED = "untrained"
CE = "next_byte_ce"
JEPA = "infonce_jepa"
HYBRID = "hybrid_infonce_ce"
ARMS = (UNTRAINED, CE, JEPA, HYBRID)
EPOCHS = 10
STEPS = 60
HISTOGRAM_FLOOR = .20
GATE12 = ROOT / "evidence/gate12-deltanet-dev-001.json"


def corpus_rows():
    data = manifest()["splits"]
    fit_rows, ev = data["train"], data["eval"]
    train = expanded_train(fit_rows, ev) + [r for r in fit_rows if r["domain"] == "code"]
    return train, fit_rows, ev


def experiment_identity(train):
    evidence = json.loads(GATE12.read_text())
    if evidence["schema"] != "nexus-gate12-deltanet-dev-v1" or not evidence["parity"]["passed"]:
        raise ValueError("missing qualified Gate 12 capacity evidence")
    backbones = (ATTENTION, DELTA) if evidence["delta_eligible_for_gate13"] else (ATTENTION,)
    hashed = hashlib.sha256(json.dumps(train, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return (dict(identity(), expanded_rows_sha256=hashed, gate12_sha256=digest(GATE12)), backbones)


def batches(train, seed, epoch):
    rng = np.random.default_rng(seed + epoch)
    width = (20, 2, 20)
    ids = [rng.permutation([i for i, r in enumerate(train) if r["domain"] == d]) for d in DOMAINS]
    if [len(i) for i in ids] != [STEPS * n for n in width]:
        raise ValueError("expanded corpus cardinality changed")
    for step in range(STEPS):
        yield [train[i] for indices, n in zip(ids, width, strict=True)
               for i in indices[step * n:(step + 1) * n]]


def model_for(seed, backbone, arm):
    torch.manual_seed(seed)
    encoder = attention() if backbone == ATTENTION else DeltaNetMonarchByteCore()
    return BackboneJEPA(encoder) if arm in (JEPA, HYBRID) else encoder


def encoder_of(model):
    return model.context if isinstance(model, BackboneJEPA) else model


def run_spec(directory, ident, seed, backbone, arm):
    return EpochRun(directory, f"{seed}-{backbone}-{arm}", ident, 0 if arm == UNTRAINED else EPOCHS, STEPS)


def features(encoder, rows, key):
    return np.concatenate([
        last_hidden(encoder, *pack(raw(rows[start:start + 12], key))).numpy()
        for start in range(0, len(rows), 12)
    ])


@torch.inference_mode()
def evaluate(model, train, ev):
    encoder = encoder_of(model)
    start = time.perf_counter()
    queried = features(encoder, ev, "ctx")
    elapsed = time.perf_counter() - start
    train_context, train_future = (features(encoder, train, k) for k in ("ctx", "fut"))
    targets = features(encoder, ev, "fut")
    domains = np.asarray([r["domain"] for r in ev])
    result = dict(
        retrieval=retrieval(predict(queried, fit(train_context, train_future)), targets, ev),
        direct_retrieval=retrieval(queried, targets, ev),
        math=probe(train_context, queried, train, ev, "math", 4),
        science=probe(train_context, queried, train, ev, "science", 2),
        noncollapse={domain: noncollapse(queried[domains == domain]) for domain in DOMAINS},
        inference_ms_span=elapsed * 1000 / len(ev),
        evaluation_seconds=time.perf_counter() - start,
    )
    if isinstance(model, BackboneJEPA):
        result["native_retrieval"] = retrieval(
            model.predictor(torch.from_numpy(queried)).numpy(), features(model.target, ev, "fut"), ev,
        )
    return result


def histogram_control(train, ev):
    train_context, train_future = (hist(train, k) for k in ("ctx", "fut"))
    queried, targets = (hist(ev, k) for k in ("ctx", "fut"))
    result = dict(retrieval=retrieval(predict(queried, fit(train_context, train_future)), targets, ev),
                  math=probe(train_context, queried, train, ev, "math", 4),
                  science=probe(train_context, queried, train, ev, "science", 2))
    if result["retrieval"]["top1"] != HISTOGRAM_FLOOR:
        raise ValueError("frozen 20% histogram result did not reproduce")
    return result


def run_arm(directory, ident, seed, backbone, arm, epoch_limit, train, fit_rows, ev):
    model = model_for(seed, backbone, arm)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=.003, weight_decay=.01)

    def epoch(number):
        losses = []
        for step, rows in enumerate(batches(train, seed, number)):
            xs, ys = raw(rows, "ctx"), raw(rows, "fut")
            optimizer.zero_grad(set_to_none=True)
            if arm == CE:
                loss = ce_loss(model, xs, ys)
            else:
                domains = torch.tensor([DOMAINS.index(r["domain"]) for r in rows])
                loss = model.loss(*pack(xs), *pack(ys), domains)
                if arm == HYBRID:
                    loss = loss + ce_loss(model.context, xs, ys)
            losses.append(update(model, optimizer, loss))
            if arm in (JEPA, HYBRID):
                model.update_target((number - 1) * STEPS + step, EPOCHS * STEPS)
        return losses

    def cold(path):
        return json.loads(subprocess.check_output([
            sys.executable, __file__, "--cold", str(path), "--seed", str(seed),
            "--arm", arm, "--backbone", backbone,
        ], text=True))

    return run_spec(directory, ident, seed, backbone, arm).advance(
        model, optimizer, epoch_limit, epoch, cold, lambda: evaluate(model, fit_rows, ev),
        state_hash(encoder_of(model)),
    )


def comparisons(runs, arm):
    margins = {}
    cells = {}
    for r in runs:
        candidate, untrained, control = (r["arms"][a] for a in (arm, UNTRAINED, CE))

        def scores(run):
            return dict(retrieval=run["retrieval"]["top1"], math=run["math"]["accuracy"],
                        science=run["science"]["accuracy"])

        current, floor, ce = map(scores, (candidate, untrained, control))
        seed = str(r["seed"])
        cells[seed] = dict(
            noncollapse={d: v["effective_rank"] >= 8.0 and v["mean_std"] >= .50
                         for d, v in candidate["noncollapse"].items()},
            beats_untrained={key: value > floor[key] for key, value in current.items()},
            beats_histogram=current["retrieval"] > HISTOGRAM_FLOOR,
        )
        margins[seed] = {key: value - ce[key] for key, value in current.items()}
    floors = all(all(c["noncollapse"].values()) and all(c["beats_untrained"].values())
                 and c["beats_histogram"] for c in cells.values())
    clears_margin = {key: all(m[key] >= .05 - 1e-12 for m in margins.values())
                     for key in ("retrieval", "math", "science")}
    strong = clears_margin["retrieval"] or (clears_margin["math"] and clears_margin["science"])
    status = ("PASS" if strong else "MIXED") if floors else "FAIL"
    return dict(status=status, checks=cells, margins_over_ce=margins, clears_margin=clears_margin)


def merge(directory, ident, backbones, fit_rows, ev, dest):
    control = histogram_control(fit_rows, ev)
    results = {}
    for backbone in backbones:
        runs = []
        for seed in SEEDS:
            arms = {arm: run_spec(directory, ident, seed, backbone, arm).read() for arm in ARMS}
            if len({r["initial_hash"] for r in arms.values()}) != 1:
                raise ValueError("encoder initializations differ between matched arms")
            runs.append(dict(seed=seed, arms=arms))
        results[backbone] = dict(runs=runs, decisions={arm: comparisons(runs, arm) for arm in (JEPA, HYBRID)})
    statuses = {b: {a: t["status"] for a, t in r["decisions"].items()} for b, r in results.items()}
    status = max((s for b in statuses.values() for s in b.values()), key=("FAIL", "MIXED", "PASS").index)
    atomic(dest, dict(schema="nexus-gate13-jepa-proven-backbone-dev-v1",
                      qualification="REUSED_AUTHORED_RELATIONAL_DEV", identity=ident,
                      histogram_control=control, results=results, statuses=statuses, status=status))
    print(json.dumps(dict(status=status, statuses=statuses)))


def main():
    setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path)
    p.add_argument("--epoch-limit", type=int, default=4, choices=range(1, EPOCHS + 1))
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--backbone", choices=BACKBONES)
    p.add_argument("--arm", choices=ARMS)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--merge", type=Path)
    action.add_argument("--cold", type=Path)
    args = p.parse_args()
    if args.cold:
        start = time.monotonic()
        model = model_for(args.seed, args.backbone, args.arm)
        model.load_state_dict(torch.load(args.cold, weights_only=True))
        print(json.dumps([state_hash(model), time.monotonic() - start, peak_rss()]))
        return
    if args.directory is None:
        p.error("--directory required")
    rows, fit_rows, ev = corpus_rows()
    ident, backbones = experiment_identity(rows)
    directory = prepare_directory(args.directory, ROOT)
    if args.merge:
        merge(directory, ident, backbones, fit_rows, ev, args.merge)
    elif args.seed is not None and args.arm is not None and args.backbone in backbones:
        run_arm(directory, ident, args.seed, args.backbone, args.arm, args.epoch_limit,
                rows, fit_rows, ev)
    else:
        p.error("--seed, --arm and a Gate 12 eligible --backbone are required for training")


if __name__ == "__main__":
    main()
