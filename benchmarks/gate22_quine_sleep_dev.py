"""Four-arm physically byte-matched Gate 20 cold-retention experiment."""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.dev_artifacts import atomic, digest
from benchmarks.gate11_selective_kuramoto_dev import SEEDS, identity, setup
from benchmarks.gate20_sleep_consolidation_dev import (
    BUDGET, MANIFEST, NO_SLEEP, RANDOM, advance, fixtures, receipts,
)
from benchmarks.gate_custody import prepare_directory
from nexus_expanse.sleep_consolidation import _apply, _items, _reachable, adaptive_bytes

FREQUENCY = "gate20_sleep_control"
QUINE = "quine_sleep_unified"
ARMS = (NO_SLEEP, RANDOM, FREQUENCY, QUINE)
GATE20 = ROOT / "evidence/gate20-sleep-consolidation-dev-001.json"


def match_states(models, seed):
    initial = {arm: adaptive_bytes(model) for arm, model in models.items()}
    plans = {}
    for arm, model in models.items():
        plan = _items(model, random.Random(seed) if arm == RANDOM else None)
        if arm == NO_SLEEP:
            # Apply the existing sparse-support capacity order. Keep the
            # decoupled relational store intact in this budget-only control.
            order = {key: n for n, key in enumerate(model.episodic._table)}
            plan = sorted(((size, key) for size, key in plan if isinstance(key, bytes)),
                          key=lambda row: (model.episodic._totals[row[1]], order[row[1]]))
        if arm == QUINE:
            plan = [row for row in plan if row[1] not in model.episodic.quine_contexts]
        plans[arm] = plan, _reachable(plan)
    target = next((n for n in range(min(initial.values()), 0, -1)
                   if all((levels[-1] >> (initial[arm] - n)) & 1
                          for arm, (_, levels) in plans.items())), None)
    if target is None:
        raise ValueError("no jointly reachable physical byte count across all policies")
    removed = {arm: _apply(models[arm], items, levels, initial[arm] - target)
               for arm, (items, levels) in plans.items()}
    actual = {arm: adaptive_bytes(model) for arm, model in models.items()}
    if set(actual.values()) != {target} or target > BUDGET:
        raise ValueError("physical matching or budget invariant failed")
    return dict(target_bytes=target, before=initial, after=actual, removed=removed)


def boundary(models, ledgers, seed, i, domain):
    control = models[FREQUENCY].consolidate_sleep(ledgers[FREQUENCY])
    quines = models[QUINE].episodic.consolidate_quines(domain, ledgers[QUINE])
    matched = match_states(models, seed * 10 + i)
    return dict(gate20_sleep=control, quines=quines, match=matched)


def decision(runs):
    checks = []
    for r in runs:
        history = r["phases"]
        cold = history[-1]
        scores = cold["arms"]
        margins = {a: scores[a]["bpb"] - scores[QUINE]["bpb"] for a in ARMS[:-1]}
        match = all(len({p["arms"][a]["post_boundary_adaptive_bytes"] for a in ARMS}) == 1
                    for p in history[:-1])
        budget = all(p["arms"][a]["peak_adaptive_bytes"] <= BUDGET
                     and p["arms"][a]["post_boundary_adaptive_bytes"] <= BUDGET
                     for p in history[:-1] for a in ARMS)
        budget = budget and all(d["peak_adaptive_bytes"] <= BUDGET
                                for arm in scores.values() for d in arm["domains"].values())
        restore = set(cold["restores"]) == set(ARMS) and all(
            c["equal"] and c["fresh_process"] for c in cold["restores"].values())
        support = all(len(p["arms"][QUINE]["retention"]) == min(i + 1, 3)
                      and all(w["support"] > 0 for domain in p["arms"][QUINE]["retention"].values()
                              for w in domain)
                      for i, p in enumerate(history))
        checks.append(dict(seed=r["seed"], margins_over_controls=margins, exact_byte_match=match,
                           budget=budget, exact_cold_restore=restore, retained_witnesses=support))
    engineering = all(c["exact_byte_match"] and c["budget"] and c["exact_cold_restore"]
                      and c["retained_witnesses"] for c in checks)
    margins = [c["margins_over_controls"] for c in checks]
    strict = all(m[NO_SLEEP] >= .05 and m[RANDOM] >= .05 and m[FREQUENCY] > 0. for m in margins)
    status = "FAIL" if not engineering else "PASS" if strict else "MIXED" if any(
        v > 0. for m in margins for v in m.values()) else "FAIL"
    return dict(status=status, engineering=engineering, checks=checks)


def merge(directory, ident, dest):
    runs = []
    for seed in SEEDS:
        _, fixture = fixtures(seed)
        chain = receipts(directory, ident, seed, fixture)
        result = json.loads((directory / f"{seed}-sleep.json").read_text())
        if len(chain) != 4 or result != dict(seed=seed, phases=chain):
            raise ValueError("incomplete cold retention evidence")
        runs.append(result)
    report = decision(runs)
    atomic(dest, dict(schema="nexus-gate22-quine-sleep-dev-v1",
                      qualification="REUSED_SEQUENTIAL_CORPORA_DEV", identity=ident, runs=runs, **report))
    print(json.dumps(report))


def main():
    setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True, type=Path)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--arm", choices=("sleep",), default="sleep", help=argparse.SUPPRESS)
    p.add_argument("--epoch-limit", type=int, default=1, choices=(1,))
    p.add_argument("--merge", type=Path)
    args = p.parse_args()
    directory = prepare_directory(args.directory, ROOT)
    ident = dict(identity(), gate20_sha256=digest(GATE20), prose_manifest_sha256=digest(MANIFEST))
    if args.merge:
        merge(directory, ident, args.merge)
    elif args.seed is not None:
        advance(directory, ident, args.seed, arms=ARMS, boundary_policy=boundary, quine_arm=QUINE)
    else:
        p.error("provide --seed or --merge")


if __name__ == "__main__":
    main()
