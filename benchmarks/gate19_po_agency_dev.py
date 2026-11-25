"""Matched hidden-lock DEV worlds with causal raw-byte observations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.dev_artifacts import atomic, digest
from benchmarks.gate11_selective_kuramoto_dev import SEEDS, identity, setup
from benchmarks.gate16_agency_dev import (
    GOALS, PHASES, RANDOM, REACTIVE, fixture_hash, phase_paths, probe, receipts, restore, worlds,
)
from benchmarks.gate_custody import peak_rss, prepare_directory, writer
from nexus_expanse.active_agent import NexusActiveAgent, raw_symbol

MARKOV = "markov_gate16_agent"
FRESH = "fresh_po_planner"
PO = "nexus_po_agent"
ARMS = (RANDOM, REACTIVE, MARKOV, FRESH, PO)
BUDGET = 6
GATE16 = ROOT / "evidence/gate16-agency-dev-001.json"


def hidden_worlds(seed):
    tables, actions, schedule = worlds(seed)
    occupied = set(actions) | set(tables[0])
    unlock = random.Random(seed + 1900).choice([bytes((i,)) for i in range(256) if bytes((i,)) not in occupied])
    all_actions = actions + (unlock,)
    rotated = all_actions[1:] + all_actions[:1]
    mapping = dict(zip(all_actions, rotated, strict=True))
    shifted = {s: {a: tables[0][s][mapping[a]] for a in all_actions if mapping[a] != unlock}
               for s in tables[0]}
    shifted_unlock = next(a for a in all_actions if mapping[a] == unlock)
    return ((tables[0], unlock), (shifted, shifted_unlock), (tables[0], unlock)), all_actions, schedule


def fixture_digest(generated):
    regimes, actions, schedule = generated
    base = fixture_hash(([t for t, _ in regimes], actions, schedule))
    return hashlib.sha256(base.encode() + b"".join(lock for _, lock in regimes)).hexdigest()


def transition(world, observation, opened, action, goal):
    table, unlock = world
    if action == unlock:
        return observation, not opened
    if not opened:
        return observation, False
    successor = table[observation][action]
    return successor, successor != goal


def policy_for(arm, payload):
    if arm in (PO, FRESH):
        return NexusActiveAgent.restore(payload) if payload else NexusActiveAgent(po_mode=True)
    return restore(arm, payload)


def play(policy, world, actions, goals, seed, phase):
    initial = time.perf_counter(), time.process_time()
    trials = []
    peak = len(policy.snapshot())
    for attempt, (origin, goal) in enumerate(goals):
        if isinstance(policy, NexusActiveAgent):
            policy.begin_attempt()
        current, opened = origin, False
        trace = []
        for i in range(BUDGET):
            choices = random.Random(seed * 100_000 + phase * 10_000 + attempt * BUDGET + i).sample(actions, len(actions))
            belief = getattr(policy, "belief", None)
            before = belief.open_probability if belief is not None else None
            action, predicted = policy.choose(current, choices, goal)
            successor, opened = transition(world, current, opened, action, goal)
            surprise, switched = policy.observe(successor)
            peak = max(peak, len(policy.snapshot()))
            trace.append(dict(observation=current.hex(), action=action.hex(), outcome=successor.hex(),
                              actual_probability=float(predicted[raw_symbol(successor)]),
                              choice_order=[a.hex() for a in choices], prior_open_probability=before,
                              posterior_open_probability=belief.open_probability if belief is not None else None,
                              surprise=surprise, switched=switched))
            current = successor
            if current == goal:
                break
        trials.append(dict(start=origin.hex(), goal=goal.hex(), budget=BUDGET, minimum_steps=3,
                           completed=current == goal, moves=trace))
    return dict(goals=trials, completions=sum(t["completed"] for t in trials),
                actual_steps=sum(len(t["moves"]) for t in trials), max_steps=GOALS * BUDGET,
                peak_serialized_bytes=peak, seconds=time.perf_counter() - initial[0],
                cpu_seconds=time.process_time() - initial[1], peak_rss_bytes=peak_rss())


def advance(directory, ident, seed, arm):
    name = f"{seed}-{arm}"
    regimes, actions, schedule = generated = hidden_worlds(seed)
    fixture = fixture_digest(generated)
    with writer(directory, name):
        history = receipts(directory, name, ident, fixture)
        i = len(history)
        if i < len(PHASES):
            previous = history[-1] if i else None
            payload = phase_paths(directory, name, i - 1)[1].read_bytes() if i and arm != FRESH else b""
            policy = policy_for(arm, payload)
            recovered = None
            if i and arm in (PO, MARKOV):
                recovered = dict(fresh_process=previous["worker_pid"] != os.getpid(),
                                 exact_bytes=policy.snapshot() == payload,
                                 probe=probe(policy, actions, schedule[0]))
                recovered["equal"] = all((recovered["fresh_process"], recovered["exact_bytes"],
                                          recovered["probe"] == previous["probe"]))
                if not recovered["equal"]:
                    raise ValueError("fresh phase could not reproduce persisted agent")
            intent, path, meta = phase_paths(directory, name, i)
            data = dict(identity=ident, phase=PHASES[i], arm=name,
                        parent=previous["snapshot_sha256"] if previous else None, fixture_sha256=fixture)
            atomic(intent, data)
            measured = play(policy, regimes[i], actions, schedule[i], seed, i)
            after = policy.snapshot()
            with path.open("xb") as out:
                out.write(after)
                out.flush()
                os.fsync(out.fileno())
            result = dict(data, **measured, final_serialized_bytes=len(after), snapshot_sha256=digest(path),
                          worker_pid=os.getpid(), restore=recovered, probe=probe(policy, actions, schedule[0]))
            atomic(meta, result)
            history.append(result)
            print(json.dumps(dict(seed=seed, arm=arm, phase=PHASES[i], completions=result["completions"],
                                  peak_serialized_bytes=result["peak_serialized_bytes"])))
        final = directory / f"{name}.json"
        if len(history) == len(PHASES) and not final.exists():
            atomic(final, dict(seed=seed, arm=arm, phases=history))


def decision(runs):
    cells = []
    for r in runs:
        scores = {a: [p["completions"] for p in r["arms"][a]] for a in ARMS}
        own = scores[PO]
        totals = {a: sum(own) > sum(scores[a]) for a in ARMS[:-1]}
        cold = {a: own[-1] > scores[a][-1] for a in ARMS[:-1]}
        valid = all(r["arms"][a][-1]["restore"]["equal"] for a in (PO, MARKOV)) and all(
            p["peak_serialized_bytes"] <= 40000 for arm in r["arms"].values() for p in arm)
        cells.append(dict(seed=r["seed"], totals=totals, cold=cold, engineering=valid))
    engineering = all(c["engineering"] for c in cells)
    full = sum(all(c["totals"].values()) and all(c["cold"].values()) for c in cells)
    markov = sum(c["totals"][MARKOV] for c in cells)
    mixed = markov >= 2 and (full == 2 or (markov == 3 and not all(c["cold"][FRESH] for c in cells)))
    status = "FAIL" if not engineering else "PASS" if full == 3 else "MIXED" if mixed else "FAIL"
    return dict(status=status, engineering=engineering, checks=cells, full_wins=full, markov_total_wins=markov)


def merge(directory, ident, dest):
    runs = []
    for seed in SEEDS:
        fixture = fixture_digest(hidden_worlds(seed))
        arms = {}
        for arm in ARMS:
            name = f"{seed}-{arm}"
            phases = receipts(directory, name, ident, fixture)
            check = json.loads((directory / f"{name}.json").read_text())
            if check != dict(seed=seed, arm=arm, phases=phases) or len(phases) != 3:
                raise ValueError("incomplete arm")
            arms[arm] = phases
        runs.append(dict(seed=seed, fixture_sha256=fixture, arms=arms))
    report = decision(runs)
    atomic(dest, dict(schema="nexus-gate19-po-agency-dev-v1", qualification="SYNTHETIC_INTERACTIVE_DEV",
                      identity=ident, runs=runs, **report))
    print(json.dumps(report))


def main():
    setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True, type=Path)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--epoch-limit", default=1, type=int, choices=(1,))
    p.add_argument("--merge", type=Path)
    a = p.parse_args()
    ident = dict(identity(), gate16_sha256=digest(GATE16))
    directory = prepare_directory(a.directory, ROOT)
    if a.merge:
        merge(directory, ident, a.merge)
    elif a.seed is not None and a.arm:
        advance(directory, ident, a.seed, a.arm)
    else:
        p.error("provide a seed and arm")


if __name__ == "__main__":
    main()
