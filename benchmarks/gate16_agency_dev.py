"""Match four causal policies on interactive two-step raw-byte DEV goals."""
import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.gate11_selective_kuramoto_dev import SEEDS, identity, setup
from benchmarks.gate9_jepa_dev import atomic, digest
from benchmarks.gate_custody import peak_rss, prepare_directory, writer
from nexus_expanse.active_agent import NexusActiveAgent, exploration_bonus, raw_symbol
from nexus_expanse.consolidation import ConsolidatingMemory

RANDOM = "random_policy"
REACTIVE = "reactive_memory_only"
FRESH = "fresh_online_planner"
PERSIST = "nexus_active_agent"
ARMS = (RANDOM, REACTIVE, FRESH, PERSIST)
GOALS = 24
BUDGET = 4
STATES = 14
PHASES = ("world_a", "silent_world_b", "cold_world_a")


def worlds(seed):
    rng = random.Random(seed)
    alphabet = [bytes((i,)) for i in rng.sample(range(256), STATES + 3)]
    states, actions = alphabet[:STATES], alphabet[STATES:]
    # Random Hamiltonian cycles, with distinct outgoing results in each row.
    # The runner owns these tables. The agent sees only resulting raw bytes.
    for _ in range(1000):
        maps = []
        for _ in actions:
            path = rng.sample(states, len(states))
            maps.append(dict(zip(path, path[1:] + path[:1], strict=True)))
        if any(len({m[s] for m in maps}) != len(actions) for s in states):
            continue
        a = {s: dict(zip(actions, [m[s] for m in maps], strict=True)) for s in states}
        b = {s: dict(zip(actions, [m[s] for m in maps[1:] + maps[:1]], strict=True)) for s in states}
        pairs = [(s, g) for s in states for g in states
                 if g != s and g not in a[s].values()
                 and any(g in a[mid].values() for mid in a[s].values())]
        if len(pairs) >= GOALS * len(PHASES):
            rng.shuffle(pairs)
            schedule = [pairs[i:i + GOALS] for i in range(0, GOALS * len(PHASES), GOALS)]
            return (a, b, a), tuple(actions), schedule
    raise ValueError("could not materialize deterministic two-step worlds")


def fixture_hash(worlds_and_goals):
    tables, actions, schedule = worlds_and_goals
    desc = [{s.hex(): {a.hex(): y.hex() for a, y in row.items()} for s, row in t.items()} for t in tables]
    desc.append([a.hex() for a in actions])
    desc.append([[[s.hex(), g.hex()] for s, g in rows] for rows in schedule])
    return hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest()


class _Reactive:
    def __init__(self, memory=None):
        self.memory = ConsolidatingMemory(40_000, "sparse_support", 2) if memory is None else memory
        self.context = None

    def choose(self, observation, candidates, goal):
        result = {}
        target = raw_symbol(goal)
        for action in candidates:
            p, _ = self.memory.predict_probs(observation + action)
            result[action] = (float(p[target]) + exploration_bonus(self.memory, observation, action), p)
        selected = max(result, key=lambda a: result[a][0])
        self.context = observation + selected
        return selected, result[selected][1]

    def observe(self, successor):
        self.memory.observe_transition(self.context, raw_symbol(successor))
        self.context = None
        return False, False

    def snapshot(self):
        return self.memory.snapshot(max_bytes=40_000)


class _Random:
    def choose(self, observation, candidates, goal):
        return candidates[0], torch.full((256,), 1 / 256, dtype=torch.float64)

    def observe(self, successor):
        return False, False

    def snapshot(self):
        return b""


def restore(arm, payload):
    if arm == RANDOM:
        if payload:
            raise ValueError("random policy checkpoint must be empty")
        return _Random()
    if arm == REACTIVE:
        if not payload:
            return _Reactive()
        return _Reactive(ConsolidatingMemory.restore(payload, max_bytes=40_000))
    return NexusActiveAgent.restore(payload) if payload else NexusActiveAgent()


def probe(policy, actions, schedule):
    h = hashlib.sha256(policy.snapshot())
    for state, goal in schedule[:4]:
        action, p = copy.deepcopy(policy).choose(state, actions, goal)
        h.update(action)
        h.update(p.contiguous().numpy().tobytes())
    return h.hexdigest()


def play(policy, table, actions, goals, seed, phase):
    metrics = []
    peak = len(policy.snapshot())
    start, cpu = time.perf_counter(), time.process_time()
    for attempt, (origin, goal) in enumerate(goals):
        moves = []
        state = origin
        for step in range(BUDGET):
            rng = random.Random(seed * 10_000 + phase * 1_000 + attempt * BUDGET + step)
            candidates = rng.sample(actions, len(actions))
            action, predicted = policy.choose(state, candidates, goal)
            successor = table[state][action]
            surprise, switched = policy.observe(successor)
            peak = max(peak, len(policy.snapshot()))
            moves.append(dict(observation=state.hex(), action=action.hex(), outcome=successor.hex(),
                              forecast_top1=int(predicted.argmax()),
                              actual_probability=float(predicted[raw_symbol(successor)]),
                              action_order=[c.hex() for c in candidates],
                              surprise=surprise, switched=switched))
            state = successor
            if state == goal:
                break
        metrics.append(dict(start=origin.hex(), goal=goal.hex(), budget=BUDGET,
                            minimum_steps=2, completed=state == goal, moves=moves))
    return dict(goals=metrics, completions=sum(m["completed"] for m in metrics),
                actual_steps=sum(len(m["moves"]) for m in metrics), max_steps=GOALS * BUDGET,
                peak_serialized_bytes=peak, seconds=time.perf_counter() - start,
                cpu_seconds=time.process_time() - cpu, peak_rss_bytes=peak_rss())


def phase_paths(directory, name, phase):
    stem = directory / f"{name}-phase-{phase + 1}"
    return stem.with_suffix(".intent.json"), stem.with_suffix(".bin"), stem.with_suffix(".json")


def receipts(directory, name, ident, fixture):
    valid = []
    parent = None
    for i in range(len(PHASES)):
        intent, path, meta = phase_paths(directory, name, i)
        have = [p.exists() for p in (intent, path, meta)]
        if not any(have):
            continue
        if not all(have) or len(valid) != i:
            raise RuntimeError(f"ambiguous consumed phase, replay refused: {path}")
        given = json.loads(meta.read_text())
        match = dict(identity=ident, phase=PHASES[i], arm=name, parent=parent, fixture_sha256=fixture)
        if json.loads(intent.read_text()) != match or any(given.get(k) != v for k, v in match.items()):
            raise ValueError("phase identity drift")
        if digest(path) != given["snapshot_sha256"] or len(path.read_bytes()) != given["final_serialized_bytes"]:
            raise ValueError("phase snapshot custody mismatch")
        parent = given["snapshot_sha256"]
        valid.append(given)
    return valid


def advance(directory, ident, seed, arm):
    name = f"{seed}-{arm}"
    generated = worlds(seed)
    tables, actions, schedule = generated
    fixture = fixture_hash(generated)
    with writer(directory, name):
        completed = receipts(directory, name, ident, fixture)
        final = directory / f"{name}.json"
        if len(completed) < len(PHASES):
            i = len(completed)
            previous = completed[-1] if completed else None
            parent = previous["snapshot_sha256"] if previous else None
            payload = phase_paths(directory, name, i - 1)[1].read_bytes() if previous else b""
            policy = restore(arm, payload if arm != FRESH else b"")
            recovered = None
            if arm == PERSIST and previous:
                recovered = dict(fresh_process=previous["worker_pid"] != os.getpid(),
                                 exact_bytes=policy.snapshot() == payload,
                                 probe=probe(policy, actions, schedule[0]))
                recovered["equal"] = recovered["exact_bytes"] and recovered["fresh_process"] and (
                    recovered["probe"] == previous["probe"])
                if not recovered["equal"]:
                    raise ValueError("fresh phase could not reproduce persisted agent")
            data = dict(identity=ident, phase=PHASES[i], arm=name, parent=parent, fixture_sha256=fixture)
            intent, path, meta = phase_paths(directory, name, i)
            atomic(intent, data)
            result = play(policy, tables[i], actions, schedule[i], seed, i)
            after = policy.snapshot()
            with path.open("xb") as out:
                out.write(after)
                out.flush()
                os.fsync(out.fileno())
            data.update(result, final_serialized_bytes=len(after), snapshot_sha256=digest(path),
                        worker_pid=os.getpid(), restore=recovered, probe=probe(policy, actions, schedule[0]))
            atomic(meta, data)
            completed.append(data)
            print(json.dumps(dict(seed=seed, arm=arm, phase=PHASES[i], completions=result["completions"],
                                  peak_serialized_bytes=result["peak_serialized_bytes"])))
        if len(completed) == len(PHASES) and not final.exists():
            atomic(final, dict(seed=seed, arm=arm, phases=completed))


def decision(runs):
    cells = []
    for r in runs:
        scores = {a: [p["completions"] for p in r["arms"][a]] for a in ARMS}
        own = scores[PERSIST]
        totals = {a: sum(own) > sum(scores[a]) for a in ARMS[:-1]}
        cold = {a: own[-1] > scores[a][-1] for a in ARMS[:-1]}
        valid = r["arms"][PERSIST][-1]["restore"]["equal"] and all(
            p["peak_serialized_bytes"] <= 40_000 for phases in r["arms"].values() for p in phases)
        cells.append(dict(seed=r["seed"], totals=totals, cold=cold, engineering=valid))
    engineering = all(c["engineering"] for c in cells)
    pass_all = all(all(c["totals"].values()) and all(c["cold"].values()) for c in cells)
    mixed = all(c["totals"][RANDOM] and c["totals"][REACTIVE] for c in cells) and (
        not all(c["cold"][FRESH] for c in cells))
    status = "FAIL"
    if engineering:
        status = "PASS" if pass_all else "MIXED" if mixed else "FAIL"
    return dict(status=status, engineering=engineering, checks=cells)


def merge(directory, ident, dest):
    runs = []
    for seed in SEEDS:
        fixture = fixture_hash(worlds(seed))
        arms = {}
        for arm in ARMS:
            name = f"{seed}-{arm}"
            phases = receipts(directory, name, ident, fixture)
            check = json.loads((directory / f"{name}.json").read_text())
            if check != dict(seed=seed, arm=arm, phases=phases) or len(phases) != len(PHASES):
                raise ValueError("missing completed arm")
            arms[arm] = phases
        runs.append(dict(seed=seed, fixture_sha256=fixture, arms=arms))
    report = decision(runs)
    atomic(dest, dict(schema="nexus-gate16-agency-dev-v1", qualification="SYNTHETIC_INTERACTIVE_DEV",
                      identity=ident, runs=runs, **report))
    print(json.dumps(report))


def main():
    setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--epoch-limit", type=int, default=1, choices=(1,))
    p.add_argument("--merge", type=Path)
    args = p.parse_args()
    ident = identity()
    directory = prepare_directory(args.directory, ROOT)
    if args.merge:
        merge(directory, ident, args.merge)
    elif args.seed is not None and args.arm is not None:
        advance(directory, ident, args.seed, args.arm)
    else:
        p.error("--seed and --arm required")


if __name__ == "__main__":
    main()
