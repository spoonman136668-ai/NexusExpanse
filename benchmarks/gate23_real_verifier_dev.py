"""AST-executed synthesis with matched deterministic failed-trace corruption."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.dev_artifacts import digest
from benchmarks.gate11_selective_kuramoto_dev import SEEDS, identity, setup
from benchmarks.gate21_verifier_synthesis_dev import (
    ARMS, Problem, advance, merge, primitives,
)
from benchmarks.gate_custody import prepare_directory
from nexus_expanse.expr_verifier import EXPRESSIONS, run_pipeline, verify
from nexus_expanse.verifier_agent import VerifierFeedback

PHASES = ("primitives", "unseen", "cold_return")
GATE21 = ROOT / "evidence/gate21-verifier-synthesis-dev-001.json"


class ExpressionWorld:
    def __init__(self, seed):
        self.alphabet = bytes(range(8))
        codes = bytes(random.Random(seed).sample(range(128, 256), len(EXPRESSIONS)))
        self.skins = (codes, codes)
        self.expressions = dict(zip(codes, EXPRESSIONS, strict=True))

    def run(self, source, program, skin):
        return run_pipeline(source, program, self.expressions)

    def verify(self, problem, program):
        if len(program) != problem.depth:
            raise ValueError("program length does not match task")
        return verify(problem.source, problem.target, program, self.expressions)


def composition_tasks(world, rng):
    ops = world.skins[0]
    reference = (bytes(range(8)), bytes((3, 1, 5, 2)), bytes((2, 5, 1)))
    programs = {n: [bytes(p) for p in itertools.product(ops, repeat=n)] for n in (1, 2, 3)}
    seen = {reference}
    splits = {}
    for n, paths in programs.items():
        groups = {}
        for path in paths:
            key = tuple(world.run(x, path, 0)[-1] for x in reference)
            if key not in seen:
                groups.setdefault(key, []).append(path)
        seen.update(groups)
        ordered = sorted(groups)
        rng.shuffle(ordered)
        midpoint = len(ordered) // 2
        splits[n] = tuple([p for key in half for p in groups[key]]
                          for half in (ordered[:midpoint], ordered[midpoint:]))
    phases = []
    for stage in (0, 1):
        tasks = []
        for depth in (2, 3):
            count = 0
            used = set()
            for _ in range(4000):
                source = bytes(rng.randrange(8) for _ in range(3 + stage))
                path = rng.choice(splits[depth][stage])
                target = world.run(source, path, 0)[-1]
                shorter = [p for n in range(1, depth) for p in programs[n]]
                excluded = shorter + splits[depth][1 - stage]
                if target == source or (source, target) in used or any(
                    world.run(source, other, 0)[-1] == target for other in excluded
                ):
                    continue
                tasks.append(Problem(source, target, 0, depth, True))
                used.add((source, target))
                count += 1
                if count == 16:
                    break
            if count != 16:
                raise ValueError("could not allocate the frozen exact-depth tasks")
        rng.shuffle(tasks)
        phases.append(tasks)
    return phases


def fixture(seed):
    world = ExpressionWorld(seed)
    rng = random.Random(seed + 2300)
    seen, cold = composition_tasks(world, rng)
    initial = primitives(world, 0, range(8))
    rng.shuffle(initial)
    phases = initial, seen, cold
    body = dict(operators={str(k): v.hex() for k, v in world.expressions.items()},
                phases=[[(p.source.hex(), p.target.hex(), p.depth, p.scored) for p in rows] for rows in phases])
    return world, phases, hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def visible_trace(checked, seed, phase, task):
    if checked.passed or len(checked.program) < 2:
        return checked
    address = f"{seed}:{phase}:{task}:".encode("ascii") + checked.program
    sample = hashlib.sha256(address).digest()
    if sample[0] >= 64:
        return checked
    states = list(checked.states)
    i = 1 + sample[1] % (len(states) - 2)
    broken = bytearray(states[i])
    broken[sample[2] % len(broken)] ^= 1 << (sample[3] % 3)
    states[i] = bytes(broken)
    return VerifierFeedback(checked.program, checked.target, tuple(states))


def main():
    setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True, type=Path)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--epoch-limit", type=int, default=1, choices=(1,))
    p.add_argument("--merge", type=Path)
    a = p.parse_args()
    directory = prepare_directory(a.directory, ROOT)
    ident = dict(identity(), gate21_sha256=digest(GATE21))
    if a.merge:
        merge(directory, ident, a.merge, phases=PHASES, make_fixture=fixture,
              schema="nexus-gate23-real-verifier-dev-v1")
    elif a.seed is not None and a.arm:
        advance(directory, ident, a.seed, a.arm, phases=PHASES, make_fixture=fixture,
                structured=True, trace_for=visible_trace)
    else:
        p.error("provide --seed and --arm")


if __name__ == "__main__":
    main()
