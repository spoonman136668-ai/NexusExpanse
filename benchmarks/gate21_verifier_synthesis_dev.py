"""Bounded exact raw-byte synthesis, interference, and fresh-process transfer."""
import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.dev_artifacts import atomic, digest
from benchmarks.gate11_selective_kuramoto_dev import SEEDS, identity, setup
from benchmarks.gate_custody import peak_rss, prepare_directory, writer
from nexus_expanse.verifier_agent import VerifierFeedback, VerifierGatedNexusAgent

RANDOM = "random_search"
UNGATED = "unverified_memory_agent"
FRESH = "fresh_verifier_agent"
GATED = "verifier_gated_nexus_agent"
ARMS = (RANDOM, UNGATED, FRESH, GATED)
PHASES = ("primitives", "shifted", "distractors", "cold_return")
GATE19 = ROOT / "evidence/gate19-po-agency-dev-001.json"
LIMIT = 6


@dataclass(frozen=True)
class Problem:
    source: bytes
    target: bytes
    skin: int
    depth: int
    scored: bool


class World:
    def __init__(self, seed):
        rng = random.Random(seed)
        self.alphabet = bytes(rng.sample(range(33, 127), 8))
        ops = [(x + 1) % 8 for x in range(8)], [(3 * x) % 8 for x in range(8)], [
            x ^ 5 for x in range(8)], [((x << 1) | (x >> 2)) & 7 for x in range(8)]
        self.maps = tuple(bytes(self.alphabet[x] for x in table) for table in ops)
        codes = rng.sample(range(128, 256), 8)
        self.skins = (bytes(codes[:4]), bytes(codes[4:]), bytes(codes[1:4] + codes[:1]))
        self._tables = {skin: dict(zip(actions, self.maps, strict=True))
                        for skin, actions in enumerate(self.skins)}

    def run(self, source, program, skin):
        out = [source]
        for op in program:
            table = self._tables[skin][op]
            out.append(bytes(table[self.alphabet.index(x)] for x in out[-1]))
        return tuple(out)

    def verify(self, problem, program):
        if len(program) != problem.depth or any(a not in self.skins[problem.skin] for a in program):
            raise ValueError("unknown raw operator or illegal program length")
        return VerifierFeedback(program, problem.target, self.run(problem.source, program, problem.skin))


def primitives(world, skin, positions):
    tasks = []
    for i in positions:
        source = bytes(world.alphabet[(i + k) % 8] for k in (0, 1, 3))
        for operator in world.skins[skin]:
            target = world.run(source, bytes((operator,)), skin)[-1]
            tasks.append(Problem(source, target, skin, 1, False))
    return tasks


def compositions(world, rng):
    known = {world.alphabet: 0}
    pools = {}
    for depth in (1, 2, 3):
        signatures = set()
        for path in itertools.product(world.skins[0], repeat=depth):
            signature = world.run(world.alphabet, bytes(path), 0)[-1]
            if signature not in known:
                signatures.add(signature)
                known[signature] = depth
        signatures = sorted(signatures)
        rng.shuffle(signatures)
        split = len(signatures) // 2
        pools[depth] = (signatures[:split], signatures[split:])
    stages = []
    for stage in range(2):
        tasks, seen = [], set()
        for depth in (2, 3):
            n = 0
            for _ in range(20_000):
                skin = n % 2
                indices = [rng.randrange(8) for _ in range(3 + stage)]
                source = bytes(world.alphabet[k] for k in indices)
                project = lambda signature: bytes(signature[k] for k in indices)
                target = project(rng.choice(pools[depth][stage]))
                if (source, target) in seen or any(
                    project(s) == target for s in known if known[s] < depth
                ) or any(project(s) == target for s in pools[depth][1 - stage]):
                    continue
                tasks.append(Problem(source, target, skin, depth, True))
                seen.add((source, target))
                n += 1
                if n == 16:
                    break
            if n != 16:
                raise ValueError("could not allocate disjoint minimum-depth tasks")
        rng.shuffle(tasks)
        stages.append(tasks)
    return stages


def fixture(seed):
    world, rng = World(seed), random.Random(seed + 2100)
    shifted, cold = compositions(world, rng)
    first = primitives(world, 0, range(8))
    distractor = primitives(world, 2, (1, 3, 5, 7))
    rng.shuffle(first)
    rng.shuffle(distractor)
    tasks = first, primitives(world, 1, (0, 2, 5)) + shifted, distractor, cold
    description = dict(alphabet=world.alphabet.hex(), maps=[t.hex() for t in world.maps],
                       skins=[s.hex() for s in world.skins],
                       phases=[[(p.source.hex(), p.target.hex(), p.skin, p.depth, p.scored) for p in rows]
                               for rows in tasks])
    return world, tasks, hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()


def programs_for(world, p):
    return [bytes(x) for x in itertools.product(world.skins[p.skin], repeat=p.depth)]


def fingerprint(agent, world, tasks):
    if agent is None:
        return hashlib.sha256(b"").hexdigest()
    code = hashlib.sha256(agent.snapshot())
    for p in tasks[:4]:
        code.update(agent.propose(p.source, p.target, programs_for(world, p), set()))
    return code.hexdigest()


def play(agent, world, tasks, seed, phase):
    start, cpu = time.perf_counter(), time.process_time()
    rows = []
    peak = len(agent.snapshot()) if agent else 0
    for i, p in enumerate(tasks):
        programs = programs_for(world, p)
        random.Random(seed * 10000 + phase * 1000 + i).shuffle(programs)
        used, probes = set(), []
        for sequence in range(min(LIMIT, len(programs))):
            program = agent.propose(p.source, p.target, programs, used) if agent else programs[sequence]
            used.add(program)
            checked = world.verify(p, program)
            admitted = agent.observe(checked) if agent else False
            if agent:
                peak = max(peak, len(agent.snapshot()))
            probes.append(dict(program=program.hex(), trace=[s.hex() for s in checked.states],
                               residual=checked.residual.hex(), surprise=checked.surprise,
                               accepted=checked.passed, admitted=admitted))
            if checked.passed:
                break
        rows.append(dict(source=p.source.hex(), target=p.target.hex(), skin=p.skin, depth=p.depth,
                         scored=p.scored, solved=probes[-1]["accepted"], probes=probes))
    return dict(tasks=rows, peak_adaptive_bytes=peak, seconds=time.perf_counter() - start,
                cpu_seconds=time.process_time() - cpu, peak_rss_bytes=peak_rss())


def paths(directory, prefix, phase):
    root = directory / f"{prefix}-stage-{phase + 1}"
    return root.with_suffix(".intent.json"), root.with_suffix(".bin"), root.with_suffix(".json")


def history(directory, ident, prefix, fixture_hash):
    rows = []
    for i, phase in enumerate(PHASES):
        intent, snapshot, receipt = paths(directory, prefix, i)
        exists = [p.exists() for p in (intent, snapshot, receipt)]
        if not any(exists):
            continue
        if not all(exists) or len(rows) != i:
            raise ValueError("ambiguous consumed synthesis stage, replay refused")
        expected = dict(identity=ident, phase=phase, fixture_sha256=fixture_hash, arm=prefix,
                        parent=rows[-1]["sha256"] if rows else None)
        data = json.loads(receipt.read_text())
        if (json.loads(intent.read_text()) != expected or any(data.get(k) != v for k, v in expected.items())
                or data["sha256"] != digest(snapshot)):
            raise ValueError("synthesis checkpoint custody mismatch")
        rows.append(data)
    return rows


def advance(directory, ident, seed, arm):
    world, tasks, key = fixture(seed)
    prefix = f"{seed}-{arm}"
    with writer(directory, prefix):
        rows = history(directory, ident, prefix, key)
        i = len(rows)
        if i == len(PHASES):
            return
        previous = rows[-1] if rows else None
        payload = paths(directory, prefix, i - 1)[1].read_bytes() if previous else b""
        if arm == FRESH and i in (1, 2):
            payload = b""
        agent = None
        if arm != RANDOM:
            agent = VerifierGatedNexusAgent.restore(payload) if payload else VerifierGatedNexusAgent(gated=arm != UNGATED)
        restore = None
        if payload:
            restore = dict(fresh_process=previous["pid"] != os.getpid(),
                           exact_bytes=agent.snapshot() == payload, probe=fingerprint(agent, world, tasks[0]))
            restore["equal"] = all((restore["fresh_process"], restore["exact_bytes"],
                                    restore["probe"] == previous["fingerprint"]))
            if not restore["equal"]:
                raise ValueError("synthesis cold restore did not match")
        data = dict(identity=ident, phase=PHASES[i], fixture_sha256=key, arm=prefix,
                    parent=previous["sha256"] if previous else None)
        intent, snapshot, receipt = paths(directory, prefix, i)
        atomic(intent, data)
        result = play(agent, world, tasks[i], seed, i)
        with snapshot.open("xb") as out:
            out.write(agent.snapshot() if agent else b"")
            out.flush()
            os.fsync(out.fileno())
        data.update(result, sha256=digest(snapshot), pid=os.getpid(), restore=restore,
                    fingerprint=fingerprint(agent, world, tasks[0]))
        atomic(receipt, data)
        rows.append(data)
        if len(rows) == len(PHASES):
            atomic(directory / f"{prefix}.json", dict(seed=seed, arm=arm, phases=rows))
        print(json.dumps(dict(seed=seed, phase=PHASES[i], arm=arm,
                              solved=sum(t["solved"] for t in result["tasks"]))))


def summarize(phases):
    shifted, cold = ([t for t in phases[i]["tasks"] if t["scored"]] for i in (1, 3))
    solved = [len(t["probes"]) for t in shifted + cold if t["solved"]]
    return dict(shifted_rate=sum(t["solved"] for t in shifted) / len(shifted),
                cold_rate=sum(t["solved"] for t in cold) / len(cold),
                mean_calls_per_solved=statistics.mean(solved) if solved else None,
                phase3_depth3_rate=sum(t["solved"] for t in cold if t["depth"] == 3) / sum(
                    t["depth"] == 3 for t in cold))


def decision(runs):
    checks = []
    for r in runs:
        arms = r["metrics"]
        candidate = arms[GATED]
        controls = {}
        own = candidate["mean_calls_per_solved"]
        for a in ARMS[:-1]:
            other = arms[a]["mean_calls_per_solved"]
            controls[a] = dict(shifted=candidate["shifted_rate"] > arms[a]["shifted_rate"],
                               cold=candidate["cold_rate"] > arms[a]["cold_rate"],
                               calls=own is not None and other is not None and own < other)
        valid = r["arms"][GATED][-1]["restore"]["equal"] and all(
            phase["peak_adaptive_bytes"] <= 40000 for phases in r["arms"].values() for phase in phases)
        checks.append(dict(seed=r["seed"], controls=controls, engineering=valid))
    engineering = all(r["engineering"] for r in checks)
    wins = [x for r in checks for c in r["controls"].values() for x in c.values()]
    status = "FAIL" if not engineering else "PASS" if all(wins) else "MIXED" if any(wins) else "FAIL"
    return dict(status=status, engineering=engineering, checks=checks)


def merge(directory, ident, dest):
    runs = []
    for seed in SEEDS:
        _, _, key = fixture(seed)
        arms = {}
        for arm in ARMS:
            prefix = f"{seed}-{arm}"
            found = history(directory, ident, prefix, key)
            if (len(found) != len(PHASES) or json.loads((directory / f"{prefix}.json").read_text())
                    != dict(seed=seed, arm=arm, phases=found)):
                raise ValueError("incomplete synthesis evidence")
            arms[arm] = found
        runs.append(dict(seed=seed, arms=arms, metrics={a: summarize(r) for a, r in arms.items()}))
    verdict = decision(runs)
    atomic(dest, dict(schema="nexus-gate21-verifier-synthesis-dev-v1",
                      qualification="SYNTHETIC_FINITE_RAW_BYTE_DEV", identity=ident, runs=runs, **verdict))
    print(json.dumps(verdict))


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
    ident = dict(identity(), gate19_sha256=digest(GATE19))
    if a.merge:
        merge(directory, ident, a.merge)
    elif a.seed is not None and a.arm:
        advance(directory, ident, a.seed, a.arm)
    else:
        p.error("--seed and --arm required")


if __name__ == "__main__":
    main()
