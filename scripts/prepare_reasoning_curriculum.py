"""Build a balanced, offline five-category raw-byte curriculum."""

import argparse
from dataclasses import dataclass
import hashlib
from itertools import repeat
import json
import math
import operator
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.dev_artifacts import _publish, atomic
from nexus_expanse.livebench_eval import BUILTIN_SEED, CATEGORIES, VERIFIERS, dev_records
from nexus_expanse.verified_search import make_verifier
from scripts.prepare_gpu_corpus import GRAM_BYTES, assert_disjoint

SCHEMA = "nexus-reasoning-curriculum-v1"
BLOCK_ROUNDS = 1024
DELIMITER = b"\n\n"


def arithmetic_steps(left, symbol, right):
    tens, units = divmod(left, 10)
    if symbol == "*":
        low, high = units*right, tens*10*right
        return f"{units}*{right}={low}; {tens*10}*{right}={high}; {low}+{high}={low+high}."
    if symbol == "%":
        multiple = (left // right) * right
        return f"{left}//{right}={left//right}; {left}-{multiple}={left-multiple}."
    change = right if symbol == "+" else -right
    carry, digit = divmod(units+change, 10)
    return f"{units}{change:+}={units+change}; tens {tens}{carry:+}={tens+carry}; units {digit}."


def scratchpad(steps, answer):
    return f"<think>{steps}</think>\n{answer}"


def examples(seed, split, index, rng, *, include_think=False):
    """One task per category, with compact prompts and split-specific record markers."""
    mark = f"[{seed}.{index:x}.{split}]\n"
    shift = 32 + 2*index + (split == "E")
    left, right = rng.randrange(13, 97), rng.randrange(1, 13)
    symbol, op = (("+", operator.add), ("-", operator.sub), ("*", operator.mul), ("%", operator.mod))[index % 4]
    answer = f"{op(left,right)}\n"
    if include_think:
        steps = arithmetic_steps(left, symbol, right).replace("; ", f"; {split}: ")
        answer = scratchpad(f"{split}: {steps} [{split}]", answer)
    yield "math", mark+f"{left}{symbol}{right}=", answer, None, None

    letter, prompt = ("+", "Add") if index % 2 == 0 else ("*", "Multiply by")
    rule = operator.add if letter == "+" else operator.mul
    tests = [dict(args=[x], expected=rule(x, shift)) for x in (-11,-3,0,2,7,13)]
    yield "coding", mark+f"# {prompt} {shift}.\ndef f(x):\n    ", f"return x{letter}{shift}\n", tests, None

    next_slot = shift+1
    constraints = [dict(type="all_different", variables=["A","B"]), dict(type="before", left="A", right="B")]
    spec = dict(variables={"A":[shift,next_slot], "B":[shift,next_slot]}, constraints=constraints)
    answer = f"A={shift},B={next_slot}\n"
    if include_think:
        answer = scratchpad(f"{next_slot}: drop A, no later B. Keep A={shift}, then B={next_slot}.", answer)
    yield "reasoning", mark+f"Slots {shift},{next_slot}. A before B. A=n,B=n.\n", answer, None, spec

    target = ("csv", "json", "tsv", "markdown")[index % 4]
    if target == "csv":
        answer = f"id,value\n{shift},{next_slot}\n"
    elif target == "tsv":
        answer = f"id\tvalue\n{shift}\t{next_slot}\n"
    elif target == "markdown":
        answer = f"|id|value|\n|---|---|\n|{shift}|{next_slot}|\n"
    else:
        answer = json.dumps([dict(id=shift, value=next_slot)], separators=(",", ":")) + "\n"
    spec = dict(format=target, columns=["id","value"], rows=[[shift,next_slot]])
    yield "data_analysis", mark+f"Output {target} id,value: {shift},{next_slot}.\n", answer, None, spec

    spec = dict(min_words=2, max_words=2, exact_lines=1, must_start_with="go ", must_end_with=f" {shift}")
    yield "instruction_following", mark+f"Two words, go then {shift}.\n", f"go {shift}\n", None, spec


@dataclass(frozen=True)
class StreamPlan:
    block: bytes
    repetitions: int
    tail: bytes
    rounds: int
    unique_rounds: int

    def coverage(self):
        """Compact witness with the exact windows of the expanded stream."""
        extension = self.block if self.repetitions > 1 else self.tail
        return self.block + extension[:GRAM_BYTES-1]


def plan(seed, split, limit, *, include_think=False):
    rng = random.Random(f"phase23:{seed}:{split}")
    records, size = [], 0
    for index in range(BLOCK_ROUNDS):
        chunk = bytearray()
        for category, prompt, answer, tests, spec in examples(seed, split, index, rng, include_think=include_think):
            prompt, answer = prompt.encode(), answer.encode()
            result = make_verifier(VERIFIERS[category], prompt, tests, spec=spec, allow_think=include_think)(answer)
            if not result.passed or result.total_tests <= 0 or result.passed_tests != result.total_tests:
                raise ValueError(f"curriculum verifier failed for {category} record {index}")
            chunk.extend(prompt + answer + DELIMITER)
        if size + len(chunk) > limit:
            break
        records.append(bytes(chunk))
        size += len(chunk)
    if not records:
        raise ValueError("size cap must fit at least one complete five-category round in each split")
    block = b"".join(records)
    repetitions, budget = divmod(limit, len(block))
    tail, n = bytearray(), 0
    for record in records:
        if len(tail) + len(record) > budget:
            break
        tail.extend(record)
        n += 1
    return StreamPlan(block, repetitions, bytes(tail), repetitions*len(records)+n, len(records))


def builtin_references():
    records = [row["turns"][0].encode() + answer for row, answer in dev_records(BUILTIN_SEED)[2]]
    # Check both plain prompt-answer adjacency and the raw record join convention.
    return {"builtin prompts and answers": b"".join(records),
            "builtin records with delimiters": DELIMITER.join(records)}


def _write_stream(path, stream):
    partial = path.with_suffix(path.suffix + ".partial")
    digest = hashlib.sha256()
    size = 0
    with partial.open("xb") as handle:
        for chunk in repeat(stream.block, stream.repetitions):
            handle.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        handle.write(stream.tail)
        digest.update(stream.tail)
        size += len(stream.tail)
        handle.flush()
        os.fsync(handle.fileno())
    _publish(partial, path)
    return dict(bytes=size, sha256=digest.hexdigest(), examples=stream.rounds*len(CATEGORIES),
                per_category=dict.fromkeys(CATEGORIES, stream.rounds), unique_per_category=stream.unique_rounds)


def prepare(directory, *, size_bytes, seed, include_think=False):
    if type(size_bytes) is not int or size_bytes <= 0:
        raise ValueError("positive total size cap required")
    held_cap = size_bytes // 5
    splits = {name: plan(seed, split, cap, include_think=include_think)
              for name, split, cap in (("train", "T", size_bytes-held_cap), ("eval", "E", held_cap))}
    builtin = builtin_references()
    fit, held = (splits[name].coverage() for name in ("train", "eval"))
    assert_disjoint((fit,), {"curriculum eval":held})
    assert_disjoint((fit, held), builtin)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "curriculum_manifest.json"
    lock = directory / ".prepare-reasoning-curriculum.lock"
    paths = {key:directory / f"stream-{key}.bin" for key in splits}
    finals = (*paths.values(), manifest_path)
    partials = [p.with_suffix(p.suffix + ".partial") for p in finals]
    with lock.open("xb"):
        pass
    created = []
    try:
        for path in (*finals, *partials):
            if os.path.lexists(path):
                raise FileExistsError(path)
        try:
            files = {}
            for split, path in paths.items():
                files[path.name] = _write_stream(path, splits[split])
                created.append(path)
            manifest = dict(schema=SCHEMA, seed=seed, size_cap_bytes=size_bytes, files=files,
                            verifier_passed=sum(x["examples"] for x in files.values()),
                            builtin_reference_sha256={name:hashlib.sha256(raw).hexdigest() for name,raw in builtin.items()},
                            shared_32_byte_windows=dict(train_eval=0, train_builtin=0, eval_builtin=0),
                            donor_forward_calls=0)
            if include_think:
                manifest["include_think"] = True
            atomic(manifest_path, manifest)
        except BaseException:
            for path in (*created, *partials):
                path.unlink(missing_ok=True)
            raise
        return manifest
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--size-mb", type=float, default=1.0, help="total cap in decimal megabytes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-think", action="store_true", help="include worked math and logic scratchpads")
    parser.add_argument("--builtin-dev-suite", action="store_true",
                        help="check builtin held-suite disjointness (always enabled)")
    args = parser.parse_args()
    if not math.isfinite(args.size_mb) or args.size_mb <= 0:
        parser.error("--size-mb must be finite and positive")
    try:
        manifest = prepare(args.out_dir, size_bytes=int(args.size_mb*1_000_000), seed=args.seed,
                           include_think=args.include_think)
    except (ValueError, OSError, OverflowError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, allow_nan=False))


if __name__ == "__main__":
    main()
