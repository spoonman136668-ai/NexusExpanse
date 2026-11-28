"""Build a deterministic six-file raw-byte corpus for Nexus Expanse CLI training."""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile


SPLITS = ("train", "eval")
DOMAINS = ("math", "code", "prose")
MIN_BYTES = 50_000
GRAM_BYTES = 32


def math_record(rng, split):
    if split == "train":
        a, b = rng.randrange(1000, 10000), rng.randrange(2, 100)
        return f"{a}+{b}={a + b}\n{a}-{b}={a - b}\n{a}*{b}={a * b}\n{a}%{b}={a % b}\n"
    a, b = rng.randrange(10000, 100000), rng.randrange(101, 1000)
    return f"{a} + {b} = {a + b}\n{a} - {b} = {a - b}\n{a} * {b} = {a * b}\n{a} % {b} = {a % b}\n"


def code_record(rng, split, index):
    factor = rng.randrange(2, 120)
    delta = rng.randrange(1, 120)
    if split == "train":
        return (
            f"def scale_{index:07d}(value):\n"
            f"    return value * {factor} + {delta}\n\n"
            f"def accept_{index:07d}(value):\n"
            f"    return value % {factor} == {delta % factor}\n\n"
        )
    return (
        f"def adjust_{index:07d}(entry, baseline={delta}):\n"
        f"    total = entry * {factor}\n"
        f"    return total - baseline\n\n"
        f"def within_{index:07d}(entry):\n"
        f"    return {delta} <= entry < {delta + factor}\n\n"
    )


def prose_record(rng, split):
    count = rng.randrange(2, 120)
    if split == "train":
        verb = rng.choice(("logged", "reviewed", "checked", "archived"))
        item = rng.choice(("sensor readings", "trial summaries", "work orders", "field sketches"))
        when = rng.choice(("before lunch", "at the end of the shift", "after calibration"))
        return f"## Shift log\n\nThe technician {verb} {count} {item} {when}. The next shift took over.\n\n"
    place = rng.choice(("harbor", "ridge", "crossroads", "orchard"))
    weather = rng.choice(("A thin mist drifted", "The wind rose", "Rain gathered in pools"))
    return f"### Morning notes\n\n{weather} near the {place}. By noon, {count} travelers had passed.\n\n"


def corpus_bytes(seed, split, domain, size):
    rng = random.Random(f"{seed}:{split}:{domain}")
    raw = bytearray()
    index = 0
    while len(raw) < size:
        if domain == "math":
            record = math_record(rng, split)
        elif domain == "code":
            record = code_record(rng, split, index)
        else:
            record = prose_record(rng, split)
        raw.extend(record.encode("ascii"))
        index += 1
    return bytes(raw[:size])


def grams(raw):
    return (raw[i:i + GRAM_BYTES] for i in range(len(raw) - GRAM_BYTES + 1))


class CorpusOverlapError(ValueError):
    """An exact byte window occurs in both corpus splits."""


def assert_disjoint(train, evaluation):
    train = tuple(train)
    if len(evaluation) == 1:
        name, raw = next(iter(evaluation.items()))
        if len(raw) < sum(map(len, train)):
            held_grams = set(grams(raw))
            if any(gram in held_grams for data in train for gram in grams(data)):
                raise CorpusOverlapError(f"{name} shares an exact {GRAM_BYTES}-byte window with the training split")
            return
    train_grams = {gram for raw in train for gram in grams(raw)}
    for name, raw in evaluation.items():
        if any(gram in train_grams for gram in grams(raw)):
            raise CorpusOverlapError(f"{name} shares an exact {GRAM_BYTES}-byte window with the training split")


def allocate(total):
    # 80% train and 20% eval, evenly divided among the three domains.
    train = total * 4 // 5
    sizes = {}
    for split, split_size in zip(SPLITS, (train, total - train), strict=True):
        size, remainder = divmod(split_size, len(DOMAINS))
        for index, domain in enumerate(DOMAINS):
            sizes[split, domain] = size + int(index < remainder)
    return sizes


def prepare(out_dir, total, seed):
    if type(total) is not int or total < MIN_BYTES:
        raise ValueError(f"corpus requires at least {MIN_BYTES} bytes")
    files = {}
    train, evaluation = {}, {}
    for (split, domain), size in allocate(total).items():
        name = f"{domain}-{split}.bin"
        raw = corpus_bytes(seed, split, domain, size)
        (train if split == "train" else evaluation)[name] = raw
        files[f"{split}_{domain}"] = dict(file=name, bytes=size, sha256=hashlib.sha256(raw).hexdigest())
    assert_disjoint(train.values(), evaluation)
    manifest = dict(
        schema="nexus-gpu-corpus-v1", seed=seed, total_bytes=total,
        train_eval_32byte_overlap=0, files=files,
    )
    payloads = {**train, **evaluation}
    payloads["corpus_manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode("ascii")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nexus-corpus-", dir=out_dir) as temp:
        stage = Path(temp)
        for name, raw in payloads.items():
            (stage / name).write_bytes(raw)
        published = []
        try:
            # Link atomically without overwriting. Publish the manifest last.
            for name in payloads:
                target = out_dir / name
                os.link(stage / name, target)
                published.append(name)
        except OSError:
            for name in published:
                target = out_dir / name
                if target.exists() and target.samefile(stage / name):
                    target.unlink()
            raise
    return manifest


def size_bytes(text):
    try:
        megabytes = Decimal(text)
        if not megabytes.is_finite() or megabytes < Decimal("0.05"):
            raise ValueError
        return int(megabytes * 1_000_000)
    except (InvalidOperation, ValueError, OverflowError):
        raise argparse.ArgumentTypeError("size must be a finite number of MB, at least 0.05") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--size-mb", type=size_bytes, default=4_000_000, metavar="MB",
                        help="total decimal MB across all six files (default: 4.0)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        manifest = prepare(args.out_dir, args.size_mb, args.seed)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"prepare_gpu_corpus: {exc}\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
