"""Pack whole local or opt-in Hub documents into six leakage-checked byte files."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.dev_artifacts import atomic, atomic_bytes
from scripts.prepare_gpu_corpus import DOMAINS, GRAM_BYTES, assert_disjoint, grams, math_record, size_bytes

MIN_BYTES = 50_000
SPLIT_ATTEMPTS = 24
MAX_GRAPH_DOCUMENTS = 256
RANKED_SPLIT_ATTEMPTS = 4
MAX_HUB_RECORDS = 50_000
SUFFIXES = {"prose": {".md", ".txt", ".rst"}, "math": {".md", ".txt", ".rst", ".tex"},
            "code": {".py", ".js", ".ts", ".go", ".rs", ".c", ".cpp", ".java", ".sh"}}


def local_documents(path, domain):
    path = Path(path)
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = sorted(p for p in path.rglob("*")
                       if p.suffix.lower() in SUFFIXES[domain] and p.is_file() and not p.is_symlink())
    else:
        raise ValueError(f"missing {domain} corpus input: {path}")
    for file in paths:
        raw = file.read_bytes()
        raw.decode("utf-8")
        yield raw


def hub_documents(spec, field):
    repo, separator, config = spec.partition(":")
    if not repo or (separator and not config):
        raise ValueError("expected a Hub dataset as repo or repo:config")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ValueError("Hub streaming requires the optional datasets package") from exc
    for record in load_dataset(repo, config or None, split="train", streaming=True):
        value = record.get(field)
        if not isinstance(value, str):
            raise ValueError(f"Hub record requires a text field named {field}")
        yield value.encode("utf-8")


def collect(source, limit, *, streaming=False):
    documents, seen = [], set()
    total = 0
    # Bound the received Hub text even in streams dominated by duplicates.
    received = 0
    for index, raw in enumerate(source):
        received += len(raw)
        if streaming and (received > 3 * limit or index >= MAX_HUB_RECORDS):
            break
        if not raw.strip() or len(raw) + 1 > limit - MIN_BYTES:
            continue
        identity = hashlib.sha256(raw).digest()
        if identity in seen:
            continue
        seen.add(identity)
        documents.append(raw + b"\n")
        total += len(documents[-1])
        if streaming and total >= 2 * limit:
            break
    return documents


def eval_target(cap, available, fraction):
    return min(cap - MIN_BYTES, max(MIN_BYTES, int(min(cap, available) * fraction)))


def eval_documents(order, limit, target):
    picked = set()
    total = 0
    for index, doc in enumerate(order):
        if total + len(doc) > limit:
            continue
        picked.add(index)
        total += len(doc)
        if total >= target:
            break
    evaluation = [doc for index, doc in enumerate(order) if index in picked]
    candidates = [doc for index, doc in enumerate(order) if index not in picked]
    return evaluation, candidates


def train_documents(candidates, limit, held_grams):
    raw = bytearray()
    selected = 0
    for doc in candidates:
        if len(raw) + len(doc) > limit:
            continue
        # Include new n-grams that cross the boundary of two retained documents.
        boundary = bytes(raw[-(GRAM_BYTES - 1):]) + doc
        if any(gram in held_grams for gram in grams(boundary)):
            continue
        raw.extend(doc)
        selected += 1
    return bytes(raw), selected


def synthetic_math(seed, cap, fraction):
    target = eval_target(cap, cap, fraction)
    evaluation, training = [], []
    held_rng = random.Random(f"real-corpus-math:{seed}:eval")
    held_bytes = 0
    while held_bytes < target:
        record = math_record(held_rng, "eval").encode("ascii")
        if held_bytes + len(record) > cap - MIN_BYTES:
            break
        evaluation.append(record)
        held_bytes += len(record)
    train_rng = random.Random(f"real-corpus-math:{seed}:train")
    train_bytes = 0
    while train_bytes + held_bytes < cap:
        record = math_record(train_rng, "train").encode("ascii")
        if train_bytes + held_bytes + len(record) > cap:
            break
        training.append(record)
        train_bytes += len(record)
    if min(held_bytes, train_bytes) < MIN_BYTES:
        raise ValueError("increase the byte cap to fit whole synthetic math records")
    return {"train": training, "eval": evaluation}


def conflict_graph(documents):
    rows = [(domain, doc) for domain in DOMAINS for doc in documents.get(domain, ())]
    owners = {}
    conflicts = [1 << i for i in range(len(rows))]
    for i, (_, doc) in enumerate(rows):
        bit = 1 << i
        prior = 0
        for gram in set(grams(doc)):
            others = owners.get(gram, 0)
            owners[gram] = others | bit
            prior |= others
        conflicts[i] |= prior
        while prior:
            low = prior & -prior
            conflicts[low.bit_length() - 1] |= bit
            prior ^= low
    return rows, conflicts


def conflict_aware_holdout(graph, targets, caps, seed):
    """Choose whole held documents that leave enough compatible training bytes."""
    rows, conflicts = graph
    order = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(order)
    lengths = [len(doc) for _, doc in rows]
    noise = {i: rng.uniform(0.8, 1.2) for i in order}
    held = {d: [] for d in DOMAINS}
    held_bytes = dict.fromkeys(DOMAINS, 0)
    chosen = set()
    blocked = 0
    while any(held_bytes[d] < targets[d] for d in DOMAINS):
        best, best_cost = None, float("inf")
        for i in order:
            domain, _ = rows[i]
            remaining = targets[domain] - held_bytes[domain]
            if i in chosen or remaining <= 0 or held_bytes[domain] + lengths[i] > caps[domain] - MIN_BYTES:
                continue
            combined = blocked | conflicts[i]
            available = dict.fromkeys(DOMAINS, 0)
            loss = 0
            for j, (kind, _) in enumerate(rows):
                bit = 1 << j
                if not combined & bit:
                    available[kind] += lengths[j]
                elif not blocked & bit:
                    loss += lengths[j]
            if any(targets[kind] and available[kind] < MIN_BYTES for kind in DOMAINS):
                continue
            cost = (loss + 1) * noise[i] / min(lengths[i], remaining)
            if cost < best_cost:
                best, best_cost = i, cost
        if best is None:
            return None
        domain, doc = rows[best]
        held[domain].append(doc)
        held_bytes[domain] += lengths[best]
        chosen.add(best)
        blocked |= conflicts[best]
    candidates = {d: [] for d in DOMAINS}
    for i in order:
        domain, doc = rows[i]
        if i not in chosen and not (blocked & (1 << i)):
            candidates[domain].append(doc)
    return held, candidates


def holdout_scores(documents):
    """Rank by shared windows without materializing document pairs."""
    frequencies = Counter()
    for docs in documents.values():
        for doc in docs:
            frequencies.update(set(grams(doc)))
    scores = {}
    for docs in documents.values():
        for doc in docs:
            highest = total = 0
            for gram in grams(doc):
                shared = frequencies[gram] - 1
                highest = max(highest, shared)
                total += shared
            scores[doc] = highest, total / max(1, len(doc) - GRAM_BYTES + 1)
    return scores


def split_payloads(held, candidates, caps):
    evaluation = {domain: b"".join(held[domain]) for domain in DOMAINS}
    held_grams = {gram for raw in evaluation.values() for gram in grams(raw)}
    result = {}
    for domain, raw in evaluation.items():
        result["train", domain] = train_documents(candidates[domain], caps[domain] - len(raw), held_grams)
        result["eval", domain] = raw, len(held[domain])
    return result


def split_corpus(documents, seed, fraction, caps, *, fixed_math=None):
    available = {domain: sum(map(len, documents[domain])) for domain in DOMAINS}
    for domain in DOMAINS:
        if domain == "math" and fixed_math is not None:
            continue
        if available[domain] < 2 * MIN_BYTES or len(documents[domain]) < 2:
            raise ValueError(f"{domain} needs enough distinct whole documents for two {MIN_BYTES:,}-byte files")
    active = {d: documents[d] if d != "math" or fixed_math is None else [] for d in DOMAINS}
    large = sum(map(len, active.values())) > MAX_GRAPH_DOCUMENTS
    scores = holdout_scores(active) if large else None
    targets = {d: (0 if d == "math" and fixed_math is not None else eval_target(caps[d], available[d], fraction))
               for d in DOMAINS}
    graph = None
    repairs = {}
    for attempt in range(RANKED_SPLIT_ATTEMPTS if large else SPLIT_ATTEMPTS):
        if large or attempt < 4:
            held, candidates = {}, {}
            for domain in DOMAINS:
                order = list(active[domain])
                random.Random(f"real-corpus:{seed}:{domain}:{attempt}").shuffle(order)
                if scores is not None:
                    # Penalize a ubiquitous prefix first, then try average
                    # collision frequency for corpora with many short fragments.
                    primary = int(attempt >= RANKED_SPLIT_ATTEMPTS // 2)
                    order.sort(key=lambda doc: (scores[doc][primary], scores[doc][1 - primary]))
                held[domain], candidates[domain] = eval_documents(order, caps[domain] - MIN_BYTES, targets[domain])
        else:
            if graph is None:
                graph = conflict_graph(active)
            selected = conflict_aware_holdout(graph, targets, caps, f"real-corpus:{seed}:{attempt}")
            if selected is None:
                continue
            held, candidates = selected
        if fixed_math is not None:
            held["math"], candidates["math"] = fixed_math["eval"], fixed_math["train"]
        result = split_payloads(held, candidates, caps)
        short = tuple(d for d in DOMAINS if any(len(result[s, d][0]) < MIN_BYTES for s in ("train", "eval")))
        if not short:
            return result
        if large and all(active[d] for d in short) and sum(len(active[d]) for d in short) <= MAX_GRAPH_DOCUMENTS:
            # A hard local split can be retried without adding a large Hub domain
            # to the graph. The rebuilt payload still checks against every holdout.
            repair_targets = {d: targets[d] if d in short else 0 for d in DOMAINS}
            if short not in repairs:
                repairs[short] = conflict_graph({d: active[d] for d in short})
            selected = conflict_aware_holdout(repairs[short], repair_targets, caps, f"real-corpus:repair:{seed}:{attempt}")
            if selected is not None:
                new_held, new_candidates = selected
                for domain in short:
                    held[domain], candidates[domain] = new_held[domain], new_candidates[domain]
                result = split_payloads(held, candidates, caps)
                if all(len(raw) >= MIN_BYTES for raw, _ in result.values()):
                    return result
    raise ValueError(f"could not form six leakage-free whole-document files of {MIN_BYTES:,} bytes within the cap")


def publish(directory, payloads, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "corpus_manifest.json"
    lock = directory / ".prepare-real-corpus.lock"
    with lock.open("xb"):
        pass
    created = []
    try:
        paths = [*(directory / name for name in payloads), manifest_path]
        for path in paths:
            for item in (path, path.with_suffix(path.suffix + ".partial")):
                if os.path.lexists(item):
                    raise FileExistsError(item)
        for path in paths[:-1]:
            atomic_bytes(path, payloads[path.name])
            created.append(path)
        atomic(manifest_path, manifest)
    except BaseException:
        for path in created:
            path.unlink()
        raise
    finally:
        lock.unlink()


def prepare(directory, *, local, hub, field="text", seed=42, fraction=0.10,
            limit=3_000_000, math_fallback=False):
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("eval-fraction must be between 0 and 1")
    if limit < MIN_BYTES * 6:
        raise ValueError("size cap must hold six 50,000-byte files")
    cap, remainder = divmod(limit, len(DOMAINS))
    caps = {domain: cap + (i < remainder) for i, domain in enumerate(DOMAINS)}
    documents, sources = {}, {}
    for domain in DOMAINS:
        path, repo = local.get(domain), hub.get(domain)
        if path and repo:
            raise ValueError(f"choose local input or Hub streaming for {domain}")
        if path:
            documents[domain] = collect(local_documents(path, domain), caps[domain])
            sources[domain] = "local"
        elif repo:
            documents[domain] = collect(hub_documents(repo, field), caps[domain], streaming=True)
            sources[domain] = "hf"
        elif domain == "math" and math_fallback:
            documents[domain] = []
        else:
            raise ValueError(f"provide a corpus input for {domain}")

    try:
        splits = split_corpus(documents, seed, fraction, caps)
    except ValueError:
        if not math_fallback:
            raise
        fixed = synthetic_math(seed, caps["math"], fraction)
        splits = split_corpus(documents, seed, fraction, caps, fixed_math=fixed)
        sources["math"] = "synthetic"
    assert_disjoint([splits["train", d][0] for d in DOMAINS],
                    {domain: splits["eval", domain][0] for domain in DOMAINS})
    del documents
    files, payloads = {}, {}
    for (split, domain), (raw, count) in splits.items():
        name = f"{domain}-{split}.bin"
        payloads[name] = raw
        files[f"{split}_{domain}"] = dict(file=name, bytes=len(raw), documents=count,
                                         sha256=hashlib.sha256(raw).hexdigest())
    manifest = dict(schema="nexus-real-corpus-v1", seed=seed, eval_fraction=fraction, size_limit_bytes=limit,
                    total_bytes=sum(map(len, payloads.values())), sources=sources,
                    overlap_32b=dict.fromkeys(DOMAINS, 0), global_overlap_32b=0, files=files)
    publish(Path(directory), payloads, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    for domain in DOMAINS:
        source = parser.add_mutually_exclusive_group()
        source.add_argument(f"--{domain}-dir", type=Path)
        source.add_argument(f"--hf-{domain}", metavar="REPO[:CONFIG]")
    parser.add_argument("--math-fallback-synthetic", action="store_true")
    parser.add_argument("--hf-text-field", default="text")
    parser.add_argument("--size-mb", type=size_bytes, default=3_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-fraction", type=float, default=0.10)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.out_dir, local={d: getattr(args, d + "_dir") for d in DOMAINS},
                         hub={d: getattr(args, "hf_" + d) for d in DOMAINS}, field=args.hf_text_field,
                         seed=args.seed, fraction=args.eval_fraction, limit=args.size_mb,
                         math_fallback=args.math_fallback_synthetic)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"prepare_real_corpus: {exc}\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
