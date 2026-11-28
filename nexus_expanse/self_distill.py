"""Harvest exact, independently re-verified byte traces from a frozen snapshot."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

from benchmarks.dev_artifacts import atomic_bytes
from nexus_expanse.livebench_eval import CATEGORIES, dev_records, suite_from_rows
from nexus_expanse.verified_search import MAX_OUTPUT, Verification
from scripts.prepare_gpu_corpus import assert_disjoint

DELIMITER = b"\n\n"
BUILTIN_TRAIN_SEED = 3601
POOL_PER_CATEGORY = {c: 24 if c == "math" else 4 for c in CATEGORIES}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def distillation_partitions(seed=BUILTIN_TRAIN_SEED):
    """Short local fixture prompts that can be checked at byte-window granularity."""
    partitions = dev_records(seed)
    for group in partitions:
        for index, (row, answer) in enumerate(group):
            category = row["category"]
            if category == "math":
                continue
            value = int(row["question_id"].rsplit("-", 1)[1])
            if category == "coding":
                op = "+" if value % 2 else "*"
                prompt = f"# x{op}{value}\ndef f(x):\n"
            elif category == "reasoning":
                prompt = f"Slots {value},{value+1}. A<B.\n"
            elif category == "data_analysis":
                target = row["spec"]["format"]
                prompt = f"{target} id,value {value},{value+1}\n"
                if target == "json":
                    answer = json.dumps([{"id":value,"value":value+1}], separators=(",", ":")).encode() + DELIMITER
            else:
                prompt = f"Two words: go then {value}.\n"
            group[index] = ({**row, "turns":[prompt]}, answer)
    return partitions


def train_task_rows(fit):
    used = dict.fromkeys(CATEGORIES, 0)
    rows = []
    for row, _ in fit:
        category = row["category"]
        if used[category] < POOL_PER_CATEGORY[category]:
            rows.append(row)
            used[category] += 1
    return rows


def builtin_train_suite():
    fit, _, _ = distillation_partitions()
    return suite_from_rows(train_task_rows(fit))


@dataclass(frozen=True)
class Harvest:
    payload: bytes
    records: tuple[dict, ...]
    counts: dict
    rejected: tuple[dict, ...]
    unsupported: tuple[dict, ...]
    snapshot_sha256: str
    donor_forward_calls: int

    def report(self):
        return dict(snapshot_sha256=self.snapshot_sha256, output_sha256=_sha(self.payload),
                    output_bytes=len(self.payload), accepted=len(self.records), rejected=len(self.rejected),
                    unsupported=list(self.unsupported), counts=self.counts, donor_forward_calls=self.donor_forward_calls)


def harvest_verified_traces(engine, suite, *, candidates=16, max_bytes=MAX_OUTPUT, seed=42, temperature=0.7, top_p=0.9):
    """Use only the selected candidate's exact bytes after a second verifier call."""
    payload, records, rejected = bytearray(), [], []
    counts = {name: dict(accepted=0, rejected=0) for name in CATEGORIES}
    donor_calls = 0
    for index, task in enumerate(suite.tasks):
        def verify(raw):
            if task.mode == "arithmetic" and raw.partition(b"\n")[2].strip(b"\r\n"):
                return Verification(False, 0, 1, "arithmetic training traces cannot contain unchecked extra lines")
            return task.verify(raw)

        result = engine.run(task.prompt, verify, candidates=candidates, max_bytes=min(max_bytes, task.max_bytes),
                            stop_bytes=task.stop, seed=seed+64*index, temperature=temperature, top_p=top_p)
        donor_calls += result.donor_forward_calls
        if donor_calls:
            raise RuntimeError("verified harvesting cannot write results after a donor forward")
        selected = result.selected
        check = selected.verification
        if (check.passed is not True or check.total_tests <= 0 or check.passed_tests != check.total_tests
                or verify(selected.raw) != check):
            counts[task.category]["rejected"] += 1
            rejected.append(dict(question_id=task.id, category=task.category))
            continue
        trace = task.prompt + selected.raw + DELIMITER
        records.append(dict(question_id=task.id, category=task.category, verifier=task.mode, offset=len(payload),
                            length=len(trace), prompt_bytes=len(task.prompt), solution_bytes=len(selected.raw),
                            trace_sha256=_sha(trace), solution_sha256=_sha(selected.raw), logprob=selected.logprob,
                            passed_tests=check.passed_tests, total_tests=check.total_tests))
        payload.extend(trace)
        counts[task.category]["accepted"] += 1
    return Harvest(bytes(payload), tuple(records), counts, tuple(rejected), suite.unsupported,
                   engine.snapshot_sha256, donor_calls)


def write_harvest(harvest, out_bin, *, out_jsonl=None, evaluation=None):
    """Check held byte windows before writing immutable artifacts."""
    out_bin = Path(out_bin)
    audit = Path(out_jsonl) if out_jsonl is not None else None
    outputs = (out_bin,) if audit is None else (out_bin, audit)
    resolved = [p.resolve() for p in outputs]
    reserved = {p.with_suffix(p.suffix + ".partial") for p in resolved}
    if len(set(resolved)) != len(outputs) or reserved.intersection(resolved):
        raise ValueError("harvest output paths must be distinct from each other and from their staging paths")
    for path in outputs:
        if os.path.lexists(path) or os.path.lexists(path.with_suffix(path.suffix + ".partial")):
            raise FileExistsError(path)
        if not path.parent.is_dir():
            raise FileNotFoundError(path.parent)
    if harvest.donor_forward_calls:
        raise ValueError("cannot write a harvest with donor forwards")
    if evaluation is not None:
        assert_disjoint((harvest.payload,), {"held-out evaluation": evaluation})
    # Publish the binary last. The SHA-bound audit can only describe this exact payload.
    if audit is None:
        atomic_bytes(out_bin, harvest.payload)
        return
    digest = _sha(harvest.payload)
    body = b"".join((json.dumps(dict(record, output_sha256=digest), allow_nan=False) + "\n").encode()
                    for record in harvest.records)
    atomic_bytes(audit, body)
    published = audit.lstat()
    try:
        atomic_bytes(out_bin, harvest.payload)
    except BaseException:
        if audit.exists() and os.path.samestat(published, audit.lstat()):
            audit.unlink(missing_ok=True)
        raise
