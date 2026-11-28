"""Local, objective scoring over LiveBench-shaped task files."""

import csv
from dataclasses import dataclass
import json
import operator
from pathlib import Path
import random
import re
from statistics import mean
import time
from typing import Callable

from nexus_expanse.structured_verifiers import parse_table, strict_json
from nexus_expanse.verified_search import MAX_OUTPUT, MAX_PROMPT, SearchResult, Verification, make_verifier, scratchpad_verifier

VERIFIERS = {"math": "arithmetic", "coding": "python_fn", "reasoning": "logic_csp",
             "data_analysis": "table_format", "instruction_following": "instruction_constraints"}
CATEGORIES = tuple(VERIFIERS)
MAX_JSONL_LINE = 65_536
BUILTIN_SEED = 3501


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    prompt: bytes
    verify: Callable[[bytes], Verification]
    max_bytes: int
    stop: bytes
    mode: str


@dataclass(frozen=True)
class Suite:
    tasks: tuple[Task, ...]
    unsupported: tuple[dict, ...]


class UnsupportedTask(ValueError):
    pass


def _reference(answer):
    if type(answer) not in (int, str):
        raise UnsupportedTask("no scalar exact reference")
    text = str(answer).rstrip("\r\n")
    if not text or "\n" in text or "\r" in text or len(text.encode("utf-8")) > MAX_OUTPUT:
        raise UnsupportedTask("reference is not a bounded single line")
    expected = text.encode("utf-8")

    def verify(raw):
        passed = raw.rstrip(b"\r\n") == expected
        return Verification(passed, int(passed), 1, None)

    return verify


def _table_reference(row, prompt, answer):
    if type(answer) is dict and set(answer) == {"format", "columns", "rows"}:
        return answer
    if row.get("subtask", row.get("task")) != "tablereformat" or type(answer) is not str:
        raise UnsupportedTask("no exact table reference")
    target = row.get("target_format")
    if target is None:
        match = re.search(r"(?:^Target Format: *|format to +)(csv|tsv|json|markdown)\b", prompt, re.M | re.I)
        target = match[1].lower() if match else None
    if target not in ("csv", "tsv", "json", "markdown"):
        raise UnsupportedTask("uncovered tablereformat target")
    try:
        columns, rows = parse_table(answer.rstrip("\r\n"), target)
    except (ValueError, csv.Error, RecursionError) as exc:
        raise UnsupportedTask("table reference does not match the local grammar") from exc
    return dict(format=target, columns=columns, rows=rows)


def _instruction_reference(row, answer):
    if type(answer) is dict and answer:
        return answer
    ids, kwargs = row.get("instruction_id_list"), row.get("kwargs")
    if type(ids) is not list or not ids or type(kwargs) is not list or len(ids) != len(kwargs):
        raise UnsupportedTask("no structured instruction constraints")
    spec = {}
    for name, arg in zip(ids, kwargs, strict=True):
        if type(arg) is not dict:
            raise UnsupportedTask("invalid instruction arguments")
        if name == "length_constraints:number_words" and type(arg.get("num_words")) is int:
            if arg.get("relation") == "at least":
                key, value = "min_words", arg["num_words"]
            elif arg.get("relation") == "less than" and arg["num_words"] > 0:
                key, value = "max_words", arg["num_words"] - 1
            else:
                raise UnsupportedTask("uncovered word count relation")
        elif name == "detectable_format:number_bullet_lists":
            key, value = "bullet_count", arg.get("num_bullets")
        elif name == "startend:end_checker":
            key, value = "must_end_with", arg.get("end_phrase")
        else:
            raise UnsupportedTask("uncovered instruction ID")
        if key in spec:
            raise UnsupportedTask("multiple clauses require explicit local constraints")
        spec[key] = value
    return spec


def _official(row, category, prompt, raw):
    kind = VERIFIERS[category]
    answer = row.get("ground_truth")
    if category == "math" and type(answer) in (int, str):
        if raw.rstrip().endswith(b"="):
            try:
                check = make_verifier(kind, raw)
                if check(str(answer).encode()).passed:
                    return check, kind
            except ValueError:
                pass
        return _reference(answer), "exact_reference"
    if category == "coding":
        cases = row.get("tests", answer)
        if type(cases) is str and cases.lstrip().startswith("["):
            cases = strict_json(cases)
        if type(cases) is not list or not cases:
            raise UnsupportedTask("coding requires pure JSON argument/expected cases")
        return make_verifier(kind, raw, cases), kind
    if category == "reasoning":
        if type(answer) is dict and set(answer) == {"variables", "constraints"}:
            spec = answer
        elif type(answer) is dict and answer:
            spec = dict(variables={k: [v] for k, v in answer.items()},
                        constraints=[dict(type="equals", left=k, value=v) for k, v in answer.items()])
        else:
            return _reference(answer), "exact_reference"
    elif category == "data_analysis":
        spec = _table_reference(row, prompt, answer)
    elif category == "instruction_following":
        spec = _instruction_reference(row, answer)
    else:
        return make_verifier(kind, raw), kind
    return make_verifier(kind, raw, spec=spec), kind


def _task(row, key, category, *, allow_think):
    declared = row.get("verifier")
    is_local = "verifier" in row or "spec" in row
    if category not in VERIFIERS:
        raise UnsupportedTask("category outside the five local graders")
    if "prompt" in row:
        prompt = row["prompt"]
    else:
        turns = row.get("turns")
        if type(turns) is not list or len(turns) != 1:
            raise UnsupportedTask("requires exactly one prompt turn")
        prompt = turns[0]
    if type(prompt) is not str or not 1 <= len(raw := prompt.encode("utf-8")) <= MAX_PROMPT:
        raise UnsupportedTask("prompt outside the local byte limit")
    if is_local:
        kind = VERIFIERS[category]
        if declared is not None and declared != kind:
            raise ValueError("verifier conflicts with category")
        verify, mode = make_verifier(kind, raw, row.get("tests"), spec=row.get("spec")), kind
    else:
        try:
            verify, mode = _official(row, category, prompt, raw)
        except UnsupportedTask:
            raise
        except (ValueError, RecursionError) as exc:
            raise UnsupportedTask(str(exc)) from exc
    max_bytes = row.get("max_bytes", MAX_OUTPUT)
    default_stop = "\n" if mode in ("arithmetic", "logic_csp", "exact_reference") else "\n\n"
    stop = row.get("stop_text", default_stop)
    if (type(max_bytes) is not int or not 1 <= max_bytes <= MAX_OUTPUT or type(stop) is not str
            or not 1 <= len(stop.encode("utf-8")) <= 16):
        raise ValueError("invalid search limits in task row")
    if allow_think:
        verify = scratchpad_verifier(verify)
    return Task(key, category, raw, verify, max_bytes, stop.encode("utf-8"), mode)


def suite_from_rows(rows, *, allow_think=False):
    tasks, unsupported, seen = [], [], set()
    for n, row in enumerate(rows, 1):
        if type(row) is not dict:
            raise ValueError(f"row {n}: expected a task object")
        key = row.get("question_id", row.get("id"))
        if type(key) not in (str, int) or not 1 <= len(key := str(key)) <= 256 or key in seen:
            raise ValueError(f"row {n}: missing or duplicate question ID")
        seen.add(key)
        raw_category = row.get("category")
        if type(raw_category) is not str or not 1 <= len(raw_category) <= 128:
            raise ValueError(f"row {n}: missing category")
        category = raw_category.lower().strip().replace(" ", "_").replace("-", "_")
        try:
            tasks.append(_task(row, key, category, allow_think=allow_think))
        except UnsupportedTask as exc:
            unsupported.append(dict(question_id=key, category=category, reason=str(exc)))
        except (ValueError, RecursionError) as exc:
            raise ValueError(f"row {n} ({key}): {exc}") from exc
    if not seen:
        raise ValueError("empty task suite")
    return Suite(tuple(tasks), tuple(unsupported))


def load_tasks_jsonl(path, *, allow_think=False):
    def read_rows():
        with Path(path).open("rb") as source:
            line_number = 0
            while raw := source.readline(MAX_JSONL_LINE + 1):
                line_number += 1
                if len(raw) > MAX_JSONL_LINE:
                    raise ValueError(f"JSONL line {line_number} exceeds its byte limit")
                if raw.strip():
                    try:
                        yield strict_json(raw)
                    except (ValueError, RecursionError) as exc:
                        raise ValueError(f"invalid JSONL line {line_number}: {exc}") from exc
    return suite_from_rows(read_rows(), allow_think=allow_think)


def evaluate_budgets(engine, suite, budgets, *, seed=42, max_bytes=MAX_OUTPUT, temperature=0.7, top_p=0.9, allow_think=False):
    """Score nested greedy/nucleus budgets on the same raw proposals."""
    started = time.perf_counter()
    maximum = max(budgets)
    options = {"allow_think": True} if allow_think else {}
    outputs = [engine.run(t.prompt, t.verify, candidates=maximum, max_bytes=min(max_bytes, t.max_bytes),
                          stop_bytes=t.stop, seed=seed + 64*i, temperature=temperature, top_p=top_p, **options)
               for i, t in enumerate(suite.tasks)]
    scoreboards = {}
    for budget in budgets:
        counts = {c: dict(passed=0, total=0,
                          unsupported=sum(u["category"] == c for u in suite.unsupported)) for c in CATEGORIES}
        tasks, decoded, elapsed = [], 0, 0.0
        for task, result in zip(suite.tasks, outputs, strict=True):
            raced = result.candidates[:budget]
            selected = SearchResult(raced, result.donor_forward_calls).selected
            decoded += sum(len(c.raw) for c in raced)
            elapsed += sum(c.seconds for c in raced)
            group = counts[task.category]
            group["total"] += 1
            group["passed"] += selected.verification.passed
            tasks.append(dict(question_id=task.id, category=task.category, mode=task.mode,
                              selected_hex=selected.raw.hex(), logprob=selected.logprob,
                              **selected.verification._asdict()))
        scores = {c: v["passed"] / v["total"] if v["total"] else None for c, v in counts.items()}
        populated = [v for v in scores.values() if v is not None]
        scoreboards[budget] = dict(**scores, overall_score=mean(populated) if populated else None, counts=counts,
                                   graded_categories=len(populated), candidates=budget, tasks=tasks, decoded_bytes=decoded,
                                   candidate_seconds=elapsed, donor_forward_calls=sum(r.donor_forward_calls for r in outputs),
                                   unsupported=list(suite.unsupported), snapshot_sha256=engine.snapshot_sha256)
    return scoreboards, time.perf_counter() - started


def dev_records(seed=BUILTIN_SEED):
    """Generate local FIT, calibration, and CHECK tasks with disjoint IDs/prompts."""
    rng = random.Random(seed)
    groups = {c: [] for c in CATEGORIES}

    def add(category, index, prompt, answer, limit, **extra):
        row = dict(question_id=f"{category}-{index}", category=category, verifier=VERIFIERS[category],
                   turns=[prompt], max_bytes=limit, **extra)
        groups[category].append((row, answer.encode("utf-8")))

    for glyph, op in (("+", operator.add), ("-", operator.sub), ("*", operator.mul), ("%", operator.mod)):
        for a in range(1, 13):
            for b in range(1, 9):
                if glyph in ("+", "*") and a > b:
                    continue
                lhs = f"{a}{glyph}{b}"
                add("math", lhs, lhs+"=", f"{op(a,b)}\n", 8)
    for k in range(1, 25):
        glyph, op, verb = ("+", operator.add, "Add") if k % 2 else ("*", operator.mul, "Multiply by")
        add("coding", k, f"# {verb} {k}.\ndef f(x):\n    ", f"return x{glyph}{k}\n\n", 24,
            tests=[dict(args=[x], expected=op(x,k)) for x in (-9,-3,0,2,7,13)])
        positions = [k, k+1]
        add("reasoning", k, f"Place A before B in distinct slots {k},{k+1}. Use A=n,B=n.\n", f"A={k},B={k+1}\n", 22,
            spec=dict(variables={"A":positions,"B":positions}, constraints=[dict(type="all_different",variables=["A","B"]),
                      dict(type="before",left="A",right="B")]))
        output = "csv" if k % 2 else "json"
        answer = f"id,value\n{k},{k+1}\n\n" if output == "csv" else json.dumps([{"id": k, "value": k+1}]) + "\n\n"
        add("data_analysis", k, f"Output {output} with headers id,value and record {k},{k+1}.\n", answer, 48,
            spec=dict(format=output, columns=["id","value"], rows=[[k,k+1]]))
        add("instruction_following", k, f"Write exactly two words, go then {k}.\n", f"go {k}\n\n", 16,
            spec=dict(min_words=2,max_words=2,exact_lines=1,must_start_with="go ",must_end_with=f" {k}"))
    fit, val, check = [], [], []
    for rows in groups.values():
        rng.shuffle(rows)
        check.extend(rows[:6]); val.extend(rows[6:8]); fit.extend(rows[8:])
    return fit, val, check


def builtin_dev_suite(*, allow_think=False):
    return suite_from_rows((row for row, _ in dev_records()[2]), allow_think=allow_think)
