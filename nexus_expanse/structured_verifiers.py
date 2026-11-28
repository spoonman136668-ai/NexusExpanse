"""Pure bounded verifiers for assignment, table, and instruction responses."""

import csv
import io
import json
import math
import re

MAX_TEXT = 512
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,31}\Z")
MARKDOWN_RULE = re.compile(r":?-{3,}:?\Z")
BULLET = re.compile(r"^[ \t]*[-*+] +\S")


def _delimited_grammar(separator):
    field = rf'(?:"(?:[^"]|"")*"|[^"{separator}\r\n]*)'
    row = rf"{field}(?:{separator}{field})*"
    return re.compile(rf"{row}(?:\r?\n{row})*")


DELIMITED = {"csv": (",", _delimited_grammar(",")), "tsv": ("\t", _delimited_grammar("\t"))}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def strict_json(raw):
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=_invalid_constant)


def _text(raw):
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_TEXT:
        raise ValueError("expected a bounded byte continuation")
    text = raw.decode("utf-8").replace("\r\n", "\n")
    if any((ord(ch) < 32 and ch not in "\n\t") or ord(ch) == 127 for ch in text):
        raise ValueError("unexpected control character")
    text = text.rstrip("\n")
    if not text.strip():
        raise ValueError("empty continuation")
    return text


def _check_report(checks):
    passed = sum(checks)
    return passed == len(checks), passed, len(checks), None


def _scalar(value):
    if type(value) is str and len(value) <= 256:
        return value
    if value is None or type(value) is bool or (type(value) is int and abs(value) <= 2**53):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("expected a bounded scalar value")


def _typed(value):
    return type(value), _scalar(value)


def _plain(value):
    return value if type(value) is str else json.dumps(value, ensure_ascii=False, allow_nan=False)


def logic_csp(spec):
    if type(spec) is not dict or set(spec) != {"variables", "constraints"}:
        raise ValueError("CSP requires variables and constraints")
    source, clauses = spec["variables"], spec["constraints"]
    if type(source) is not dict or not 1 <= len(source) <= 32 or type(clauses) is not list or not 1 <= len(clauses) <= 64:
        raise ValueError("CSP requires 1..32 variables and 1..64 constraints")
    domains, literals = {}, {}
    for key, values in source.items():
        if type(key) is not str or not NAME.fullmatch(key) or type(values) is not list or not 1 <= len(values) <= 32:
            raise ValueError("invalid CSP variable or domain")
        domains[key] = {_typed(x) for x in values}
        if len(domains[key]) != len(values):
            raise ValueError("duplicate CSP domain value")
        literals[key] = {}
        for value in values:
            literals[key].setdefault(_plain(value), []).append(value)

    def variable(name):
        if type(name) is not str or name not in domains:
            raise ValueError("unknown CSP variable")
        return name

    remaining = 64

    def prepare(clause, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 4 or type(clause) is not dict:
            raise ValueError("CSP constraint bounds exceeded")
        op = clause.get("type")
        keys = set(clause)
        if op == "all_different" and keys == {"type", "variables"}:
            group = clause["variables"]
            if type(group) is not list or not 2 <= len(group) <= len(domains):
                raise ValueError("all_different requires distinct variables")
            names = tuple(map(variable, group))
            if len(set(names)) != len(names):
                raise ValueError("all_different requires distinct variables")
            return lambda assignment: len({_typed(assignment[n]) for n in names}) == len(names)
        if op == "implies" and keys == {"type", "if", "then"}:
            premise, consequence = (prepare(clause[key], depth + 1) for key in ("if", "then"))
            return lambda assignment: not premise(assignment) or consequence(assignment)
        if op in ("equals", "not_equals") and keys in ({"type", "left", "right"}, {"type", "left", "value"}):
            left = variable(clause["left"])
            if "right" in clause:
                right = variable(clause["right"])
                def compare(assignment):
                    return _typed(assignment[left]) == _typed(assignment[right])
            else:
                value = _typed(clause["value"])
                def compare(assignment):
                    return _typed(assignment[left]) == value
            return lambda assignment: compare(assignment) == (op == "equals")
        if op in ("before", "adjacent") and keys == {"type", "left", "right"}:
            left, right = variable(clause["left"]), variable(clause["right"])
            if any(kind is not int for kind, _ in domains[left] | domains[right]):
                raise ValueError("ordering constraints require integer positions")
            if op == "before":
                return lambda assignment: assignment[left] < assignment[right]
            return lambda assignment: abs(assignment[left] - assignment[right]) == 1
        raise ValueError("unsupported or malformed CSP constraint")

    checks = tuple(map(prepare, clauses))

    def verify(raw):
        try:
            text = _text(raw).strip()
            if text.startswith("{"):
                assignment = strict_json(text)
            else:
                assignment = {}
                for item in re.split(r"[,\n]", text):
                    key, sep, token = item.partition("=")
                    key, token = key.strip(), token.strip()
                    options = literals.get(key, {}).get(token, ())
                    if not sep or key in assignment or len(options) != 1:
                        raise ValueError("invalid or ambiguous assignment token")
                    assignment[key] = options[0]
            if (type(assignment) is not dict or set(assignment) != set(domains)
                    or any(_typed(value) not in domains[key] for key, value in assignment.items())):
                raise ValueError("assignment must supply exactly one in-domain value per variable")
        except (ValueError, RecursionError) as exc:
            return False, 0, len(checks), str(exc)
        return _check_report([check(assignment) for check in checks])

    return verify


def _markdown_row(line):
    body = line.strip()
    if not body or "|" not in body:
        raise ValueError("expected a Markdown table row")
    raw = re.split(r"(?<!\\)\|", body)
    if raw[0] == "":
        raw.pop(0)
    if raw and raw[-1] == "":
        raw.pop()
    return [v.strip().replace(r"\|", "|") for v in raw]


def parse_table(text, target):
    """Read headers and typed rows for the four supported table formats."""
    if target == "json":
        data = strict_json(text)
        if type(data) is list and data and all(type(row) is dict for row in data):
            columns = list(data[0])
            if any(list(row) != columns for row in data):
                raise ValueError("inconsistent JSON table columns")
            return columns, [list(row.values()) for row in data]
        if type(data) is dict and set(data) == {"columns", "rows"}:
            return data["columns"], data["rows"]
        raise ValueError("expected a JSON table")
    if target in DELIMITED:
        separator, grammar = DELIMITED[target]
        if not grammar.fullmatch(text):
            raise ValueError("invalid delimited table quoting")
        rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=separator, strict=True))
        if not rows:
            raise ValueError("missing table headers")
        return rows[0], rows[1:]
    if target == "markdown":
        headings, *body = [_markdown_row(line) for line in text.split("\n")]
        if not body or len(body[0]) != len(headings) or not all(MARKDOWN_RULE.fullmatch(rule) for rule in body[0]):
            raise ValueError("missing or malformed Markdown delimiter row")
        return headings, body[1:]
    raise ValueError("unknown table format")


def _shape(columns, rows):
    if (type(columns) is not list or not 1 <= len(columns) <= 32
            or any(type(c) is not str or not c or len(c) > 128 for c in columns) or len(set(columns)) != len(columns)
            or type(rows) is not list or len(rows) > 128
            or any(type(row) is not list or len(row) != len(columns) for row in rows)):
        raise ValueError("invalid table shape")
    return tuple(columns), tuple(tuple(_scalar(value) for value in row) for row in rows)


def table_format(spec):
    if type(spec) is not dict or set(spec) != {"format", "columns", "rows"}:
        raise ValueError("table requires format, columns, and rows")
    target = spec["format"]
    if type(target) is not str or target not in ("json", "csv", "tsv", "markdown"):
        raise ValueError("unknown table format")
    headers, expected = _shape(spec["columns"], spec["rows"])
    convert = _typed if target == "json" else _plain
    expected = tuple(tuple(map(convert, row)) for row in expected)

    def verify(raw):
        try:
            columns, rows = _shape(*parse_table(_text(raw), target))
            if target == "json":
                rows = tuple(tuple(map(_typed, row)) for row in rows)
        except (ValueError, csv.Error, RecursionError) as exc:
            return False, 0, len(expected) + 2, str(exc)
        checks = [headers == columns, len(rows) == len(expected)]
        checks.extend(i < len(rows) and row == rows[i] for i, row in enumerate(expected))
        return _check_report(checks)

    return verify


def instruction_constraints(spec):
    integers = {"min_words", "max_words", "exact_lines", "bullet_count"}
    strings = {"must_start_with", "must_end_with"}
    lists = {"required_substrings", "forbidden_substrings"}
    if type(spec) is not dict or not spec or set(spec) - (integers | strings | lists):
        raise ValueError("expected supported instruction constraints")
    checks = []
    for key, value in spec.items():
        if key in integers:
            if type(value) is not int or not 0 <= value <= MAX_TEXT:
                raise ValueError("expected a bounded nonnegative instruction count")
            checks.append((key, value))
        elif key in strings:
            if type(value) is not str or not 1 <= len(value) <= MAX_TEXT:
                raise ValueError("expected a nonempty instruction string")
            checks.append((key, value))
        else:
            if type(value) is not list or not 1 <= len(value) <= 32 or any(type(v) is not str or not v or len(v) > MAX_TEXT for v in value):
                raise ValueError("expected bounded nonempty instruction substrings")
            checks.extend((key, v) for v in value)
    if spec.get("min_words", 0) > spec.get("max_words", MAX_TEXT):
        raise ValueError("minimum word count exceeds maximum")

    def verify(raw):
        try:
            text = _text(raw)
        except ValueError as exc:
            return False, 0, len(checks), str(exc)
        lines = text.split("\n")
        words = len(text.split())
        bullets = sum(bool(BULLET.match(line)) for line in lines)
        predicates = {"min_words": lambda n: words >= n, "max_words": lambda n: words <= n,
                      "exact_lines": lambda n: len(lines) == n, "bullet_count": lambda n: bullets == n,
                      "must_start_with": text.startswith, "must_end_with": text.endswith,
                      "required_substrings": lambda s: s in text, "forbidden_substrings": lambda s: s not in text}
        return _check_report([predicates[key](value) for key, value in checks])

    return verify


STRUCTURED = {"logic_csp": logic_csp, "table_format": table_format, "instruction_constraints": instruction_constraints}
