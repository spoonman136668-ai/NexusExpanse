"""Bounded test-time byte search with a pure, deterministic AST interpreter."""
from contextlib import closing
from dataclasses import dataclass
import ast
import hashlib
import math
import operator
import re
import textwrap
import time
from typing import NamedTuple

import torch

from nexus_expanse.byte_decode import engram_forecast, nucleus_byte
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.expr_verifier import MAX_DEPTH, MAX_NODES, MAX_SOURCE, MAX_VALUE
from nexus_expanse.structured_verifiers import MAX_TEXT, STRUCTURED
from nexus_expanse.unified import NexusExpanseUnified

MAX_CASES = 32
MAX_ARGS = 8
MAX_PROMPT = 4096
MAX_OUTPUT = MAX_TEXT
MAX_CASE_BYTES = 16_384
MAX_STEPS = 64
IDENTIFIER = r"[A-Za-z_][A-Za-z_0-9]*"
SIGNATURE = re.compile(rf"(?:^|\n)(def +{IDENTIFIER} *\([^\n]*\) *:)[ \t\r\n]*\Z")
INTEGER = re.compile(rb"[ \t]*([+-]?[0-9]{1,10})[ \r\t]*\Z")
MATH = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
           ast.Gt: operator.gt, ast.GtE: operator.ge}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
PURE = {"abs", "min", "max", "len"}
MATH_NODES = {ast.Constant, ast.BinOp, ast.UnaryOp, *MATH, *UNARY}
PYTHON_NODES = MATH_NODES | {ast.Name, ast.Load, ast.List, ast.Call, ast.IfExp, ast.Not, ast.Compare,
                            ast.BoolOp, ast.And, ast.Or, ast.Subscript, *COMPARE}


class Verification(NamedTuple):
    passed: bool
    passed_tests: int
    total_tests: int
    error: str | None


def _int(value):
    if type(value) is not int or abs(value) > MAX_VALUE:
        raise ValueError("integer outside the verifier bounds")
    return value


def _value(value, depth=0):
    if depth > 3:
        raise ValueError("test value nesting too deep")
    if type(value) is int:
        return _int(value)
    if value is None or type(value) is bool or (type(value) is str and len(value) <= 128):
        return value
    if type(value) is list and len(value) <= 32:
        return [_value(item, depth + 1) for item in value]
    raise ValueError("expected a bounded JSON integer, boolean, null, string, or list")


def _equal(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is list:
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def _parse(raw, mode):
    if not 1 <= len(raw) <= MAX_SOURCE or not raw.isascii():
        raise ValueError("expected a bounded ASCII expression or definition")
    try:
        tree = ast.parse(raw, mode=mode)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError("invalid verifier syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ValueError("verifier AST node limit exceeded")
    return tree


def _validate(node, names, arithmetic, depth=0):
    allowed = MATH_NODES if arithmetic else PYTHON_NODES
    kind = type(node)
    if depth > MAX_DEPTH or kind not in allowed:
        raise ValueError(f"unsupported verifier syntax: {kind.__name__}")
    if kind is ast.Call:
        if (type(node.func) is not ast.Name or node.func.id not in PURE or node.func.id in names
                or node.keywords or not 1 <= len(node.args) <= MAX_ARGS):
            raise ValueError("unsupported verifier call")
        children = node.args
    else:
        children = ast.iter_child_nodes(node)
    if kind is ast.Name and node.id not in names:
        raise ValueError("unknown verifier name")
    if kind is ast.Constant:
        (_int if arithmetic else _value)(node.value)
    for child in children:
        _validate(child, names, arithmetic, depth + 1)


def _evaluate(root, bindings):
    remaining = MAX_STEPS

    def visit(node):
        nonlocal remaining
        remaining -= 1
        if remaining < 0:
            raise ValueError("verifier evaluation step limit exceeded")
        if type(node) is ast.Constant:
            return node.value
        if type(node) is ast.Name:
            return bindings[node.id]
        if type(node) is ast.BinOp:
            return _int(MATH[type(node.op)](_int(visit(node.left)), _int(visit(node.right))))
        if type(node) is ast.UnaryOp:
            value = visit(node.operand)
            return not value if type(node.op) is ast.Not else _int(UNARY[type(node.op)](_int(value)))
        if type(node) is ast.List:
            return _value([visit(item) for item in node.elts])
        if type(node) is ast.Subscript:
            value = visit(node.value)
            if type(value) not in (str, list):
                raise ValueError("indexing requires a string or list")
            return value[_int(visit(node.slice))]
        if type(node) is ast.Compare:
            left = visit(node.left)
            for op, expression in zip(node.ops, node.comparators, strict=True):
                right = visit(expression)
                if not COMPARE[type(op)](left, right):
                    return False
                left = right
            return True
        if type(node) is ast.IfExp:
            return visit(node.body if visit(node.test) else node.orelse)
        if type(node) is ast.BoolOp:
            for expression in node.values:
                value = visit(expression)
                if bool(value) == (type(node.op) is ast.Or):
                    return value
            return value
        if type(node) is ast.Call:
            args = [visit(arg) for arg in node.args]
            name = node.func.id
            if name == "len" and len(args) == 1 and type(args[0]) in (str, list):
                return len(args[0])
            if name == "abs" and len(args) == 1:
                return _int(abs(_int(args[0])))
            if name in ("min", "max") and len(args) >= 2:
                return (min if name == "min" else max)(map(_int, args))
            raise ValueError("unsupported pure-call arguments")
        raise ValueError("unsupported verifier node")

    return _value(visit(root))


def _function(code):
    statements = _parse(code, "exec").body  # Parse to AST only. Nothing is compiled or run.
    if len(statements) != 1 or type(statements[0]) is not ast.FunctionDef:
        raise ValueError("expected one plain function")
    fn = statements[0]
    args = fn.args
    if (fn.decorator_list or fn.returns or fn.type_comment or getattr(fn, "type_params", ())
            or args.defaults or args.kwonlyargs or args.posonlyargs or args.vararg or args.kwarg
            or len(args.args) > MAX_ARGS or any(a.annotation or a.type_comment or "__" in a.arg for a in args.args)
            or "__" in fn.name or len(fn.body) != 1 or type(fn.body[0]) is not ast.Return or fn.body[0].value is None):
        raise ValueError("expected a plain function with one return expression")
    names = tuple(a.arg for a in args.args)
    if len(set(names)) != len(names):
        raise ValueError("duplicate verifier parameter")
    return (fn.name, names), fn.body[0].value


def _cases(rows, arity):
    if type(rows) is not list or not 1 <= len(rows) <= MAX_CASES:
        raise ValueError(f"expected 1..{MAX_CASES} verifier test cases")
    verified = []
    for row in rows:
        if type(row) is not dict or set(row) != {"args", "expected"} or type(row["args"]) is not list or len(row["args"]) != arity:
            raise ValueError("each test requires args matching the signature and an expected value")
        verified.append((tuple(_value(value) for value in row["args"]), _value(row["expected"])))
    return tuple(verified)


def make_verifier(kind, prompt, tests=None, *, spec=None):
    """Bind the bounded task once and return a pure continuation verifier."""
    if type(prompt) is not bytes or not 1 <= len(prompt) <= MAX_PROMPT:
        raise ValueError("expected a bounded raw-byte prompt")
    if kind in STRUCTURED:
        if tests is not None:
            raise ValueError("structured verifiers require a spec rather than test cases")
        check = STRUCTURED[kind](spec)
        return lambda candidate: Verification(*check(candidate))
    if spec is not None:
        raise ValueError("arithmetic and Python verifiers do not use a structured spec")
    if kind == "arithmetic":
        line = prompt.rstrip().rsplit(b"\n", 1)[-1]
        if not line.endswith(b"=") or tests is not None:
            raise ValueError("arithmetic requires a final equation and no test cases")
        expression = _parse(line[:-1].strip(), "eval").body
        _validate(expression, (), True)
        try:
            expected = _evaluate(expression, {})
        except ArithmeticError as exc:
            raise ValueError("invalid arithmetic equation") from exc

        def verify(candidate):
            match = INTEGER.fullmatch(candidate.split(b"\n", 1)[0])
            passed = match is not None and int(match[1]) == expected
            return Verification(passed, int(passed), 1, None if match else "expected one integer on the first line")

        return verify
    if kind not in ("python_fn", "python_expr"):
        raise ValueError("unsupported verifier")
    match = SIGNATURE.search(prompt.decode("ascii", errors="replace"))
    if not match:
        raise ValueError("Python verifier prompt must end in a plain def signature")
    header = match[1]
    signature, _ = _function(header + "\n    return None")
    names = signature[1]
    cases = _cases(tests, len(names))

    def verify(candidate):
        total = len(cases)
        try:
            if not 1 <= len(candidate) <= MAX_OUTPUT:
                raise ValueError("expected a bounded byte continuation")
            code = candidate.decode("ascii")
            if kind == "python_expr":
                expression = _parse(code.strip(), "eval").body
            else:
                code = textwrap.dedent(code).strip()
                source = code if code.startswith("def ") else header + "\n" + textwrap.indent(code, "    ")
                actual, expression = _function(source)
                if actual != signature:
                    raise ValueError("candidate signature does not match the task")
            _validate(expression, names, False)
        except (UnicodeError, ValueError) as exc:
            return Verification(False, 0, total, str(exc))
        passed, error = 0, None
        for inputs, expected in cases:
            try:
                passed += _equal(_evaluate(expression, dict(zip(names, inputs, strict=True))), expected)
            except (ArithmeticError, IndexError, TypeError, ValueError) as exc:
                if error is None:
                    error = str(exc)
        return Verification(passed == total, passed, total, error)

    return verify


@dataclass(frozen=True)
class ByteCandidate:
    raw: bytes
    logprob: float
    verification: Verification
    seconds: float


@dataclass(frozen=True)
class SearchResult:
    candidates: tuple[ByteCandidate, ...]
    donor_forward_calls: int

    @property
    def selected(self):
        return max(self.candidates, key=lambda c: (c.verification.passed, c.logprob))


class VerifiedByteSearch:
    """Greedy then seeded nucleus proposals from independently restored cursors."""

    def __init__(self, snapshot, *, expected_sha256=None, device="cpu", always_wake=False,
                 engram=None, engram_alpha=0.35):
        if type(snapshot) is not bytes or not 1 <= len(snapshot) <= NexusExpanseUnified.SNAPSHOT_LIMIT:
            raise ValueError("expected a bounded unified snapshot")
        self.snapshot_sha256 = hashlib.sha256(snapshot).hexdigest()
        if expected_sha256 is not None and expected_sha256.lower() != self.snapshot_sha256:
            raise ValueError("snapshot SHA-256 mismatch")
        if not math.isfinite(engram_alpha) or not 0 <= engram_alpha <= 0.9:
            raise ValueError("engram alpha must be finite and in [0.0, 0.9]")
        self.snapshot = snapshot
        self.device = device
        self.always_wake = always_wake
        self.engram = engram
        self.engram_alpha = engram_alpha

    def _forecasts(self, prompt):
        model = NexusExpanseUnified.restore(self.snapshot, fast_decode=True, decode_device=self.device)
        if self.always_wake:
            model.reader.minimum_probability = 2.0
        forecast = model.prefill(prompt, adapt=False)
        store = self.engram
        suffix = prompt[-store.max_order:] if store is not None else b""
        while True:
            probabilities = forecast.mixed
            if store is not None:
                probabilities, _ = engram_forecast(probabilities, suffix, store, self.engram_alpha)
            byte = yield probabilities
            forecast = model.step(byte, adapt=False)
            if store is not None:
                suffix = (suffix + bytes((byte,)))[-store.max_order:]

    def forecast_trace(self, prompt, observed):
        """Score after prefill and each supplied byte, for causal checks."""
        if type(prompt) is not bytes or not 1 <= len(prompt) <= MAX_PROMPT:
            raise ValueError("expected a bounded raw-byte prompt")
        if type(observed) is not bytes or len(observed) > MAX_OUTPUT:
            raise ValueError("expected a bounded observed continuation")
        with forbid_donor_forward(), closing(self._forecasts(prompt)) as trace:
            return torch.stack([next(trace), *(trace.send(byte) for byte in observed)])

    def run(self, prompt, verifier, *, candidates=16, max_bytes=128, temperature=0.7, top_p=0.9,
            seed=42, stop_bytes=b"\n"):
        if type(prompt) is not bytes or not 1 <= len(prompt) <= MAX_PROMPT:
            raise ValueError("expected a bounded raw-byte prompt")
        if (type(candidates) is not int or not 1 <= candidates <= 64 or type(max_bytes) is not int
                or not 1 <= max_bytes <= MAX_OUTPUT or type(seed) is not int
                or type(stop_bytes) is not bytes or not 1 <= len(stop_bytes) <= 16):
            raise ValueError("invalid byte search bounds")
        if (not math.isfinite(temperature) or temperature < 0 or (candidates > 1 and temperature == 0)
                or not math.isfinite(top_p) or not 0 < top_p <= 1):
            raise ValueError("sampling requires positive temperature and top-p in (0, 1]")
        rows = []
        with forbid_donor_forward() as guard:
            for k in range(candidates):
                start = time.perf_counter()
                generator = torch.Generator().manual_seed((seed + k) % 2**64) if k else None
                raw, logprob = self._proposal(prompt, max_bytes, stop_bytes, temperature, top_p, generator)
                verification = verifier(raw)
                rows.append(ByteCandidate(raw, logprob, verification, time.perf_counter() - start))
        return SearchResult(tuple(rows), guard.donor_forward_calls)

    def _proposal(self, prompt, length, stop_bytes, temperature, top_p, generator):
        output = bytearray()
        logprob = 0.0
        with closing(self._forecasts(prompt)) as trace:
            probs = next(trace)
            for pos in range(length):
                byte = (int(probs.argmax()) if generator is None
                        else nucleus_byte(probs.log(), temperature, top_p, generator))
                logprob += math.log(float(probs[byte]))
                output.append(byte)
                if output.endswith(stop_bytes) or pos + 1 == length:
                    break
                probs = trace.send(byte)
        return bytes(output), logprob
