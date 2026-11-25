"""Deterministic interpreter for a bounded subset of Python expressions."""
import ast
from functools import lru_cache
import operator

from nexus_expanse.verifier_agent import VerifierFeedback

MAX_VALUE = 2**31 - 1
MAX_LENGTH = 128
MAX_SOURCE = 512
MAX_NODES = 64
MAX_DEPTH = 16

_INTEGER_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Mod: operator.mod, ast.BitXor: operator.xor,
}
_VECTOR_OPS = {"add": operator.add, "mul": operator.mul, "mod": operator.mod, "xor": operator.xor}
# Numeric transforms act in the selected eight-value alphabet. Reverse and
# the left rotation preserve lengths so the learner can compare positions.
EXPRESSIONS = (
    b"mod(add(x, 1), 8)", b"mod(mul(x, 3), 8)", b"mod(x, 5)",
    b"reverse(x)", b"x[1:] + x[:1]", b"xor(x, 5)",
)


def _checked(value):
    if type(value) is int and -MAX_VALUE <= value <= MAX_VALUE:
        return value
    if type(value) is str and len(value) <= MAX_LENGTH:
        return value
    if type(value) is list and len(value) <= MAX_LENGTH and all(
        type(v) is int and -MAX_VALUE <= v <= MAX_VALUE for v in value
    ):
        return value
    raise ValueError("value outside the bounded integer, string or integer-list domain")


@lru_cache(maxsize=64)
def _parsed(source):
    if type(source) is not bytes or not 1 <= len(source) <= MAX_SOURCE:
        raise ValueError("expected bounded ASCII expression bytes")
    try:
        node = ast.parse(source.decode("ascii"), mode="eval")
    except (SyntaxError, UnicodeError, RecursionError) as exc:
        raise ValueError("invalid Python expression") from exc
    if sum(1 for _ in ast.walk(node)) > MAX_NODES:
        raise ValueError("expression node limit exceeded")
    return node.body


def evaluate(source, value):
    """Interpret allowlisted operations without Python compilation or evaluation."""
    if type(source) is not bytes:
        raise ValueError("expected bounded ASCII expression bytes")
    data = _checked(value)

    def walk(node, depth):
        if depth > MAX_DEPTH:
            raise ValueError("expression depth limit exceeded")

        def child(n):
            return walk(n, depth + 1)

        if isinstance(node, ast.Name) and node.id == "x":
            return data
        if isinstance(node, ast.Constant):
            return _checked(node.value)
        if isinstance(node, ast.List):
            return _checked([child(v) for v in node.elts])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            n = child(node.operand)
            if type(n) is int:
                return _checked(-n)
        if isinstance(node, ast.BinOp) and type(node.op) in _INTEGER_OPS:
            a, b = child(node.left), child(node.right)
            if type(a) is int and type(b) is int:
                return _checked(_INTEGER_OPS[type(node.op)](a, b))
            if isinstance(node.op, ast.Add) and type(a) is type(b) and type(a) in (list, str):
                return _checked(a + b)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name == "reverse" and len(node.args) == 1:
                v = child(node.args[0])
                if type(v) in (list, str):
                    return v[::-1]
            if name in _VECTOR_OPS and len(node.args) == 2:
                v, k = [child(n) for n in node.args]
                if type(k) is int:
                    f = _VECTOR_OPS[name]
                    if type(v) is int:
                        return _checked(f(v, k))
                    if type(v) is list:
                        return _checked([f(item, k) for item in v])
        if isinstance(node, ast.Subscript):
            v = child(node.value)
            if type(v) in (list, str):
                if isinstance(node.slice, ast.Slice):
                    bounds = [None if i is None else child(i)
                              for i in (node.slice.lower, node.slice.upper, node.slice.step)]
                    if all(i is None or type(i) is int for i in bounds):
                        return v[slice(*bounds)]
                else:
                    i = child(node.slice)
                    if type(i) is int:
                        return v[i]
        raise ValueError(f"unsupported expression node or operand: {type(node).__name__}")

    try:
        result = walk(_parsed(source), 0)
    except (ArithmeticError, IndexError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    return result.copy() if type(result) is list else result


def run_pipeline(source, pipeline, expressions):
    """Return exact raw-byte state after each allowed, length-preserving expression."""
    if type(source) is not bytes or not 1 <= len(source) <= MAX_LENGTH:
        raise ValueError("invalid byte input")
    if type(pipeline) is not bytes or not 1 <= len(pipeline) <= 3:
        raise ValueError("invalid byte pipeline length")
    states = [source]
    for op in pipeline:
        if op not in expressions:
            raise ValueError("unknown byte operator")
        result = evaluate(expressions[op], list(states[-1]))
        if type(result) is not list or len(result) != len(source) or any(not 0 <= b <= 255 for b in result):
            raise ValueError("pipeline expression must return a same-length byte list")
        states.append(bytes(result))
    return tuple(states)


def verify(source, target, pipeline, expressions):
    return VerifierFeedback(pipeline, target, run_pipeline(source, pipeline, expressions))
