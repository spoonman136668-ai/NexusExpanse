import itertools

import pytest

from benchmarks.gate23_real_verifier_dev import fixture, visible_trace
from nexus_expanse.expr_verifier import evaluate, verify
from nexus_expanse.verifier_agent import VerifierFeedback, VerifierGatedNexusAgent


def test_interpreter_evaluates_integers_strings_and_lists():
    assert evaluate(b"(x * 3 + 1) % 8", 5) == 0
    assert evaluate(b"reverse(x)[1:] + x[:1]", "abc") == "baa"
    assert evaluate(b"x[1:] + x[:1]", [1, 2, 3]) == [2, 3, 1]
    assert evaluate(b"mod(add(x, 3), 8)", [1, 6]) == [4, 1]


@pytest.mark.parametrize("source", [b"unknown(x)", b"x.name", b"[i for i in x]", b"x ** 2",
                                   b"x[::-0]", b"2147483647 + 1", b"x" * 513])
def test_interpreter_rejects_outside_its_bounded_expression_subset(source):
    with pytest.raises(ValueError):
        evaluate(source, [1, 2])


def test_corruption_preserves_true_final_and_gates_admission():
    wrong = VerifierFeedback(b"ab", b"xyz", (b"abc", b"acb", b"bac"))
    cases = [visible_trace(wrong, 42, 1, i) for i in range(128)]
    assert any(c.states != wrong.states for c in cases) and any(c.states == wrong.states for c in cases)
    poisoned = next(c for c in cases if c.states != wrong.states)
    assert poisoned.states[0] == wrong.states[0] and poisoned.states[-1] == wrong.states[-1]
    assert poisoned.residual == wrong.residual
    gated, all_attempts = VerifierGatedNexusAgent(structured=True), VerifierGatedNexusAgent(gated=False, structured=True)
    assert not gated.observe(poisoned) and all_attempts.observe(poisoned)
    expected = verify(b"\x01\x02\x03", b"\x03\x02\x01", b"r", {ord("r"): b"reverse(x)"})
    assert gated.observe(expected) and visible_trace(expected, 42, 1, 3) == expected
    loaded = VerifierGatedNexusAgent.restore(gated.snapshot())
    assert loaded.snapshot() == gated.snapshot()
    assert loaded.propose(expected.states[0], expected.target, (b"q", b"r"), set()) == b"r"


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_scored_fixture_has_exact_held_out_depth(seed):
    world, phases, sha = fixture(seed)
    assert fixture(seed)[-1] == sha
    assert len(phases[0]) == 48
    for phase in phases[1:]:
        assert len(phase) == 32 and sum(p.depth == 3 for p in phase) == 16
        for p in phase:
            ops = world.skins[0]
            assert any(world.verify(p, bytes(seq)).passed for seq in itertools.product(ops, repeat=p.depth))
            for n in range(1, p.depth):
                assert all(world.run(p.source, bytes(seq), 0)[-1] != p.target
                           for seq in itertools.product(ops, repeat=n))
    assert {p.source for p in phases[1]}.isdisjoint(p.source for p in phases[2])
