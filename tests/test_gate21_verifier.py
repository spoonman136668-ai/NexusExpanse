import itertools

import pytest

from benchmarks.gate21_verifier_synthesis_dev import World, Problem, fixture, programs_for
from nexus_expanse.verifier_agent import VerifierFeedback, VerifierGatedNexusAgent


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_fixture_minimum_depth_and_separated_phases(seed):
    world, phases, digest = fixture(seed)
    assert fixture(seed)[-1] == digest
    assert len(phases[0]) == 32 and len(phases[2]) == 16
    assert world.skins[0] != world.skins[1] and set(world.skins[0]) == set(world.skins[2])
    for phase in (phases[1], phases[3]):
        tasks = [p for p in phase if p.scored]
        assert len(tasks) == 32 and sum(p.depth == 3 for p in tasks) == 16
        for p in tasks:
            assert any(world.verify(p, pipeline).passed for pipeline in programs_for(world, p))
            assert p.source != p.target
            for n in range(1, p.depth):
                for program in itertools.product(world.skins[p.skin], repeat=n):
                    assert world.run(p.source, bytes(program), p.skin)[-1] != p.target


def test_failed_feedback_mutates_only_the_ungated_control():
    trace = VerifierFeedback(b"x", b"B", (b"A", b"C"))
    assert not trace.passed and trace.residual == b"\x01" and trace.surprise == .125
    gated, ungated = VerifierGatedNexusAgent(), VerifierGatedNexusAgent(gated=False)
    before = gated.snapshot()
    assert not gated.observe(trace) and gated.snapshot() == before
    empty = ungated.snapshot()
    assert ungated.observe(trace) and ungated.snapshot() != empty
    accepted = VerifierFeedback(b"xy", b"D", (b"A", b"B", b"D"))
    assert gated.observe(accepted) and accepted.passed
    fixed = VerifierGatedNexusAgent.restore(gated.snapshot())
    assert fixed.snapshot() == gated.snapshot()
    assert fixed.propose(b"A", b"D", [b"yx", b"xy"], set()) == b"xy"


def test_verifier_rejects_unknown_primitive_and_incorrect_length():
    w = World(42)
    p = Problem(w.alphabet, w.alphabet, 0, 2, True)
    with pytest.raises(ValueError, match="illegal"):
        w.verify(p, b"x")
    with pytest.raises(ValueError, match="unknown"):
        w.verify(p, b"xy")
