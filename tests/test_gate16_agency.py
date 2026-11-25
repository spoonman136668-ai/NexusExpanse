import pytest
import torch

from benchmarks.gate16_agency_dev import ARMS, FRESH, PERSIST, decision, fixture_hash, worlds
from nexus_expanse.active_agent import NexusActiveAgent


@pytest.mark.parametrize("seed", (42, 43, 44))
def test_worlds_match_and_require_two_consequential_actions(seed):
    fixture = worlds(seed)
    assert fixture_hash(fixture) == fixture_hash(worlds(seed))
    tables, actions, groups = fixture
    assert len(set(pair for phase in groups for pair in phase)) == 72
    for table, phases in zip(tables, groups, strict=True):
        for start, goal in phases:
            assert start != goal and goal not in table[start].values()
            assert any(goal in table[mid].values() for mid in table[start].values())
    assert all(tables[0][state][a] != tables[1][state][a] for state in tables[0] for a in actions)


def test_agent_learns_only_after_observation_and_restores():
    agent = NexusActiveAgent()
    initial = agent.snapshot()
    with pytest.raises(ValueError, match="choose"):
        agent.observe(b"C")
    a, forecast = agent.choose(b"A", (b"x", b"y"), b"C")
    with pytest.raises(ValueError, match="completed"):
        agent.snapshot()
    with pytest.raises(ValueError, match="outstanding"):
        agent.choose(b"A", (b"x", b"y"), b"C")
    assert a == b"x"
    assert forecast[67] == 1 / 256
    agent.observe(b"C")
    saved = agent.snapshot()
    assert saved != initial and len(saved) <= 40000
    restored = NexusActiveAgent.restore(saved)
    assert restored.snapshot() == saved
    a, before = agent.choose(b"A", (b"x", b"y"), b"C")
    b, after = restored.choose(b"A", (b"x", b"y"), b"C")
    assert a == b and torch.equal(before, after)


def test_agent_surprise_switch_keeps_prior_bank():
    agent = NexusActiveAgent()
    for _ in range(6):
        agent.choose(b"A", (b"x",), b"B")
        assert agent.observe(b"B") == (False, False)
    first = agent.banks[0].snapshot()
    agent.choose(b"A", (b"x",), b"B")
    assert agent.observe(b"C") == (True, False)
    resumed = NexusActiveAgent.restore(agent.snapshot())
    resumed.choose(b"A", (b"x",), b"B")
    assert resumed.observe(b"C") == (True, True)
    assert resumed.active == 1
    assert resumed.banks[0].snapshot() == first


def test_decision_needs_both_totals_and_cold_return():
    r = {a: [dict(completions=1, restore={"equal": True}, peak_serialized_bytes=0)
             for _ in range(3)] for a in ARMS}
    for phase in r[PERSIST]:
        phase["completions"] = 2
    run = dict(seed=42, arms=r)
    assert decision([run])["status"] == "PASS"
    r[FRESH][-1]["completions"] = 2
    assert decision([run])["status"] == "MIXED"
    r[PERSIST][-1]["restore"]["equal"] = False
    assert decision([run])["status"] == "FAIL"
