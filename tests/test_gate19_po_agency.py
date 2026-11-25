import torch
from benchmarks.gate19_po_agency_dev import hidden_worlds, transition
from nexus_expanse.active_agent import NexusActiveAgent


def test_hidden_lock_is_an_observation_alias_with_permuted_unlock():
    regimes, actions, assignments = hidden_worlds(42)
    assert regimes[0][1] != regimes[1][1]
    for world, trials in zip(regimes, assignments, strict=True):
        table, unlock = world
        for start, goal in trials:
            assert goal not in set(table[start].values()) | {start}
            assert transition(world, start, False, unlock, goal) == (start, True)
            for action in actions:
                if action != unlock:
                    assert transition(world, start, False, action, goal) == (start, False)
            for a, mid in table[start].items():
                if goal in table[mid].values():
                    assert transition(world, start, True, a, goal) == (mid, True)
                    b = next(b for b, y in table[mid].items() if y == goal)
                    assert transition(world, mid, True, b, goal) == (goal, False)
                    break
            else:
                raise AssertionError("no valid three-step path")


def test_macro_credits_only_preceding_distinct_probe_after_motion():
    agent = NexusActiveAgent(po_mode=True)
    agent.begin_attempt()
    agent.choose(b"A", (b"x",), b"C")
    agent.observe(b"A")
    assert agent.belief.success[0][ord("x")] == 0
    saved = agent.snapshot()
    restored = NexusActiveAgent.restore(saved)
    a, p = agent.choose(b"A", (b"x", b"y"), b"C")
    b, q = restored.choose(b"A", (b"x", b"y"), b"C")
    assert a == b == b"y" and torch.equal(p, q)
    agent.observe(b"B")
    restored.observe(b"B")
    assert agent.belief.success[0][ord("x")] == 1
    assert agent.belief.success[0][ord("y")] == 0
    assert restored.snapshot() == agent.snapshot()
    assert len(saved) <= 40000


def test_default_agent_retains_gate16_snapshot_encoding():
    agent = NexusActiveAgent()
    checkpoint = agent.snapshot()
    assert checkpoint[:4] == b"NA01"
    assert NexusActiveAgent.restore(checkpoint).snapshot() == checkpoint
