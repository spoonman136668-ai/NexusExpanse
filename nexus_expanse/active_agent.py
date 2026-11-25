"""Causal raw-byte planning for a bounded, offline interactive DEV world."""
import math
import struct


from nexus_expanse.arbitration import ReliabilityArbitrator
from nexus_expanse.consolidation import ConsolidatingMemory
from nexus_expanse.hopfield_vsa import HopfieldVSAMemory


def raw_symbol(value):
    if not isinstance(value, bytes) or len(value) != 1:
        raise ValueError("expected one opaque raw byte")
    return int.from_bytes(value)


def transition_support(memory, observation, action):
    # Use the stored count for the complete context, without suffix backoff.
    return memory._totals.get(observation + action, 0.0)


def exploration_bonus(memory, observation, action):
    return .30 / (1.0 + transition_support(memory, observation, action))


class _TransitionVSA(HopfieldVSAMemory):
    # Two banks, each with at most 44 bound contexts and a 6,000-byte
    # episodic store, fit the single 40,000-byte serialized budget.
    SLOTS = 44


class _TransitionBank:
    LIMIT = 6000
    META = struct.Struct("<IIdQ")

    def __init__(self):
        self.episodic = ConsolidatingMemory(self.LIMIT, "sparse_support", 2)
        self.relations = _TransitionVSA(None)
        self.reliability = ReliabilityArbitrator()

    def experts(self, observation, action):
        context = observation + action
        pm, confidence = self.episodic.predict_probs(context)
        pv = self.relations.predict(tuple(context), pm)
        return pm, pv, confidence

    def predict(self, observation, action):
        pm, pv, confidence = self.experts(observation, action)
        w = self.reliability.weight(confidence)
        return (1.0 - w) * pm + w * pv

    def learn(self, observation, action, successor):
        pm, pv, confidence = self.experts(observation, action)
        target = raw_symbol(successor)
        self.reliability.observe(confidence, float(pm[target]), float(pv[target]))
        self.episodic.observe_transition(observation + action, target)
        self.relations.observe(tuple(observation + action), target)

    def snapshot(self):
        episodic = self.episodic.snapshot(max_bytes=self.LIMIT)
        relational = self.relations.snapshot()
        return self.META.pack(len(episodic), len(relational), self.reliability.weight(0.0),
                              self.reliability.updates) + episodic + relational

    @classmethod
    def read(cls, payload, start):
        n, m, weight, updates = cls.META.unpack_from(payload, start)
        start += cls.META.size
        end = start + n + m
        if end > len(payload) or not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("invalid transition memory snapshot")
        result = cls()
        result.episodic = ConsolidatingMemory.restore(payload[start:start + n], max_bytes=cls.LIMIT)
        result.relations.restore(payload[start + n:end])
        result.reliability._weights[0] = weight
        result.reliability.updates = updates
        return result, end


class NexusActiveAgent:
    """Plan two transitions, then learn from the actual observed successor.

    The observation, candidate actions and goal arrive as independent opaque
    raw bytes. The agent has no transition table, phase marker or world labels.
    """

    BUDGET = 40_000
    SUPPORT = 2
    SURPRISE = .20
    SWITCH_MATCH = .50
    SWITCH_COUNT = 2

    def __init__(self):
        self.banks = [_TransitionBank(), _TransitionBank()]
        self.active = 0
        self.held = []
        self._pending = None

    def choose(self, observation, candidates, goal):
        if self._pending is not None:
            raise ValueError("observe the outstanding action before choosing again")
        raw_symbol(observation)
        target = raw_symbol(goal)
        candidates = tuple(candidates)
        for action in candidates:
            raw_symbol(action)
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("candidate actions must be nonempty and distinct")
        bank = self.banks[self.active]
        futures = {
            i: max(float(bank.predict(bytes((i,)), action)[target]) for action in candidates)
            for i in sorted(bank.relations.outcomes) if i != target
        }
        distributions = {action: bank.predict(observation, action) for action in candidates}
        scores = {}
        for action, probs in distributions.items():
            lookahead = sum(float(probs[i]) * q for i, q in futures.items())
            scores[action] = float(probs[target]) + .90 * lookahead + exploration_bonus(
                bank.episodic, observation, action)
        selected = max(scores, key=scores.get)
        other = self.banks[1 - self.active].predict(observation, selected)
        predicted = distributions[selected]
        self._pending = observation, selected, predicted, other
        return selected, predicted.clone()

    def observe(self, successor):
        target = raw_symbol(successor)
        if self._pending is None:
            raise ValueError("choose an action before observing a successor")
        observation, action, expected, alternative = self._pending
        self._pending = None
        bank, other = self.banks[self.active], self.banks[1 - self.active]
        surprise = transition_support(bank.episodic, observation, action) >= self.SUPPORT and (
            float(expected[target]) < self.SURPRISE)
        if not surprise:
            bank.learn(observation, action, successor)
            return False, False

        self.held.append((observation, action, successor))
        corroborated = transition_support(other.episodic, observation, action) >= self.SUPPORT and (
            float(alternative[target]) >= self.SWITCH_MATCH)
        if corroborated or len(self.held) == self.SWITCH_COUNT:
            self.active = 1 - self.active
            for seen, tried, outcome in self.held:
                other.learn(seen, tried, outcome)
            self.held.clear()
            return True, True
        return True, False

    def snapshot(self):
        if self._pending is not None:
            raise ValueError("checkpoint requires a completed action observation")
        payload = b"NA01" + bytes((self.active, len(self.held)))
        payload += b"".join(b"".join(triple) for triple in self.held)
        payload += b"".join(bank.snapshot() for bank in self.banks)
        if len(payload) > self.BUDGET:
            raise ValueError("adaptive state exceeds the serialized limit")
        return payload

    @classmethod
    def restore(cls, payload):
        if not 6 <= len(payload) <= cls.BUDGET or payload[:4] != b"NA01":
            raise ValueError("invalid active agent snapshot")
        active, held_count = payload[4:6]
        if active not in (0, 1) or held_count >= cls.SWITCH_COUNT:
            raise ValueError("invalid active bank or surprise evidence")
        result = cls()
        result.active = active
        cursor = 6
        for _ in range(held_count):
            result.held.append(tuple(bytes((i,)) for i in payload[cursor:cursor + 3]))
            cursor += 3
        try:
            for i in range(2):
                result.banks[i], cursor = _TransitionBank.read(payload, cursor)
        except struct.error as exc:
            raise ValueError("truncated active agent snapshot") from exc
        if cursor != len(payload):
            raise ValueError("unexpected active agent snapshot suffix")
        return result
