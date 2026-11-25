"""Small two-mode filter for opaque, aliased action outcomes."""
import math
import struct


class LockBelief:
    """Track movement availability and retrospectively credit the previous probe."""

    HEADER = struct.Struct("<dH32s")

    def __init__(self):
        self.success = [bytearray(256), bytearray(256)]
        self.failure = [bytearray(256), bytearray(256)]
        self.reset()

    def reset(self):
        self.open_probability = .05
        self.probe = None
        self.tried = set()

    def toggle(self, bank, action):
        yes, no = self.success[bank][action], self.failure[bank][action]
        return (yes + 1) / (yes + no + 2)

    def options(self, bank, candidates):
        available = [a for a in candidates if a[0] not in self.tried] or list(candidates)
        if self.open_probability < .35:
            def exploration(action):
                a = action[0]
                n = self.success[bank][a] + self.failure[bank][a]
                return self.toggle(bank, a) + .35 / (n + 1)
            return (max(available, key=exploration),)
        if self.probe is not None:
            available = [a for a in available if a[0] != self.probe] or available
        # A known mode switch is a poor movement action when already open.
        return tuple(a for a in available if self.toggle(bank, a[0]) < .67) or tuple(available)

    def motion_probability(self, bank, action):
        return self.open_probability * (1 - self.toggle(bank, action))

    def observe(self, bank, action, moved, completed):
        self.tried.add(action)
        if self.probe is not None and action != self.probe:
            counts = self.success if moved else self.failure
            counts[bank][self.probe] = min(255, counts[bank][self.probe] + 1)
        if moved:
            self.open_probability = .98
            self.tried.clear()
            self.probe = None
        else:
            switch = self.toggle(bank, action)
            p = self.open_probability
            self.open_probability = p * (1 - switch) + (1 - p) * switch
            self.probe = action
        if completed:
            self.reset()

    def snapshot(self):
        mask = sum(1 << a for a in self.tried)
        return self.HEADER.pack(self.open_probability, self.probe if self.probe is not None else 256,
                                mask.to_bytes(32, "little")) + b"".join(self.success + self.failure)

    @classmethod
    def restore(cls, data):
        offset = cls.HEADER.size
        if len(data) != offset + 1024:
            raise ValueError("invalid belief payload size")
        probability, probe, mask = cls.HEADER.unpack_from(data)
        if not math.isfinite(probability) or not 0 <= probability <= 1 or probe > 256:
            raise ValueError("invalid two-mode posterior")
        b = cls()
        b.open_probability = probability
        b.probe = None if probe == 256 else probe
        tried = int.from_bytes(mask, "little")
        b.tried = {i for i in range(256) if tried & (1 << i)}
        parts = [bytearray(data[i:i + 256]) for i in range(offset, len(data), 256)]
        b.success, b.failure = parts[:2], parts[2:]
        return b
