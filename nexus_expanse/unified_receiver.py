"""Use a trained unified snapshot as a frozen, context-local neural expert."""

from pathlib import Path

from torch import nn

from nexus_expanse.jepa import state_hash
from nexus_expanse.unified import NexusExpanseUnified


class UnifiedSnapshotSkill(nn.Module):
    """Route the archived native backbone and retain its original storage bytes."""

    def __init__(self, archive):
        super().__init__()
        archive = bytes(archive)
        runtime = NexusExpanseUnified.restore(archive)
        self.core = runtime.backbone
        if not hasattr(self.core, "max_seq_len"):
            self.core.max_seq_len = runtime.max_seq_len
        self.snapshot_quant = runtime.snapshot_quant
        self._archive = archive
        self._archived_state = state_hash(self.core)
        self.eval()

    @classmethod
    def from_file(cls, path):
        with Path(path).open("rb") as source:
            return cls(source.read(NexusExpanseUnified.SNAPSHOT_LIMIT + 1))

    def forward(self, raw):
        # Routers supply a sliding prefix. The saved unified cursor does not
        # contribute observations to that prefix.
        return self.core.forward_neural(raw)[0]

    def archive(self):
        if state_hash(self.core) != self._archived_state:
            raise ValueError("unified receiver changed after snapshot restore")
        return self._archive

    def repack(self):
        """Store an adapted clone and route its actual dequantized weights."""
        runtime = NexusExpanseUnified.restore(self._archive)
        runtime.backbone.load_state_dict(self.core.state_dict())
        runtime.reset_context()
        return type(self)(runtime.snapshot(snapshot_quant=self.snapshot_quant))
