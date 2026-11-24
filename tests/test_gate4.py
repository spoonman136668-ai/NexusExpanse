import numpy as np
import torch

from benchmarks.ablation_dev import CONFIG
from nexus_expanse import NexusExpanseByteCore
from nexus_expanse.patching import BytePatchReader
from nexus_expanse.latent_association import LatentAssociation


def test_patch_uses_observed_surprise_and_resets_history():
    model = NexusExpanseByteCore(**CONFIG).eval()
    counts = np.zeros((256,256), dtype=np.uint32)
    counts[65,65] = 1000
    reader = BytePatchReader(model,counts)
    first = reader.forecast(65)
    folded = reader.forecast(65)
    assert reader.deep_calls == 1 and reader.pending == 65
    assert folded[65].item() == 1001 / 1256
    reader.forecast(66)
    assert list(reader.events) == [65,65,66] and reader.deep_calls == 2
    reader.reset_context()
    assert torch.equal(first,reader.forecast(65))


def test_wta_writes_only_the_winning_edge_and_restores():
    arm = LatentAssociation(NexusExpanseByteCore(**CONFIG).eval())
    arm.outcomes.add(65)
    arm.observe((65,97),66)
    arm.records = 20
    arm.observe((65,97,98),67)
    assert arm.edges[(65,97)] == {66:1}
    assert arm.edges[(66,98)] == {67:1}
    assert len(arm.edges) == 2
    copy = LatentAssociation(arm.model)
    copy.restore(arm.snapshot())
    assert copy.snapshot() == arm.snapshot()
