from collections import Counter

from benchmarks.ablation_dev import CONFIG
from benchmarks.kv_parity_dev import compare
from nexus_expanse import NexusExpanseByteCore
from nexus_expanse.latent_association import AlignedLatentAssociation


def test_aligned_code_and_permanent_edge():
    model=NexusExpanseByteCore(**CONFIG).eval()
    arm=AlignedLatentAssociation(model)
    a,r,y=(arm.codes_by_byte[b] for b in (65,97,66))
    assert arm.encode(b'aAa')[0] == (r,a,r)
    arm.outcomes.add(a)
    arm.edges[a,r]=Counter({y:2})
    arm.records=arm.frequency[r]=100
    assert arm.route((a,r)) == [(a,r)]
    arm.observe((a,r),66)
    assert arm.edges[a,r] == {y:3} and arm.output_byte(y)==66


def test_kv_prefix_and_window_rebuild():
    import torch
    threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for kind in ('dense','monarch'):
            assert compare(42,kind,True,32,(1,7,32,33)) == (0.,0)
    finally:
        torch.set_num_threads(threads)
