"""Unit tests for BPE causal-history folding (zero donor execution, strict byte causality)."""

import tempfile
from pathlib import Path

import torch

from nexus_expanse.bpe_folding import (
    FoldedBoundarySkill,
    build_entropy_join_table,
    build_static_join_table,
    compute_causal_chunk_schedule,
    rebuild_folded_skill_from_state,
)
from nexus_expanse.donor_assimilation import DenseTokenDonorLM, learn_byte_bpe


def _make_dummy_donor(tmp: Path, seed: int = 123):
    corpus = [
        b"def compute_state(prefix, value):\n    return (prefix + value) & 255\n",
        b"class NexusKernel:\n    def step(self, x): return x\n",
    ]
    vocab, _, _ = learn_byte_bpe(corpus, 320)
    torch.manual_seed(seed)
    donor = DenseTokenDonorLM(320, 32, 2, 4, 2, 64)
    path = tmp / "donor.pt"
    torch.save(
        dict(
            weights=donor.state_dict(),
            byte_to_row=torch.arange(256),
            spellings=[v.hex() for v in vocab],
        ),
        path,
    )
    return path


def test_causal_chunk_schedule_zero_future_leakage():
    torch.manual_seed(7)
    join = build_entropy_join_table()
    x1 = torch.randint(32, 127, (4, 32), dtype=torch.long)
    for split in (1, 7, 16, 31):
        x2 = x1.clone()
        x2[:, split:] = torch.randint(32, 127, (4, 32 - split), dtype=torch.long)
        s1, p1, m1 = compute_causal_chunk_schedule(x1, join)
        s2, p2, m2 = compute_causal_chunk_schedule(x2, join)
        assert torch.equal(s1[:, :split], s2[:, :split])
        assert torch.equal(p1[:, :split], p2[:, :split])
        assert torch.equal(m1[:, :split, :split], m2[:, :split, :split])


def test_folded_boundary_skill_exact_causality_and_cold_restore():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        donor_path = _make_dummy_donor(tmp)
        for mode in ("entropy_fold", "static_trie_fold"):
            model = FoldedBoundarySkill(donor_path, "residual8", mode).eval()
            torch.manual_seed(99)
            for seed in range(10):
                g = torch.Generator().manual_seed(seed)
                x1 = torch.randint(0, 256, (2, 24), generator=g, dtype=torch.long)
                cut = 12
                x2 = x1.clone()
                x2[:, cut:] = torch.randint(0, 256, (2, 12), generator=g, dtype=torch.long)
                with torch.inference_mode():
                    y1 = model(x1)
                    y2 = model(x2)
                    diff = (y1[:, :cut] - y2[:, :cut]).abs().max().item()
                    assert diff == 0.0, f"{mode} causality diff {diff}"

            x_probe = torch.randint(0, 256, (2, 20), dtype=torch.long)
            with torch.inference_mode():
                out_orig = model(x_probe)
            restored = rebuild_folded_skill_from_state(model.state_dict(), "residual8", mode).eval()
            with torch.inference_mode():
                out_rest = restored(x_probe)
            assert torch.equal(out_orig, out_rest)


def test_static_join_table_from_vocab_only():
    vocab = [bytes([i]) for i in range(256)] + [b"th", b"the", b"ing", b"er", b"re"]
    table = build_static_join_table(vocab, threshold=0.2)
    assert table.shape == (256, 256)
    assert table.dtype == torch.bool
    assert bool(table[ord("t"), ord("h")]) is True
    assert bool(table[ord(" "), ord("t")]) is False
