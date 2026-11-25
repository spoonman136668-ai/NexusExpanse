"""Unit test for width-lifted 32->64 BPE BoundarySkill parity."""

import tempfile
from pathlib import Path

import torch

from benchmarks.checkpoint_bpe_scaled_width_kl import WidthBoundarySkill
from nexus_expanse.donor_assimilation import DenseTokenDonorLM, learn_byte_bpe


def test_width64_boundary_skill_matches_width32_zero_update():
    corpus = [
        b"def compute_state(prefix, value):\n    return (prefix + value) & 255\n",
        b"class NexusKernel:\n    def step(self, x): return x\n",
    ]
    vocab, _, _ = learn_byte_bpe(corpus, 320)
    torch.manual_seed(17)
    donor = DenseTokenDonorLM(320, 32, 2, 4, 2, 64)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "donor.pt"
        torch.save(
            dict(weights=donor.state_dict(), byte_to_row=torch.arange(256), spellings=[v.hex() for v in vocab]),
            path,
        )
        m32 = WidthBoundarySkill(path, "residual8", width_factor=1).eval()
        m64 = WidthBoundarySkill(path, "residual8", width_factor=2).eval()
        x = torch.randint(0, 256, (2, 16), dtype=torch.long)
        with torch.inference_mode():
            y32 = m32(x)
            y64 = m64(x)
        assert (y32 - y64).abs().max().item() < 1e-2
