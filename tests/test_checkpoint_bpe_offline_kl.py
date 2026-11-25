"""Unit tests for one-time offline byte-marginalized KL cache."""

import tempfile
from pathlib import Path

import torch

from nexus_expanse.donor_assimilation import DenseTokenDonorLM, forbid_donor_forward, learn_byte_bpe
from nexus_expanse.offline_byte_kl import (
    build_topk_byte_cache,
    load_offline_cache,
    offline_kl_loss,
    save_offline_cache,
)


def test_offline_cache_roundtrip_and_zero_student_donor_calls():
    corpus = [
        b"def compute_state(prefix, value):\n    return (prefix + value) & 255\n",
        b"class NexusKernel:\n    def step(self, x): return x\n",
    ]
    vocab, _, merges = learn_byte_bpe(corpus, 320)
    torch.manual_seed(11)
    donor = DenseTokenDonorLM(320, 32, 2, 4, 2, 64).eval()
    x = torch.randint(32, 122, (2, 16), dtype=torch.long)
    y = torch.randint(32, 122, (2, 16), dtype=torch.long)

    with tempfile.TemporaryDirectory() as d:
        cache_path = Path(d) / "fit_cache.pt"
        cache = build_topk_byte_cache(donor, vocab, merges, x, top_k=16)
        assert cache["indices"].dtype == torch.uint8
        assert cache["probs"].dtype == torch.float16
        sums = cache["probs"].float().sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-2)
        save_offline_cache(cache, cache_path)
        del donor

        with forbid_donor_forward() as guard:
            loaded = load_offline_cache(cache_path)
            student_logits = torch.randn(2, 16, 256, requires_grad=True)
            loss = offline_kl_loss(student_logits, y, loaded["indices"], loaded["probs"], kl_weight=0.5)
            loss.backward()
            assert student_logits.grad is not None
            assert torch.isfinite(loss)
            assert guard.donor_forward_calls == 0
