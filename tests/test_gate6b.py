"""Unit tests for Gate 6B KuramotoMonarchByteCore O(1) streaming & prefix exactness."""

from __future__ import annotations

import torch

from nexus_expanse.kuramoto import KuramotoMonarchByteCore


def test_kuramoto_bit_exact_prefix_and_streaming_parity() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    m = KuramotoMonarchByteCore(n_layers=2, dim=32, ffn_mult=2, enable_skip=False, dropout=0.0).eval()
    seq = torch.tensor(list(b"nexus expanse kuramoto phase binding"), dtype=torch.long)[None]

    with torch.inference_mode():
        full_logits = m.forward_neural(seq)[0]
        pref_logits = m.forward_neural(seq[:, :13])[0]
        assert torch.equal(full_logits[:, :13], pref_logits)

        states = m.init_stream_state(1)
        for idx in range(seq.shape[1]):
            p_step, states = m.step_streaming(int(seq[0, idx].item()), states)
            p_batch = full_logits[0, idx].double().softmax(0)
            assert torch.equal(p_step, p_batch)
