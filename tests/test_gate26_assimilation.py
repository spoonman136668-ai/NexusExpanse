"""Exact tensor, byte-probability, guard, and restore checks for Gate 26."""

from io import BytesIO
import math
import weakref

import pytest
import torch

from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.donor_assimilation import (
    DenseTokenDonorLM, assimilate_zero_run, byte_marginalized_probabilities, canonical_byte_bpe,
    export_offline_byte_kl_cache, forbid_donor_forward, learn_byte_bpe, load_offline_byte_kl_cache,
    sparse_byte_kl,
)
from nexus_expanse.jepa import state_hash


CONFIG = dict(dim=16, n_layers=2, dropout=0.0, max_seq_len=32, enable_skip=False, fixed_eval_geometry=True)
VOCAB = [bytes([byte]) for byte in range(256)] + [b"ab"]
MERGES = [(ord("a"), ord("b"))]


@pytest.fixture
def donor():
    torch.manual_seed(2601)
    model = DenseTokenDonorLM(vocab_size=len(VOCAB), dim=CONFIG["dim"])
    with torch.no_grad():
        for layer in model.layers:
            for parameter in (layer.attn.in_proj_bias, layer.attn.out_proj.bias,
                              layer.attn_norm.weight, layer.attn_norm.bias,
                              layer.ffn_norm.weight, layer.ffn_norm.bias):
                parameter.copy_(torch.randn_like(parameter))
    return model.eval().requires_grad_(False)


def prior():
    mass = torch.ones(len(VOCAB), dtype=torch.float64)
    mass[256] = 200
    mass[ord("a")] = 50
    return mass / mass.sum()


def test_canonical_bpe_reproduces_training_ids():
    spans = [b"banana bandana banana", b"byte bytes byte"]
    vocab, ids, merges = learn_byte_bpe(spans, vocab_size=260)
    ranks = {pair: token for token, pair in enumerate(merges, 256)}
    assert [list(canonical_byte_bpe(raw, ranks)) for raw in spans] == ids
    for raw in spans:
        for t in range(len(raw)):
            assert b"".join(vocab[token] for token in canonical_byte_bpe(raw[:t], ranks)) == raw[:t]


@pytest.mark.parametrize("mode", ["zero_run_full_graft", "shuffled_full_graft"])
def test_full_graft_preserves_exact_copied_tensors(donor, mode):
    state, vocab = donor.export_static(VOCAB)
    student = NexusExpanseByteCore(**CONFIG)
    report = assimilate_zero_run(student, state, vocab, mode=mode, bridge_steps=12, shuffle_seed=26)
    assert report.donor_forward_calls == donor.forward_calls == 0
    assert len(report.copied_max_abs_error) == 16
    assert set(report.copied_max_abs_error.values()) == {0.0}
    assert len(report.reconstruction) == 4
    assert report.bridge_mse_after < report.bridge_mse_before
    if mode == "zero_run_full_graft":
        assert all(torch.equal(student.state_dict()[key], state[key]) for key in report.copied_max_abs_error)
    else:
        key = "layers.0.attn.out_proj.weight"
        shuffled = student.state_dict()[key]
        assert not torch.equal(shuffled, state[key])
        torch.testing.assert_close(torch.linalg.svdvals(shuffled), torch.linalg.svdvals(state[key]))


def test_exact_canonical_trie_marginal_is_normalized_and_causal(donor):
    masses = prior()
    full, _ = byte_marginalized_probabilities(donor, [b"abaxqu", b"abayzz"], VOCAB, MERGES, masses)
    assert donor.forward_calls == 1
    assert torch.equal(full[0][:4], full[1][:4])
    for rows in full:
        torch.testing.assert_close(rows.sum(-1), torch.ones(len(rows), dtype=rows.dtype), atol=1e-5, rtol=0)

    # Prefix b"a" can end at a singleton boundary or inside b"ab".
    with torch.inference_mode():
        token_probs = donor(torch.tensor([[ord("a")]]))[0, 0].double().softmax(-1)
    first_bytes = torch.tensor([spelling[0] for spelling in VOCAB])
    expected = torch.zeros(256, dtype=torch.float64)
    expected.index_add_(0, first_bytes, token_probs * masses[ord("a")])
    expected[ord("b")] += masses[256]
    expected /= expected.sum()
    torch.testing.assert_close(full[0][1], expected, atol=1e-7, rtol=1e-7)


def test_offline_and_hybrid_train_from_disk_after_donor_deletion(tmp_path):
    donor = DenseTokenDonorLM(vocab_size=len(VOCAB), dim=CONFIG["dim"]).eval().requires_grad_(False)
    path = tmp_path / "bytes.bin"
    spans = [b"aba test", b"ab hello"]
    receipt = export_offline_byte_kl_cache(donor, path, spans, VOCAB, MERGES, prior())
    assert donor.forward_calls == receipt["donor_forward_calls_offline"] == 1
    assert receipt["cache_bytes"] == path.stat().st_size == 45 + 3 * 16 * sum(map(len, spans))
    state, vocab = donor.export_static(VOCAB)
    reference = weakref.ref(donor)
    del donor
    assert reference() is None
    with forbid_donor_forward() as guard:
        cache = load_offline_byte_kl_cache(path, spans, receipt["sha256"])
        with pytest.raises(ValueError, match="FIT identity"):
            load_offline_byte_kl_cache(path, spans[::-1], receipt["sha256"])
        with pytest.raises(ValueError, match="digest"):
            load_offline_byte_kl_cache(path, spans, "0" * 64)
        for hybrid in (False, True):
            model = NexusExpanseByteCore(**CONFIG)
            if hybrid:
                assimilate_zero_run(model, state, vocab, mode="zero_run_full_graft", bridge_steps=2)
            x = torch.tensor([list(spans[0][:-1])])
            ids, probs = (values[None, 1:] for values in cache[0])
            torch.testing.assert_close(probs.sum(-1), torch.ones(x.shape), atol=1e-5, rtol=0)
            opt = torch.optim.AdamW(model.parameters(), lr=0.003)
            before = state_hash(model)
            logits = model.forward_neural(x)[0]
            compact = sparse_byte_kl(logits, ids, probs)
            dense = torch.zeros_like(logits).scatter_(-1, ids, probs)
            direct = torch.nn.functional.kl_div(logits.log_softmax(-1), dense, reduction="sum") / x.numel()
            torch.testing.assert_close(compact, direct, atol=1e-6, rtol=1e-6)
            target = torch.tensor(list(spans[0][1:]))
            loss = torch.nn.functional.cross_entropy(logits[0], target) + 0.5 * compact
            assert math.isfinite(float(loss.detach()))
            loss.backward()
            opt.step()
            assert state_hash(model) != before
        assert guard.donor_forward_calls == 0


def test_full_graft_prefix_and_restored_predictions_are_bit_exact(donor):
    student = NexusExpanseByteCore(**CONFIG)
    assimilate_zero_run(student, *donor.export_static(VOCAB), mode="zero_run_full_graft", bridge_steps=3)
    row = torch.arange(24)[None]
    changed = row.clone()
    changed[:, 9:] = 255
    student.eval()
    checkpoint = BytesIO()
    torch.save(student.state_dict(), checkpoint)
    restored = NexusExpanseByteCore(**CONFIG).eval()
    checkpoint.seek(0)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert state_hash(student) == state_hash(restored)
    with torch.inference_mode():
        logits = student.forward_neural(row)[0]
        for probe in (row[:, :9], changed):
            measured = student.forward_neural(probe)[0][:, :9]
            assert float((logits[:, :9] - measured).abs().max()) == 0.0
        assert logits.numpy().tobytes() == restored.forward_neural(row)[0].numpy().tobytes()
