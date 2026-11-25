"""Engineering checks for the Gate 25 static donor path."""

from dataclasses import asdict
from io import BytesIO
import weakref

import pytest
import torch

from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.donor_assimilation import (
    DenseTokenDonorLM, assimilate_zero_run, fit_byte_bpe, forbid_donor_forward, graft_dense_to_monarch,
)
from nexus_expanse.jepa import state_hash
from nexus_expanse.substrate import DirectMonarchLinear


CONFIG = dict(dim=16, n_layers=2, dropout=0.0, max_seq_len=32)


@pytest.fixture
def static_donor():
    torch.manual_seed(2501)
    vocab, _ = fit_byte_bpe([b"banana bandana banana", b"byte bytes byte"], vocab_size=260)
    donor = DenseTokenDonorLM(vocab_size=260, dim=CONFIG["dim"])
    state, vocab = donor.export_static(vocab)
    reference = weakref.ref(donor)
    del donor
    assert reference() is None
    return state, vocab


def test_byte_bpe_is_deterministic_and_lossless():
    raw = [b"\x00\xffab\x00\xffab", b"ababcabc\r\n"]
    vocab, spans = fit_byte_bpe(raw, 260)
    assert (vocab, spans) == fit_byte_bpe(raw, 260)
    assert vocab[:256] == [bytes([value]) for value in range(256)]
    assert len(set(vocab)) == 260
    assert [b"".join(vocab[token] for token in row) for row in spans] == raw


def test_graft_reduces_frobenius_error_and_preserves_shapes():
    generator = torch.Generator().manual_seed(2502)
    target = DirectMonarchLinear(16, 32, n_blocks=4)
    dense = torch.randn(32, 16, generator=generator)
    shape = {name: param.shape for name, param in target.named_parameters()}

    def measured_error():
        with torch.no_grad():
            actual = target(torch.eye(16)).T
            return torch.linalg.vector_norm(actual - dense) / torch.linalg.vector_norm(dense)

    before = float(measured_error())
    errors = graft_dense_to_monarch(dense, target)
    after = float(measured_error())
    assert after < before
    assert errors.initial_relative_error == pytest.approx(before, abs=1e-6)
    assert errors.relative_error == pytest.approx(after, abs=1e-6)
    assert {name: param.shape for name, param in target.named_parameters()} == shape


def test_static_assimilation_and_controls(static_donor):
    state, vocab = static_donor
    original = {key: value.clone() for key, value in state.items()}
    results, students = {}, {}
    for mode in ("graft_and_bridge", "graft_only", "shuffled_donor_control", "scratch_nexus"):
        torch.manual_seed(2503)
        student = NexusExpanseByteCore(**CONFIG)
        before = state_hash(student)
        with forbid_donor_forward() as guard:
            results[mode] = assimilate_zero_run(student, state, vocab, mode=mode, bridge_steps=12)
            assert guard.donor_forward_calls == 0
        assert results[mode].donor_forward_calls == 0
        students[mode] = student
        if mode == "scratch_nexus":
            assert state_hash(student) == before

    full = results["graft_and_bridge"]
    graft_only = results["graft_only"]
    shuffled = results["shuffled_donor_control"]
    assert len(full.reconstruction) == len(shuffled.reconstruction) == 4
    assert asdict(full)["reconstruction"] == asdict(graft_only)["reconstruction"]
    assert full.bridge_mse_after < full.bridge_mse_before
    assert shuffled.bridge_mse_after < shuffled.bridge_mse_before
    assert graft_only.bridge_mse_before is None
    assert not torch.equal(students["graft_and_bridge"].emb.weight, students["graft_only"].emb.weight)
    assert torch.equal(students["graft_only"].emb.weight, students["scratch_nexus"].emb.weight)
    assert all(torch.equal(value, state[key]) for key, value in original.items())


def test_live_donor_is_rejected_and_guard_counts_attempts(static_donor):
    state, vocab = static_donor
    donor = DenseTokenDonorLM(vocab_size=len(vocab), dim=CONFIG["dim"])
    student = NexusExpanseByteCore(**CONFIG)
    with pytest.raises(TypeError, match="static CPU tensor"):
        assimilate_zero_run(student, donor, vocab)
    with pytest.raises(TypeError, match="static CPU tensor"):
        assimilate_zero_run(student, {**state, "donor": donor}, vocab)
    with pytest.raises(RuntimeError, match="attempted during assimilation"):
        with forbid_donor_forward() as guard:
            with pytest.raises(RuntimeError, match="forbidden"):
                donor(torch.tensor([[1, 2]]))
            assert guard.donor_forward_calls == 1
    assert donor.forward_calls == 0


def test_assimilated_byte_core_has_exact_prefix_causality(static_donor):
    student = NexusExpanseByteCore(**CONFIG)
    assimilate_zero_run(student, *static_donor, bridge_steps=12)
    student.eval()
    row = torch.arange(23, dtype=torch.long)[None]
    changed = row.clone()
    changed[:, 11:] = 255 - changed[:, 11:]
    with torch.inference_mode():
        expected = student.forward_neural(row)[0][:, :11]
        for probe in (row[:, :11], changed):
            measured = student.forward_neural(probe)[0][:, :11]
            assert float((measured - expected).abs().max()) == 0.0


def test_assimilated_byte_core_cold_checkpoint_is_bit_exact(static_donor):
    student = NexusExpanseByteCore(**CONFIG)
    assimilate_zero_run(student, *static_donor, bridge_steps=12)
    checkpoint = BytesIO()
    torch.save(student.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = NexusExpanseByteCore(**CONFIG)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    assert state_hash(student) == state_hash(restored)
    row = torch.tensor([[0, 10, 90, 250, 255]])
    with torch.inference_mode():
        expected = student.eval().forward_neural(row)[0]
        actual = restored.eval().forward_neural(row)[0]
    assert expected.numpy().tobytes() == actual.numpy().tobytes()
