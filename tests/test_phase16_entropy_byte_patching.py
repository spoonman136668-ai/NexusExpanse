"""Offline coverage for causal patch boundaries, streaming, and optimization."""

from io import BytesIO

import pytest
import torch
from torch.nn import functional as F

from benchmarks.phase13_warmstart_train import build_corpus
from benchmarks.phase16_entropy_patching import DOMAINS, PATCH, check_stream, prefix_case, tensor
from nexus_expanse.entropy_patching import EntropyPatchedByteCore
from nexus_expanse.jepa import state_hash


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


@pytest.mark.parametrize("seed", range(100))
def test_prefix_differences_are_zero_after_truncation_and_suffix_mutation(seed):
    found = prefix_case(seed)
    assert all(value == 0.0 for key, value in found.items() if key not in ("seed", "prefix_bytes")), found


def test_threshold_cap_and_strict_availability_of_completed_patches():
    core = EntropyPatchedByteCore(dim=16, max_patch_bytes=3, training_bytes=b"a" * 4096).eval()
    raw = b"aaaaabaaaa"
    with torch.inference_mode():
        trace = core.patch_trace(tensor(raw))
        assert trace.starts[0].nonzero().flatten().tolist() == [0, 3, 5, 6, 9]
        assert trace.completed_ends.tolist() == [[2, 4, 5, 8]]
        assert not trace.visible_latents[0, :3].count_nonzero()
        assert torch.equal(trace.visible_latents[0, 3], trace.latents[0, 0])
        assert torch.equal(trace.visible_latents[0, 5], trace.latents[0, 1])
        assert not core.patch_trace(tensor(b"aa")).latents.numel()
        assert core.patch_trace(tensor(b"aaa")).completed_ends.tolist() == [[2]]

        # With no fitted transitions the surprise is exactly eight bits.
        uniform = EntropyPatchedByteCore(dim=16, entropy_threshold_bits=8.0).eval()
        assert uniform.patch_trace(tensor(b"aaaa")).starts.all()
        assert uniform.patcher.surprise_bits[97, 97] == 8.0


def test_streaming_reset_and_weights_only_restore_match_batch_across_patch_blocks():
    raw = (b"aaaaabaaaa!" * 14)[:151]
    tokens = tensor(raw)
    core = EntropyPatchedByteCore(dim=16, max_seq_len=160, training_bytes=b"a" * 4096, **PATCH).eval()
    with torch.inference_mode():
        expected = core.forward_neural(tokens)
        assert check_stream(core, tokens, expected) == {"logits": 0.0, "hidden": 0.0}
        trace = core.patch_trace(tokens)
        assert trace.latents.shape[1] > core.PATCH_BLOCK
        stream = core.new_stream()
        for value in raw:
            stream.append(value)
        assert stream.completed_count == trace.latents.shape[1]
        assert torch.equal(stream.latest, trace.latents[0, -1])
        stream.reset()
        assert stream.completed_count == stream.byte_count == 0
        first = stream.append(raw[0])
        assert all(torch.equal(a, b[0, 0]) for a, b in zip(first, expected, strict=True))

        out = BytesIO()
        torch.save(core.state_dict(), out)
        out.seek(0)
        restored = EntropyPatchedByteCore(dim=16, max_seq_len=160, **PATCH).eval()
        restored.load_state_dict(torch.load(out, weights_only=True), strict=True)
        assert state_hash(restored) == state_hash(core)
        assert all(torch.equal(a, b) for a, b in zip(expected, restored(tokens), strict=True))
        assert torch.equal(trace.visible_latents, restored.patch_trace(tokens).visible_latents)


def test_ragged_batched_patches_and_backward_step_are_finite():
    torch.manual_seed(16)
    model = EntropyPatchedByteCore(dim=16, max_seq_len=64, training_bytes=b"a" * 4096)
    tokens = torch.tensor([list(b"a" * 64), list(bytes(range(64)))])
    logits, hidden = model(tokens)
    assert logits.shape == (2, 64, 256) and hidden.shape == (2, 64, 16)
    trace = model.patch_trace(tokens)
    assert trace.starts.sum(1).tolist() == [8, 64]
    assert (trace.completed_ends[0, 8:] == -1).all() and not trace.latents[0, 8:].count_nonzero()
    assert torch.isfinite(logits).all()
    loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    for param in (model.emb.weight, model.local_kernel, model.patch_position.weight,
                  model.layers[0].attn.in_proj_weight, model.readout.weight):
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    scorer = model.patcher.surprise_bits.clone()
    before = model.layers[0].attn.in_proj_weight.detach().clone()
    torch.optim.AdamW(model.parameters(), lr=0.001).step()
    assert not torch.equal(before, model.layers[0].attn.in_proj_weight)
    assert torch.equal(scorer, model.patcher.surprise_bits)
    model.eval()
    with torch.inference_mode():
        assert torch.isfinite(model(tokens)[0]).all()
        for row in range(tokens.shape[0]):
            expected = model(tokens[row:row + 1])
            assert check_stream(model, tokens[row:row + 1], expected) == {"logits": 0.0, "hidden": 0.0}


def test_real_dev_text_compression_without_fitting_on_held_bytes(tmp_path):
    path = tmp_path / "corpus"
    build_corpus(path, 3001)
    training = [(path / f"{domain}-train.bin").read_bytes() for domain in DOMAINS]
    held = torch.stack([tensor((path / f"{domain}-eval.bin").read_bytes()[:512])[0] for domain in DOMAINS])
    model = EntropyPatchedByteCore(training_bytes=training, **PATCH)
    counts = model.patch_stats(held)
    assert counts["byte_count"] == 3 * 512
    assert counts["compression_ratio"] == counts["byte_count"] / counts["patch_count"] >= 2.0


def test_limits_reject_non_bytes_and_stream_overflow():
    core = EntropyPatchedByteCore(dim=16, max_seq_len=2)
    with pytest.raises(ValueError, match="eval model"):
        core.new_stream()
    with pytest.raises(ValueError, match="between 0 and 255"):
        core(torch.tensor([[256]]))
    stream = core.eval().new_stream()
    with pytest.raises(ValueError, match="raw byte"):
        stream.append(-1)
    stream.append(0)
    stream.append(255)
    with pytest.raises(ValueError, match="context limit"):
        stream.append(1)
