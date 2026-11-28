"""Offline BF16 and weight-only snapshot regression coverage."""

from dataclasses import replace
import hashlib
from io import BytesIO
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from benchmarks import gate17_gpu_scaling_harness as scaling
from benchmarks.phase13_warmstart_train import cli
from benchmarks.phase15_cosine_lr_and_sampling import call_cli
from benchmarks.unified_checks import exactness
from nexus_expanse.cli import default_model
from nexus_expanse.jepa import state_hash
from nexus_expanse.patching import CompactBigram
from nexus_expanse.substrate import CausalGatedMonarchFFN
from nexus_expanse.unified import NexusExpanseUnified
from scripts.prepare_gpu_corpus import assert_disjoint, corpus_bytes


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


def original_fp32_update(module, h, deep):
    selected = h[deep]
    if module.enable_skip and not module.training:
        chunks = []
        for group in selected.split(32):
            block = F.pad(group, (0, 0, 0, 32 - len(group)))
            chunks.append(module.fc2(F.silu(module.fc1(block)))[:len(group)])
        values = torch.cat(chunks)
    else:
        values = module.fc2(F.silu(module.fc1(selected)))
    result = torch.zeros_like(h)
    result[deep] = values
    return result


@pytest.mark.parametrize("ffn_kind", ["dense", "monarch"])
@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
@pytest.mark.parametrize("enable_skip", [True, False], ids=["gated", "always-on"])
def test_bf16_gate_write_and_bit_exact_fp32_gradients(ffn_kind, training, enable_skip):
    torch.manual_seed(121)
    module = CausalGatedMonarchFFN(32, 64, n_blocks=8, enable_skip=enable_skip, ffn_kind=ffn_kind).train(training)
    h = torch.randn(2, 29, 32, requires_grad=True)
    targets = (h, *module.fc1.parameters(), *module.fc2.parameters())
    fp32, _, deep = module(h, valid_length=23)
    assert int(deep.sum()) == 46
    expected = original_fp32_update(module, h, deep)
    assert torch.equal(fp32, expected)
    for got, ref in zip(torch.autograd.grad(fp32.sum(), targets), torch.autograd.grad(expected.sum(), targets), strict=True):
        assert torch.equal(got, ref)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        updated, scores, selected = module(h, valid_length=23)
    assert updated.dtype == h.dtype and scores.dtype == torch.bfloat16
    assert torch.equal(selected, deep) and torch.isfinite(updated).all()
    assert not updated[~deep].count_nonzero()
    assert all(torch.isfinite(grad).all() for grad in torch.autograd.grad(updated.sum(), targets))


def load_archive(raw):
    return torch.load(BytesIO(raw), weights_only=True, map_location="cpu")


@pytest.mark.parametrize("quant", ["fp16", "int8"])
@pytest.mark.parametrize("seed", [7, 121, 512])
def test_quantized_cold_round_trip_exact_prefix_and_archived_cursor(quant, seed):
    geometry = scaling.Geometry(seed, 32, 32, 1, 1, 1, 1)
    backbone = scaling.model_for(geometry, "causal_attention_monarch", torch.device("cpu"))
    counts = CompactBigram.from_dense(np.zeros((256, 256), dtype="<u4"))
    model = NexusExpanseUnified(backbone, counts)
    for byte in b"in-flight byte history":
        model.step(byte)
    fp32 = model.snapshot()
    assert model.snapshot(snapshot_quant="fp32") == fp32
    raw = model.snapshot(snapshot_quant=quant)
    a, b = map(load_archive, (fp32, raw))
    assert "snapshot_quant" not in a and b["snapshot_quant"] == quant
    for name in ("compact_counts", "episodic", "relation", "arbitration", "reader", "history", "delta_state"):
        assert a[name] == b[name]
    torch.testing.assert_close(a["preceding_forecast"], b["preceding_forecast"], atol=0, rtol=0)
    expected_dtype = {"fp16": torch.float16, "int8": torch.int8}[quant]
    parameters = dict(backbone.named_parameters())
    for name, tensor in b["backbone"].items():
        if name in parameters and tensor.ndim >= 2:
            assert tensor.dtype == expected_dtype
        else:
            assert tensor.dtype == a["backbone"][name].dtype and torch.equal(tensor, a["backbone"][name])
    if quant == "int8":
        name = "layers.0.ffn.fc1.b1"
        codes, scale = b["backbone"][name], b["int8_scales"][name]
        assert scale.shape == (codes.shape[0], 1, codes.shape[2]) and scale.dtype == torch.float32
        assert torch.equal(codes, (a["backbone"][name] / scale).round().clamp(-127, 127).to(torch.int8))

    first = NexusExpanseUnified.restore(raw)
    assert first.snapshot_quant == quant
    repacked = first.snapshot(snapshot_quant=quant)
    cold = NexusExpanseUnified.restore(repacked)
    second = NexusExpanseUnified.restore(raw)
    assert state_hash(first.backbone) == state_hash(second.backbone) == state_hash(cold.backbone)
    for byte in b"\x00cold continuation!\xff":
        expected = first.step(byte, adapt=False).mixed
        assert torch.equal(cold.step(byte, adapt=False).mixed, expected)
        assert torch.equal(second.step(byte, adapt=False).mixed, expected)
    cold.reset_context()
    assert set(exactness(first, cold, bytes(range(32))).values()) == {0.0}


def test_weight_only_compression_bounds_and_bpb_on_trained_256x6_fixture(record_property):
    train = {d: corpus_bytes(121, "train", d, 2048) for d in scaling.DOMAINS}
    held = {d: corpus_bytes(121, "eval", d, 129) for d in scaling.DOMAINS}
    assert_disjoint(train.values(), held)
    geometry = scaling.Geometry(121, 256, 512, 1, 16, 1, 1, 6)
    cpu = torch.device("cpu")
    backbone = scaling.model_for(geometry, "causal_attention_skill", cpu)
    batches = replace(geometry, context=64)
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=.002)
    for step in range(16):
        rows = scaling.training_rows(train, batches, step)
        batch, keep, _ = scaling.make_batch(rows, batches.context, cpu)
        optimizer.zero_grad(set_to_none=True)
        scaling.update(backbone, optimizer, scaling.loss_for(backbone, batch, keep, "fp32", cpu))
    model = scaling.build_unified(backbone, train)
    archives = {mode: model.snapshot(snapshot_quant=mode) for mode in ("fp32", "fp16", "int8")}
    sizes = {mode: len(raw) for mode, raw in archives.items()}
    record_property("snapshot_bytes", json.dumps(sizes))
    assert sizes["fp32"] / sizes["fp16"] >= 1.8 and sizes["fp32"] / sizes["int8"] >= 3.2, sizes

    def bpb(raw):
        restored = NexusExpanseUnified.restore(raw)
        scores = {}
        for domain, data in held.items():
            tokens = torch.tensor([list(data)])
            logits = restored.forward_neural(tokens[:, :-1])[0]
            scores[domain] = float(F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten()) / np.log(2))
        return scores

    baseline = bpb(archives["fp32"])
    for mode in ("fp16", "int8"):
        deltas = {domain: abs(value - baseline[domain]) for domain, value in bpb(archives[mode]).items()}
        record_property(f"{mode}_absolute_bpb_deltas", json.dumps(deltas))
        assert max(deltas.values()) <= 0.03, (mode, deltas)


@pytest.mark.parametrize("quant", ["fp32", "fp16", "int8"])
def test_verify_conversion_and_generate_always_wake(tmp_path, quant):
    source, target = tmp_path / "source.pt", tmp_path / "packed.pt"
    source.write_bytes(default_model().snapshot())
    packed = call_cli("verify", "--snapshot", source, "--save-snapshot", target, "--snapshot-quant", quant)
    assert packed == call_cli("verify", "--snapshot", target, "--snapshot-sha256", packed["snapshot_sha256"])
    assert packed["snapshot_quant"] == quant and packed["exact_state"]
    assert packed["round_trip_probabilities"] == 0.0 and set(packed["parity"].values()) == {0.0}
    assert packed["donor_forward_calls"] == 0
    generated = call_cli("generate", "--snapshot", target, "--prompt-text", "    ", "--max-bytes", 8, "--always-wake")
    assert generated["generated_bytes"] == 8 and generated["always_wake"] and generated["donor_forward_calls"] == 0
    assert generated["snapshot_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    if quant == "fp32":
        assert source.read_bytes() == target.read_bytes()


def test_train_exports_quantized_snapshot_with_verified_native_state(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for d in scaling.DOMAINS:
        for split in ("train", "eval"):
            (corpus / f"{d}-{split}.bin").write_bytes(corpus_bytes(121, split, d, 96))
    run, target = tmp_path / "run", tmp_path / "trained.pt"
    exported = cli("train", "--corpus-dir", corpus, "--directory", run, "--device", "cpu", "--dim", 32,
                   "--context", 16, "--epochs", 1, "--steps-per-epoch", 1, "--batch-per-domain", 1,
                   "--eval-blocks", 1, "--skill-backbone", "--save-snapshot", target, "--snapshot-quant", "int8")
    assert exported["snapshot_quant"] == "int8" and exported["exact_state"] and exported["donor_forward_calls"] == 0
    assert set(exported["parity"].values()) == {0.0} and exported["round_trip_probabilities"] == 0.0
    identity = json.loads((run / "gate17.json").read_bytes())
    assert exported["neural_state_sha256"] == identity["final_hash"]
    verified = call_cli("verify", "--snapshot", target, "--snapshot-sha256", exported["snapshot_sha256"])
    assert verified["snapshot_quant"] == "int8" and verified["exact_state"]


def test_restore_rejects_invalid_int8_channel_data():
    archive = load_archive(default_model().snapshot(snapshot_quant="int8"))
    archive["int8_scales"]["emb.weight"][0].zero_()
    damaged = BytesIO()
    torch.save(archive, damaged)
    with pytest.raises(ValueError, match="invalid channel data: emb.weight"):
        NexusExpanseUnified.restore(damaged.getvalue())
