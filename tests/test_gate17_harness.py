from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.gate17_gpu_scaling_harness import (
    DOMAINS, SMOKE, block_count, device_for, load_corpora, loss_for, make_batch, model_for,
    shuffled_index, training_rows,
)
from benchmarks.unified_checks import exactness
from nexus_expanse.patching import CompactBigram
from nexus_expanse.unified import NexusExpanseUnified


def test_partition_is_bijective_and_partial_tail_is_scored_once():
    raw = bytes(range(72))
    geometry = replace(SMOKE, context=16)
    count = block_count(raw, geometry.context)
    assert count == 5
    assert sorted(shuffled_index(i, count, 42) for i in range(count)) == list(range(count))
    batch, keep, valid = make_batch([(raw, i) for i in range(count)], geometry.context, torch.device("cpu"))
    assert valid == len(raw) - 1
    assert batch[:, 1:][keep].tolist() == list(raw[1:])
    assert len(training_rows({d: raw for d in DOMAINS}, geometry, 0)) == 3


def test_raw_file_custody_rejects_identical_train_eval(tmp_path):
    fields = {}
    for split in ("train", "eval"):
        for domain in DOMAINS:
            field = f"{split}_{domain}"
            path = tmp_path / field
            path.write_bytes(field.encode())
            fields[field] = path
    args = SimpleNamespace(smoke=False, **fields)
    _, _, identity = load_corpora(args)
    assert set(identity["train"]) == set(DOMAINS)
    args.eval_math.write_bytes(args.train_math.read_bytes())
    with pytest.raises(ValueError, match="identical"):
        load_corpora(args)


def test_device_selection_requires_native_cuda_bf16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert device_for("auto", "fp32").type == "cpu"
    with pytest.raises(ValueError, match="unavailable"):
        device_for("cuda", "fp32")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **_: False)
    assert device_for("auto", "fp32").type == "cuda"
    with pytest.raises(ValueError, match="BF16"):
        device_for("cuda", "bf16")


def test_hybrid_qualification_path_restores_wrapped_checkpoint():
    torch.set_num_threads(1)
    counts = CompactBigram.from_dense(np.zeros((256, 256), dtype="<u4"))
    unified = NexusExpanseUnified(model_for(SMOKE, "hybrid", torch.device("cpu")), counts)
    restored = NexusExpanseUnified.restore(unified.snapshot())
    assert not any(exactness(unified, restored, bytes(range(20))).values())
    continuation = NexusExpanseUnified.restore(unified.snapshot())
    assert torch.equal(unified.step(21).mixed, continuation.step(21).mixed)


def test_cpu_bf16_objective_produces_finite_gradients():
    torch.set_num_threads(1)
    cpu = torch.device("cpu")
    model = model_for(SMOKE, "causal_attention_monarch", cpu)
    batch, keep, _ = make_batch([(b"byte-tail", 0)], SMOKE.context, cpu)
    loss = loss_for(model, batch, keep, "bf16", cpu)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.emb.weight.grad).all()
