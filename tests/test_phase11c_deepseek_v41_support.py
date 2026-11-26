"""Offline compatibility checks for the DeepSeek-V4.1-Flash checkpoint layout."""

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch

from nexus_expanse import shard_stream_assimilation as streaming
from nexus_expanse.donor_cache_io import GPT2_BYTES, token_bytes
from nexus_expanse.weight_editing import restore_skill


BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
LITERAL = "<｜literal｜>"
BLOCK = 8
HIDDEN = 64
HEAD_DIM = 35
HEADS = 4
GROUPS = 2
QUERY_RANK = 20
OUTPUT_RANK = 17


@pytest.fixture
def one_thread():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prior)


def tokenizer_data():
    vocab = {BOS: 0, EOS: 1, **{char: byte + 2 for char, byte in GPT2_BYTES.items()}}
    return {
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": vocab},
        "added_tokens": [
            {"id": 0, "content": BOS, "special": True},
            {"id": 1, "content": EOS, "special": True},
            {"id": 258, "content": LITERAL, "special": False},
        ],
    }


@pytest.mark.parametrize(("text", "decoded"), [
    (BOS, BOS.encode("utf-8")), (EOS, EOS.encode("utf-8")),
    ("Ġx｜", "Ġx｜".encode("utf-8")), ("ĠhelloĊ", b" hello\n"), ("Ã©", b"\xc3\xa9"),
])
def test_gpt2_alphabet_and_deepseek_literal_fallback(text, decoded):
    assert token_bytes(text, "gpt2") == decoded


def test_deepseek_bytelevel_lexicon_preserves_all_bytes_and_excludes_special_ids():
    singles, ids, first = streaming._lexicon(json.dumps(tokenizer_data()).encode())
    assert singles.tolist() == list(range(2, 258))
    assert ids.tolist() == list(range(2, 259))
    assert first.tolist() == [*range(256), ord("<")]


def write_checkpoint(directory, states):
    directory.mkdir()
    weight_map = {}
    for name, weights in states.items():
        torch.save(weights, directory / name)
        weight_map.update(dict.fromkeys(weights, name))
    (directory / "model.pt.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer_data()))
    (directory / "config.json").write_text(json.dumps({
        "text_config": {"head_dim": HEAD_DIM, "num_attention_heads": HEADS, "o_groups": GROUPS},
        "quantization_config": {"weight_block_size": [BLOCK, BLOCK], "scale_fmt": "ue8m0"},
    }))


@pytest.fixture
def deepseek_shards(tmp_path):
    rng = torch.Generator().manual_seed(1103)

    def random_weight(*shape):
        return torch.randn(shape, generator=rng) * 0.1

    states = {
        "embedding.pt": {"embed.weight": random_weight(259, HIDDEN).bfloat16()},
        "head.pt": {"head.weight": random_weight(259, HIDDEN).bfloat16(), "norm.weight": torch.ones(HIDDEN)},
        "extra.pt": {
            "vision.layers.0.attn.wkv.weight": torch.empty(1),
            "vision.norm.weight": torch.empty(1),
            "mtp.2.markov_head.embed.weight": torch.empty(1),
            "mtp.2.markov_head.head.weight": torch.empty(1),
        },
    }
    reference = {name: weights.copy() for name, weights in states.items()}
    layer_weights = {
        "attn.wkv": (HEAD_DIM, HIDDEN),
        "attn.wq_a": (QUERY_RANK, HIDDEN),
        "attn.wq_b": (HEADS * HEAD_DIM, QUERY_RANK),
        "attn.wo_a": (GROUPS * OUTPUT_RANK, HEADS // GROUPS * HEAD_DIM),
        "attn.wo_b": (HIDDEN, GROUPS * OUTPUT_RANK),
        "ffn.shared_experts.w1": (48, HIDDEN),
        "ffn.shared_experts.w2": (HIDDEN, 48),
        "ffn.shared_experts.w3": (48, HIDDEN),
    }
    for layer, dtype in enumerate((torch.float8_e4m3fn, torch.float8_e5m2)):
        for key, (rows, cols) in layer_weights.items():
            # Put one output factor before its own layer's KV in the stream.
            shard = "layer-0.pt" if layer == 1 and key == "attn.wo_a" else f"layer-{layer}.pt"
            prefix = f"layers.{layer}.{key}"
            weight = random_weight(rows, cols).to(dtype)
            blocks = ((rows + BLOCK - 1) // BLOCK, (cols + BLOCK - 1) // BLOCK)
            scale = torch.exp2(torch.randint(-2, 3, blocks, generator=rng).float()).to(torch.float8_e8m0fnu)
            states.setdefault(shard, {})[prefix + ".weight"] = weight
            states[shard][prefix + ".scale"] = scale
            full_scale = scale.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
            reference.setdefault(shard, {})[prefix + ".weight"] = weight.float() * full_scale[:rows, :cols]
        shard = f"layer-{layer}.pt"
        for norm in ("attn_norm", "ffn_norm"):
            key = f"layers.{layer}.{norm}.weight"
            states[shard][key] = reference[shard][key] = torch.ones(HIDDEN)
    states["extra.pt"]["layers.1.engram.wkv.weight"] = torch.empty(1)
    reference["extra.pt"]["layers.1.engram.wkv.weight"] = torch.empty(1)

    checkpoint, dense = tmp_path / "fp8", tmp_path / "fp32"
    write_checkpoint(checkpoint, states)
    write_checkpoint(dense, reference)
    return checkpoint, dense


@pytest.mark.parametrize("remote", [False, True], ids=["local", "streamed"])
def test_deepseek_scaled_shards_match_fp32_and_restore_exactly(deepseek_shards, one_thread, monkeypatch, remote):
    checkpoint, dense = deepseek_shards
    sources = {path.with_suffix(".safetensors").name: path for path in checkpoint.glob("*.pt")}
    index = json.loads((checkpoint / "model.pt.index.json").read_bytes())["weight_map"]
    visited = []
    staged = []

    if remote:
        def download(name, directory):
            path = directory / name
            visited.append(name)
            if name == streaming.HF_INDEX:
                path.write_text(json.dumps({"weight_map": {
                    key: Path(shard).with_suffix(".safetensors").name for key, shard in index.items()
                }}))
            elif name in sources:
                assert all(not previous.exists() for previous in staged)
                shutil.copyfile(sources[name], path)
                staged.append(path)
            else:
                shutil.copyfile(checkpoint / name, path)
            return path

        @contextmanager
        def safe_open(path, *, framework, device):
            assert framework == "pt" and device == "cpu"
            weights = torch.load(path, weights_only=True)
            yield SimpleNamespace(get_tensor=weights.__getitem__)

        monkeypatch.setattr(streaming, "_hf_source", lambda _: (download, "a" * 40))
        monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=safe_open))
        archive, receipt = streaming.assimilate_zero_exec(hf_repo="offline/deepseek")
        assert visited[:2] == [streaming.HF_INDEX, "tokenizer.json"]
        assert visited.count("config.json") == 1 and "extra.safetensors" not in visited
        assert receipt["peak_disk_shards"] == 1 and all(not path.exists() for path in staged)
    else:
        archive, receipt = streaming.assimilate_zero_exec(checkpoint, mock_pt=True)

    expected, _ = streaming.assimilate_zero_exec(dense, mock_pt=True)
    assert receipt["peak_loaded_shards"] == 1 and receipt["loaded_shard_count"] == 4
    assert receipt["prefix_max_abs_diff"] == receipt["round_trip_max_abs_diff"] == 0.0
    assert receipt["donor_forward_calls"] == 0 and receipt["exact_round_trip_state"]
    skill, reference = restore_skill(archive), restore_skill(expected)
    expected_state = reference.state_dict()
    assert all(torch.equal(tensor, expected_state[key]) for key, tensor in skill.state_dict().items())


def test_shared_expert_up_prefers_w3_and_falls_back_to_w1(deepseek_shards):
    checkpoint, _ = deepseek_shards
    keys = json.loads((checkpoint / "model.pt.index.json").read_bytes())["weight_map"]
    preferred = "layers.0.ffn.shared_experts.w3.weight"
    assert list(streaming._layer_keys(keys, 2))[0]["up_proj"] == preferred
    del keys[preferred]
    assert list(streaming._layer_keys(keys, 2))[0]["up_proj"] == "layers.0.ffn.shared_experts.w1.weight"


@pytest.mark.parametrize(("scale", "expected"), [
    (torch.tensor(2.0), [[2, 4, 6], [8, 10, 12]]),
    (torch.tensor([1.0, 2.0, 4.0]), [[1, 4, 12], [4, 10, 24]]),
    (torch.tensor([2.0, 4.0]), [[2, 4, 6], [16, 20, 24]]),
])
def test_integer_weights_broadcast_scalar_column_and_row_scales(scale, expected):
    values = torch.arange(1, 7, dtype=torch.int8).reshape(2, 3)
    result = streaming._ScaledWeight(values, scale, {})[:, :]
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float32))


def test_quantized_slice_respects_uneven_block_edges_and_ue8m0_bytes():
    values = torch.ones((5, 7), dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[1, 2, 4], [8, 16, 32]], dtype=torch.float32)
    encoded = scale.to(torch.float8_e8m0fnu).view(torch.uint8)
    weight = streaming._ScaledWeight(values, encoded, {"weight_block_size": [3, 3], "scale_fmt": "ue8m0"})
    reference = scale.repeat_interleave(3, 0).repeat_interleave(3, 1)[:5, :7]
    indices = torch.tensor([4, 0, 3])
    torch.testing.assert_close(weight[indices, 2:7], reference[indices, 2:7], rtol=0, atol=0)
    torch.testing.assert_close(weight[2:5, 2:6], reference[2:5, 2:6], rtol=0, atol=0)


def test_grouped_mla_projection_preserves_linear_query_key_and_value_output_products(one_thread):
    rng = torch.Generator().manual_seed(327)
    width, head_dim, heads, groups, rank = 3, 5, 4, 2, 2
    kv = torch.randn((head_dim, width), generator=rng)
    q_a = torch.randn((rank, width), generator=rng)
    q_b = torch.randn((heads * head_dim, rank), generator=rng)
    o_a = torch.randn((groups * rank, heads // groups * head_dim), generator=rng)
    o_b = torch.randn((width, groups * rank), generator=rng)
    factors = dict(wkv=kv, wq_a=q_a, wq_b=streaming._sum_heads(q_b, head_dim, 0) / heads,
                   wo_a=streaming._sum_heads(o_a, head_dim, 1), wo_b=o_b)
    result = streaming._mla_projections(factors, width)
    queries = (q_b @ q_a).reshape(heads, head_dim, width).mean(0)
    group_values = kv.repeat(heads, 1).reshape(groups, -1, width)
    grouped = torch.einsum("grd,gds->grs", o_a.reshape(groups, rank, -1), group_values)
    torch.testing.assert_close(result["q_proj"].T @ result["k_proj"], queries.T @ kv)
    torch.testing.assert_close(result["o_proj"] @ result["v_proj"], o_b @ grouped.flatten(0, 1))


@pytest.mark.parametrize("split_scale", [False, True])
def test_integer_weight_requires_a_same_shard_scale(tmp_path, split_scale):
    weight = {"a.weight": torch.ones((2, 2), dtype=torch.int8)}
    states = {"weight.pt": weight, "scale.pt": {"a.scale": torch.tensor(2.0)}} if split_scale else {"weight.pt": weight}
    path = tmp_path / "donor"
    write_checkpoint(path, states)
    stream = streaming.ShardStream(path, mock_pt=True)
    message = "same shard" if split_scale else "floating-point or scaled"
    with pytest.raises(ValueError, match=message):
        stream.consume({"a.weight": lambda _: pytest.fail("received an undecoded weight")}, read_config=lambda: {})
    assert stream.live == 0 and stream.visits == 1


def test_large_vocab_basis_uses_float32_gram_and_bounded_rows(monkeypatch, one_thread):
    rng = torch.Generator().manual_seed(31)
    weight = torch.randn((4500, 48), generator=rng) * torch.linspace(0.1, 2, 48)
    ids = torch.arange(0, 4500, 2)
    anchors = weight[ids[:256]]
    _, singular, vh = torch.linalg.svd(weight[ids], full_matrices=False)
    expected = vh[:8].T @ vh[:8]
    reads = []

    class AuditedWeight:
        shape = weight.shape

        def __getitem__(self, rows):
            reads.append(len(rows))
            return weight[rows]

    eigh = torch.linalg.eigh
    svd = torch.linalg.svd

    def audit_eigh(gram):
        assert gram.shape == (48, 48) and gram.dtype == torch.float32
        exact = weight[ids].double()
        torch.testing.assert_close(gram, (exact.T @ exact).float(), rtol=2e-6, atol=2e-4)
        return eigh(gram)

    def audit_svd(matrix, *, full_matrices):
        assert matrix.shape == (8, 8)
        return svd(matrix, full_matrices=full_matrices)

    monkeypatch.setattr(torch.linalg, "eigh", audit_eigh)
    monkeypatch.setattr(torch.linalg, "svd", audit_svd)
    basis, spectrum = streaming._aligned_basis(AuditedWeight(), ids, anchors, torch.eye(256, 8))
    assert sum(reads) == len(ids) and max(reads) <= streaming.ROW_CHUNK
    torch.testing.assert_close(torch.tensor(spectrum), singular[:8], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(basis @ basis.T, expected, rtol=1e-4, atol=1e-5)
