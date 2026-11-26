"""Phase 9 offline cache custody, no-autograd edits, and exact byte forecasts."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from benchmarks.checkpoint_transfer_study import setup_donor, source_pack
from nexus_expanse.cache_assimilation import assimilate
from nexus_expanse.checkpoint_transport import CorrectedLinear, ImportedSkill
from nexus_expanse.donor_cache_io import (
    GPT2_BYTES, _compact_windows, export_cache, load_hf_donor, load_nexus_donor, read_cache, token_bytes,
)
from nexus_expanse.donor_assimilation import DenseTokenDonorLM
from nexus_expanse.weight_editing import (
    apply_surgical_edit, inspect_fault_direction, prefix_diff, restore_skill, verified_surgical_edit,
    verify_skill,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def single_thread():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prior)


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True, check=True)


def skill_for(seed=1):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        skill = ImportedSkill().eval()
    skill.core.fixed_eval_geometry = True
    return skill


def direction(width, index=0):
    return torch.eye(width)[index]


def test_gpt2_decodes_the_entire_byte_alphabet():
    inverse = {byte: char for char, byte in GPT2_BYTES.items()}
    assert token_bytes("".join(inverse[i] for i in range(256)), "gpt2") == bytes(range(256))
    assert token_bytes("ĠhelloĊ", "gpt2") == b" hello\n"
    assert token_bytes("Ã©", "gpt2") == "é".encode()


def test_sentencepiece_metaspace_unicode_and_fallback():
    assert token_bytes("▁café▁", "sentencepiece") == " café ".encode()
    assert b"".join(token_bytes(f"<0x{value:02X}>", "sentencepiece") for value in range(256)) == bytes(range(256))
    assert token_bytes("▁<0x41>", "sentencepiece") == b" <0x41>"


@pytest.mark.parametrize("layout", ["single", "sharded"])
def test_hf_local_loader_only_uses_safetensors(tmp_path, monkeypatch, layout, single_thread):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text(json.dumps({"decoder": {"type": "ByteLevel"}}))
    (tmp_path / "pytorch_model.bin").write_bytes(b"must not be loaded")
    with pytest.raises(ValueError, match="safetensors"):
        load_hf_donor(tmp_path)

    if layout == "single":
        (tmp_path / "model.safetensors").touch()
    else:
        index = tmp_path / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"weight": "../outside.safetensors"}}))
        with pytest.raises(ValueError, match="nonlocal weights"):
            load_hf_donor(tmp_path)
        shard = "model-00001-of-00001.safetensors"
        index.write_text(json.dumps({"weight_map": {"weight": shard}}))
        (tmp_path / shard).touch()

    vocab = dict(zip(GPT2_BYTES, GPT2_BYTES.values()))
    by_byte = {byte: char for char, byte in vocab.items()}

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(1, 256)

        def get_output_embeddings(self):
            return self.head

        def forward(self, *, input_ids, use_cache):
            assert not use_cache
            return SimpleNamespace(logits=nn.functional.one_hot(input_ids, 256).float())

    class Tokenizer:
        @classmethod
        def from_file(cls, name):
            assert name == str(tmp_path / "tokenizer.json")
            return cls()

        def get_vocab(self):
            return vocab

        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return SimpleNamespace(ids=[vocab[by_byte[byte]] for byte in text.encode()])

    def load(path, **kwargs):
        assert path == str(tmp_path)
        assert kwargs == dict(local_files_only=True, trust_remote_code=False, use_safetensors=True)
        return Model()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load),
    ))
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=Tokenizer))
    _, encode, first, forward = load_hf_donor(tmp_path)
    assert encode(b"\xc3\xa9\xff") == (195, 169, 255)
    assert encode(b"\xc3") == (195,)
    shard = _compact_windows([b"\xc3\xa9\xff"], encode, first, forward)
    assert shard["indices"][0, :, 0].tolist() == [195, 169]


def test_dense_bpe_checkpoint_uses_only_observed_prefix(tmp_path, single_thread):
    model = DenseTokenDonorLM(vocab_size=257, dim=32, n_layers=2, n_heads=4, ffn_mult=2, max_seq_len=64)
    source = tmp_path / "bpe.pt"
    torch.save(dict(weights=model.state_dict(), vocab_bytes=[bytes([b]) for b in range(256)] + [b"ab"],
                    merges=[(97, 98)]), source)
    _, encode, first, forward = load_nexus_donor(source, 4)
    assert encode(b"a") == (97,) and encode(b"ab") == (256,)
    assert first[256].item() == 97
    prefix = _compact_windows([b"abc", b"abz"], encode, first, forward)
    assert torch.equal(prefix["indices"][0], prefix["indices"][1])
    assert torch.equal(prefix["probs"][0], prefix["probs"][1])


@pytest.fixture
def exported(tmp_path):
    checkpoint = tmp_path / "donor.pt"
    source_pack(checkpoint, setup_donor(8), torch.arange(256).flip(0))
    raw = tmp_path / "raw.bin"
    raw.write_bytes(bytes(range(256)))
    directory = tmp_path / "cache"
    manifest = json.loads(run(
        ROOT / "scripts/export_donor_cache.py", "--backend", "nexus_donor", "--model", checkpoint,
        "--input", raw, "--window-bytes", 64, "--shard-windows", 2, "--out-dir", directory,
    ).stdout)
    return directory, manifest


def test_local_donor_export_and_assimilate_cli(exported, tmp_path, single_thread):
    directory, manifest = exported
    cache, _ = read_cache(directory)
    assert len(manifest["shards"]) == 2
    assert cache["raw"].flatten().tolist() == list(range(256))
    assert cache["indices"].dtype == torch.uint8 and cache["probs"].dtype == torch.float16
    assert cache["indices"].shape[-1] == 32
    assert torch.max((cache["probs"].float().sum(-1) - 1).abs()) < 0.001

    out = tmp_path / "skill.pt"
    report = json.loads(run("-m", "nexus_expanse.cli", "assimilate", "--cache-dir", directory,
                            "--out", out, "--steps", 2, "--device", "cpu").stdout)
    assert report["donor_forward_calls"] == 0
    assert report["prefix_max_abs_diff"] == report["round_trip_max_abs_diff"] == 0.0
    assert report["exact_round_trip_state"]
    assert verify_skill(restore_skill(out))[1]["round_trip_max_abs_diff"] == 0.0

    shard = directory / manifest["shards"][0]["file"]
    data = bytearray(shard.read_bytes())
    data[-1] ^= 1
    shard.write_bytes(data)
    with pytest.raises(ValueError, match="integrity mismatch"):
        read_cache(directory)


def test_assimilation_and_edited_skill_exact_prefix_50_seeds(tmp_path, monkeypatch, single_thread):
    checkpoint = tmp_path / "donor.pt"
    source_pack(checkpoint, setup_donor(12), torch.arange(256))
    directory = tmp_path / "cache"
    export_cache(directory, [bytes(range(16)), bytes(range(240, 256))],
                 backend="nexus_donor", checkpoint=checkpoint)

    def forbidden(*args, **kwargs):
        pytest.fail("student path executed a donor")

    monkeypatch.setattr(DenseTokenDonorLM, "forward", forbidden)
    for seed in range(50):
        archive, receipt = assimilate(directory, device="cpu", dtype="fp32", steps=1, kl_weight=0.9, seed=seed)
        student = restore_skill(archive)
        edited = apply_surgical_edit(student, 1, "attn.out_proj", "subspace_ablate", direction(32, seed % 32), 0.5)
        assert receipt["donor_forward_calls"] == 0
        assert prefix_diff(student, seed=seed) == prefix_diff(edited, seed=seed) == 0.0


def test_cpu_bf16_assimilation_has_exact_fp32_archive(tmp_path, single_thread):
    checkpoint = tmp_path / "donor.pt"
    source_pack(checkpoint, setup_donor(16), torch.arange(256))
    directory = tmp_path / "cache"
    export_cache(directory, [b"raw\x00\xfffixture"], backend="nexus_donor", checkpoint=checkpoint)
    archive, receipt = assimilate(directory, device="cpu", dtype="bf16", steps=1, kl_weight=0.9, seed=4)
    assert receipt["dtype"] == "bf16" and receipt["exact_round_trip_state"]
    assert restore_skill(archive).readout.weight.dtype == torch.float32


@pytest.mark.parametrize("kind", ["dense", "monarch", "corrected"])
@pytest.mark.parametrize("op", ["subspace_ablate", "rank1_steer", "bias_shift"])
def test_affine_edits_match_closed_form_and_cold_restore(kind, op, single_thread):
    skill = skill_for()
    layer = skill.core.layers[1]
    if kind == "dense":
        target = "attn.out_proj"
        before = layer.attn.out_proj
        width_out = 32
    else:
        target = "ffn.fc1"
        if kind == "corrected":
            layer.ffn.fc1 = CorrectedLinear(layer.ffn.fc1, torch.rand(64, 32), "residual8")
        before = layer.ffn.fc1
        width_out = 64
    v, u = direction(32, 4), direction(width_out, 1)
    x = torch.randn(2, 9, 32)
    state = {key: value.clone() for key, value in skill.state_dict().items()}
    edited = apply_surgical_edit(skill, 1, target, op, v, 0.75, u)
    layer = edited.core.layers[1]
    after = layer.attn.out_proj if kind == "dense" else layer.ffn.fc1
    expected = before(x)
    if op == "subspace_ablate":
        expected = before(x - 0.75 * (x @ v).unsqueeze(-1) * v)
    elif op == "rank1_steer":
        expected = expected + 0.75 * (x @ v).unsqueeze(-1) * u
    else:
        expected = expected + 0.75 * u
    torch.testing.assert_close(after(x), expected, atol=3e-6, rtol=3e-6)
    assert all(torch.equal(state[key], value) for key, value in skill.state_dict().items())
    _, verification = verify_skill(edited)
    assert verification["exact_round_trip_state"]
    assert verification["round_trip_max_abs_diff"] == verification["prefix_max_abs_diff"] == 0.0


def test_editor_disables_grad_and_rolls_back_rejected_change(monkeypatch, single_thread):
    skill = skill_for()
    raw = torch.arange(24).reshape(2, 12)

    def forbidden(*args, **kwargs):
        pytest.fail("surgical editing called autograd")

    def inference_only(*args, **kwargs):
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()

    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    original_svd = torch.linalg.svd

    def svd(*args, **kwargs):
        inference_only()
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", svd)
    with torch.enable_grad():
        info = inspect_fault_direction(skill, raw, raw + 25, 1)
        assert len(info["layers"]) == 2
        assert not info["direction_v"].requires_grad

        def edit(model):
            inference_only()
            return apply_surgical_edit(model, 1, "attn.out_proj", "bias_shift", direction(32), 1.0)

        old_bias = skill.core.layers[1].attn.out_proj.bias.clone()

        def evaluate(model):
            inference_only()
            return 5.0 - float(model.core.layers[1].attn.out_proj.bias[0])

        accepted, rolled_back, receipt = verified_surgical_edit(skill, edit, evaluate, evaluate)
        assert not accepted and rolled_back is skill
        assert receipt["fault_after_bpb"] < receipt["fault_before_bpb"]
        assert receipt["retain_drift_bpb"] > 0.02
        assert torch.equal(old_bias, skill.core.layers[1].attn.out_proj.bias)

        def retain(model):
            inference_only()
            return 4.0

        accepted, edited, receipt = verified_surgical_edit(skill, edit, evaluate, retain)
        assert accepted and edited is not skill
        assert receipt["prefix_max_abs_diff"] == receipt["round_trip_max_abs_diff"] == 0.0
        assert all(param.grad is None for param in edited.parameters())


@pytest.mark.parametrize("op", ["inspect", "subspace_ablate"])
def test_weight_edit_cli_accepts_legacy_imported_skill(tmp_path, single_thread, op):
    source = tmp_path / "original.pt"
    skill = skill_for()
    torch.save(dict(state=skill.state_dict(), width=32, method="monarch"), source)
    out = tmp_path / "checked.pt"
    report = json.loads(run("-m", "nexus_expanse.cli", "weight-edit", "--skill", source, "--out", out,
                            "--op", op, "--layer", 1, "--alpha", 0.7).stdout)
    assert report["op"] == op and len(report["direction_v"]) == 32
    assert report["prefix_max_abs_diff"] == report["round_trip_max_abs_diff"] == 0.0
    assert report["exact_round_trip_state"]
    assert not restore_skill(out).training
