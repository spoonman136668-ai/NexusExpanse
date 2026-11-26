"""Phase 10B one-sided retention, shard custody and no-donor raw-byte adaptation."""

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch

from benchmarks.phase10b_shard_stream_and_edit import MOCK_FILES, write_mock_shards
from nexus_expanse.cli import main as cli_main
from nexus_expanse.donor_assimilation import DenseTokenDonorLM
from nexus_expanse.shard_stream_assimilation import _layer_keys, adapt_skill, assimilate_zero_exec, make_student
from nexus_expanse.weight_editing import (
    apply_surgical_edit, prefix_diff, restore_skill, verified_surgical_edit, verify_skill,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


@pytest.fixture
def mock_shards(tmp_path):
    rng = torch.Generator().manual_seed(2400)

    def weight(*shape):
        return torch.randn(shape, generator=rng) * 0.12

    embed = {"model.embed_tokens.weight": weight(320, 64)}
    head = {"lm_head.weight": weight(320, 64), "engram.weight": weight(256, 256)}
    tail = {}
    for i in range(2):
        prefix = f"model.layers.{i}."
        # Split both layers across shards to exercise grouping by file.
        head[prefix + "self_attn.v_proj.weight"] = weight(64, 64)
        head[prefix + "mlp.up_proj.weight"] = weight(96, 64)
        tail[prefix + "self_attn.o_proj.weight"] = weight(64, 64)
        tail[prefix + "mlp.down_proj.weight"] = weight(64, 96)
    vocab = [bytes((b,)) for b in range(256)] + [bytes((65, b)) for b in range(64)]
    path = tmp_path / "donor"
    write_mock_shards(path, (embed, head, tail), vocab)
    return path


@pytest.mark.parametrize("retain_delta,expected", [(-0.4, True), (0.015625, True), (0.03125, False)])
def test_retention_is_one_sided(retain_delta, expected, one_thread, monkeypatch):
    skill = make_student(9).eval()

    def fail(*args, **kwargs):
        pytest.fail("editing used autograd")

    monkeypatch.setattr(torch.autograd, "grad", fail)
    monkeypatch.setattr(torch.autograd, "backward", fail)
    monkeypatch.setattr(torch.Tensor, "backward", fail)
    v = torch.eye(32)[0]

    def edit(model):
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        return apply_surgical_edit(model, 1, "attn.out_proj", "bias_shift", v, 1.0)

    def shifted(model):
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        return float(model.core.layers[1].attn.out_proj.bias[0])

    accepted, result, receipt = verified_surgical_edit(
        skill, edit, lambda m: 5 - shifted(m), lambda m: 5 + retain_delta * shifted(m),
    )
    assert accepted is expected
    assert (result is not skill) is expected
    assert receipt["retain_drift_bpb"] == pytest.approx(retain_delta)
    assert receipt["prefix_max_abs_diff"] == receipt["round_trip_max_abs_diff"] == 0.0
    assert receipt["exact_round_trip_state"]
    assert float(skill.core.layers[1].attn.out_proj.bias[0].detach()) == 0.0


def test_mock_reader_releases_each_shard_and_stays_causal_across_50_seeds(mock_shards, one_thread, monkeypatch):
    previous = []
    originals = torch.load
    reads = []

    def read(source, *args, **kwargs):
        if Path(str(source)).name not in MOCK_FILES:
            return originals(source, *args, **kwargs)
        assert kwargs["weights_only"] is True
        assert all(ref() is None for ref in previous)
        shard = originals(source, *args, **kwargs)
        previous[:] = [weakref.ref(tensor) for tensor in shard.values()]
        reads.append(Path(source).name)
        return shard

    def forbidden(*args, **kwargs):
        pytest.fail("static importer instantiated or executed a donor")

    monkeypatch.setattr(torch, "load", read)
    monkeypatch.setattr(DenseTokenDonorLM, "__init__", forbidden)
    monkeypatch.setattr(DenseTokenDonorLM, "forward", forbidden)

    for seed in range(50):
        archive, receipt = assimilate_zero_exec(mock_shards, mock_pt=True, seed=seed)
        assert receipt["peak_loaded_shards"] == 1
        assert receipt["loaded_shard_count"] == receipt["indexed_shard_count"] == 3
        assert receipt["donor_forward_calls"] == 0
        assert receipt["prefix_max_abs_diff"] == receipt["round_trip_max_abs_diff"] == 0.0
        assert receipt["exact_round_trip_state"]
        assert receipt["donor_width"] == 64 and receipt["donor_vocab_size"] == 320
        assert receipt["uses_static_engram"]
        skill = restore_skill(archive)
        assert skill.prior_alpha == 0.25
        assert prefix_diff(skill, seed=seed) == 0.0
    assert reads == list(MOCK_FILES) * 50


def test_safetensors_path_uses_local_reader_only(mock_shards, one_thread, monkeypatch):
    map_path = mock_shards / "model.pt.index.json"
    weight_map = json.loads(map_path.read_bytes())["weight_map"]
    by_safetensors = {}
    for i, source in enumerate(MOCK_FILES):
        target = f"model-{i}.safetensors"
        by_safetensors[target] = mock_shards / source
        (mock_shards / target).write_bytes(bytes([i]))
    names = dict(zip(MOCK_FILES, by_safetensors, strict=True))
    (mock_shards / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {key: names[file] for key, file in weight_map.items()}},
    ))
    live = 0

    @contextmanager
    def fake_open(path, *, framework, device):
        nonlocal live
        assert Path(path).parent == mock_shards
        assert live == 0 and framework == "pt" and device == "cpu"
        live += 1
        try:
            state = torch.load(by_safetensors[Path(path).name], weights_only=True)
            yield SimpleNamespace(get_tensor=state.__getitem__)
        finally:
            live -= 1

    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=fake_open))
    _, receipt = assimilate_zero_exec(mock_shards)
    assert live == receipt["donor_forward_calls"] == 0
    assert receipt["format"] == "safetensors" and receipt["peak_loaded_shards"] == 1
    assert receipt["local_files_only"] and not receipt["trust_remote_code"]


def test_single_file_safetensors_discovers_keys_without_index(mock_shards, one_thread, monkeypatch):
    weights = {}
    for file in MOCK_FILES:
        weights.update(torch.load(mock_shards / file, weights_only=True))
    model = mock_shards / "model.safetensors"
    model.write_bytes(b"fixture")
    visits = []

    @contextmanager
    def safe_open(path, *, framework, device):
        assert Path(path) == model and framework == "pt" and device == "cpu"
        visits.append(path)
        yield SimpleNamespace(keys=weights.keys, get_tensor=weights.__getitem__)

    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=safe_open))
    archive, receipt = assimilate_zero_exec(mock_shards)
    reference, _ = assimilate_zero_exec(mock_shards, mock_pt=True)
    assert not (mock_shards / "model.safetensors.index.json").exists()
    assert len(visits) == 3  # Header discovery, embedding projection, then the remaining reductions.
    assert receipt["peak_loaded_shards"] == receipt["indexed_shard_count"] == receipt["loaded_shard_count"] == 1
    assert receipt["exact_round_trip_state"] and receipt["donor_forward_calls"] == 0
    state = restore_skill(archive).state_dict()
    for name, tensor in restore_skill(reference).state_dict().items():
        assert torch.equal(state[name], tensor)


@pytest.mark.parametrize("dense", [True, False], ids=["qwen", "deepseek"])
def test_standard_attention_and_shared_expert_aliases(dense):
    prefix = "model.layers.0."
    attention = {f"{part}_proj": prefix + f"self_attn.{part}_proj.weight" for part in "qkvo"}
    shared = {f"{part}_proj": prefix + f"mlp.shared_experts.{part}_proj.weight" for part in ("up", "down")}
    routed = [prefix + f"mlp.experts.{i}.{part}_proj.weight" for i in range(2) for part in ("up", "down")]
    plain = {f"{part}_proj": prefix + f"mlp.{part}_proj.weight" for part in ("up", "down")}
    keys = [*attention.values(), *shared.values(), *routed]
    if dense:
        keys.extend(plain.values())
    assert list(_layer_keys(keys, 1)) == [{**attention, **(plain if dense else shared)}]


@pytest.mark.parametrize("grouped", [False, True], ids=["tied-head", "grouped-query"])
def test_hf_repo_streams_only_needed_shards_once(mock_shards, tmp_path, one_thread, monkeypatch, capsys, grouped):
    names = [f"model-{i:05d}.safetensors" for i in range(3)]
    states = dict(zip(names, [torch.load(mock_shards / file, weights_only=True) for file in MOCK_FILES], strict=True))
    first, middle, _ = states.values()
    v_key = "model.layers.0.self_attn.v_proj.weight"
    first[v_key] = middle.pop(v_key)
    if grouped:
        first[v_key] = first[v_key][:32].contiguous()
        first["lm_head.weight"] = middle.pop("lm_head.weight")
    else:
        del middle["lm_head.weight"]
    weight_map = {key: name for name, state in states.items() for key in state}
    weight_map["model.layers.2.self_attn.v_proj.weight"] = "unused.safetensors"
    index = "model.safetensors.index.json"
    metadata = {
        index: json.dumps({"weight_map": weight_map}).encode(),
        "tokenizer.json": (mock_shards / "tokenizer.json").read_bytes(),
        "config.json": b'{"num_attention_heads": 4, "num_key_value_heads": 2}',
    }
    calls = []
    paths = []
    live = 0
    revision = "a" * 40

    def download(*, repo_id, filename, revision, local_dir):
        assert repo_id == "example/donor" and revision == "a" * 40
        calls.append(filename)
        path = local_dir / filename
        if filename in metadata:
            path.write_bytes(metadata[filename])
        else:
            assert filename in states and live == 0
            assert all(not previous.exists() for previous in paths)
            path.write_bytes(filename.encode())
            paths.append(path)
        return str(path)

    def hub_url(repo, name):
        assert (repo, name) == ("example/donor", index)
        return "index-url"

    def hub_metadata(url):
        assert url == "index-url" and not calls
        return SimpleNamespace(commit_hash=revision)

    @contextmanager
    def safe_open(path, *, framework, device):
        nonlocal live
        path = Path(path)
        assert path == paths[-1] and framework == "pt" and device == "cpu"
        assert live == 0 and list(path.parent.parent.glob("shard-*/*.safetensors")) == [path]
        live += 1
        try:
            yield SimpleNamespace(get_tensor=states[path.name].__getitem__)
        finally:
            live -= 1

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        hf_hub_download=download, hf_hub_url=hub_url, get_hf_file_metadata=hub_metadata,
    ))
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", SimpleNamespace(HfHubHTTPError=OSError))
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=safe_open))
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda _: None)
    out = tmp_path / "student.pt"
    assert cli_main(["assimilate-zero-exec", "--hf-repo", "example/donor", "--out", str(out)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert calls == [index, "tokenizer.json", names[0], *(["config.json"] if grouped else []), *names[1:]]
    assert receipt["hf_revision"] == revision and receipt["hf_repo"] == "example/donor"
    assert receipt["peak_loaded_shards"] == receipt["peak_disk_shards"] == 1
    assert receipt["loaded_shard_count"] == receipt["shard_visits"] == 3
    assert receipt["indexed_shard_count"] == 4
    assert receipt["donor_forward_calls"] == live == 0
    assert all(not path.parent.parent.exists() for path in paths)
    assert receipt["exact_round_trip_state"] and restore_skill(out).prior_alpha == 0.25


def test_reject_nonlocal_shard_before_loading(mock_shards, one_thread, monkeypatch):
    map_path = mock_shards / "model.pt.index.json"
    index = json.loads(map_path.read_bytes())
    key = next(iter(index["weight_map"]))
    index["weight_map"][key] = "../outside.pt"
    map_path.write_text(json.dumps(index))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("read an unverified shard"))
    with pytest.raises(ValueError, match="local regular checkpoint"):
        assimilate_zero_exec(mock_shards, mock_pt=True)


def test_decaying_alpha_reaches_native_only_and_restores(mock_shards, one_thread):
    archive, initial = assimilate_zero_exec(mock_shards, mock_pt=True)
    student = restore_skill(archive)
    raw = torch.arange(64).reshape(4, 16)
    adaptive, final = adapt_skill(student, raw, seed=2400, steps=5)
    assert final["prior_alpha_schedule"] == [0.25, 0.1875, 0.125, 0.0625, 0.0]
    assert initial["donor_forward_calls"] == final["donor_forward_calls"] == 0
    assert final["exact_round_trip_state"]
    assert final["prefix_max_abs_diff"] == final["round_trip_max_abs_diff"] == 0.0
    final_skill = restore_skill(adaptive)
    assert final_skill.prior_alpha == 0.0
    with torch.inference_mode():
        expected = final_skill(raw)
        assert torch.equal(expected, student(raw))
        final_skill.static_byte_prior.fill_(-100)
        assert torch.equal(expected, final_skill(raw))
    assert verify_skill(student)[1]["round_trip_max_abs_diff"] == 0.0


@pytest.mark.parametrize("entrypoint", [
    ("-m", "nexus_expanse.cli", "assimilate-zero-exec"),
    ("scripts/stream_safetensors_zero_exec.py",),
])
def test_cli_fresh_process(mock_shards, tmp_path, one_thread, entrypoint):
    out = tmp_path / "student.pt"
    result = subprocess.run(
        [sys.executable, *entrypoint, "--donor-dir", str(mock_shards), "--out", str(out), "--mock-pt"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "PASS"
    assert receipt["loaded_shard_count"] == 3 and receipt["peak_loaded_shards"] == 1
    assert receipt["donor_forward_calls"] == 0
    assert receipt["prefix_max_abs_diff"] == receipt["round_trip_max_abs_diff"] == 0.0
    assert restore_skill(out).prior_alpha == 0.25
