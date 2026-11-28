"""Offline native snapshot initialization and exact curriculum custody."""

import errno
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from benchmarks import gate17_gpu_scaling_harness as scaling
from nexus_expanse import cli
from nexus_expanse.deltanet import DeltaNetMonarchByteCore
from nexus_expanse.jepa import state_hash
from nexus_expanse.livebench_eval import CATEGORIES
from nexus_expanse.unified import NexusExpanseUnified
from scripts.prepare_gpu_corpus import assert_disjoint, grams
from scripts import prepare_reasoning_curriculum as curriculum

CPU = torch.device("cpu")


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


def model_snapshot(backbone, quant="int8", *, seed=23):
    geometry = scaling.Geometry(seed, 32, 16, 1, 1, 1, 1, 2)
    model = (DeltaNetMonarchByteCore(dim=32) if backbone == "delta" else
             scaling.model_for(geometry, backbone, CPU))
    return scaling.build_unified(model, {scaling.STREAM:b"base examples about bytes"}).snapshot(snapshot_quant=quant)


@pytest.fixture
def raw_paths(tmp_path, monkeypatch):
    source, held = tmp_path / "fit.bin", tmp_path / "check.bin"
    source.write_bytes((b"test-time native neural weights\n" * 3)[:65])
    held.write_bytes(b"independent cold bytes\n" * 2)
    git = scaling.git
    monkeypatch.setattr(scaling, "git", lambda *args: b"" if args[0] == "status" else git(*args))
    return source, held


def train_args(raw_paths, source, directory):
    return ("train", "--train-file", raw_paths[0], "--eval-file", raw_paths[1], "--directory", directory,
            "--init-snapshot", source, "--device", "cpu", "--steps-per-epoch", "1", "--batch-per-domain", "1",
            "--eval-blocks", "1")


def invoke(capsys, *argv):
    cli.main(list(map(str,argv)))
    output = capsys.readouterr().out.splitlines()
    return [json.loads(line) for line in output if line.startswith("{")]


@pytest.mark.parametrize("backbone,quant", [("causal_attention_skill","int8"), ("causal_attention_monarch","fp16"),
                                              ("hybrid","fp32"), ("delta","fp32")])
def test_init_snapshot_uses_verified_native_layout_in_cli_and_cold_export(tmp_path, raw_paths, capsys, backbone, quant):
    source, exported = tmp_path / "source.pt", tmp_path / "out.pt"
    raw = model_snapshot(backbone,quant)
    source.write_bytes(raw)
    options = () if backbone == "delta" else ("--context", "32")
    if quant == "fp16":
        options += ("--dtype", "bf16")
    report = invoke(capsys, *train_args(raw_paths, source, tmp_path / "run"), "--epochs", "1", *options,
                    "--save-snapshot", exported, "--snapshot-quant", "int8")[-1]
    model = NexusExpanseUnified.restore(exported.read_bytes())
    assert model.max_seq_len == (256 if backbone == "delta" else 32)
    assert report["init_snapshot_sha256"] == hashlib.sha256(raw).hexdigest()
    assert set(report["parity"].values()) == {0.0} and report["round_trip_probabilities"] == 0.0
    assert report["exact_state"] and report["donor_forward_calls"] == 0
    receipt = json.loads((tmp_path / "run/gate17.json").read_bytes())
    assert receipt["cold_equal"] and receipt["identity"]["backbone"] == backbone
    assert receipt["identity"]["dtype"] == ("bf16" if quant == "fp16" else "fp32")
    assert receipt["identity"]["geometry"]["dim"] == 32
    assert receipt["identity"]["init_snapshot_sha256"] == report["init_snapshot_sha256"]


def test_snapshot_initialization_copies_all_weights_and_only_seeds_new_positions():
    saved = NexusExpanseUnified.restore(model_snapshot("causal_attention_skill"))
    args = SimpleNamespace(dim=32, layers=2, context=24, skill_backbone=True, use_sdpa=False)
    dim,layers,context,backbone,architecture = scaling.snapshot_geometry(saved,args)
    geometry = scaling.Geometry(826,dim,context,1,1,1,1,layers)
    model = scaling.model_for(geometry,backbone,CPU,snapshot_architecture=architecture)
    new_positions = model.pos_emb.weight[16:].detach().clone()
    scaling.initialize_from_snapshot(model,saved)
    for key, before in saved.backbone.state_dict().items():
        current = model.state_dict()[key]
        assert torch.equal(before,current[:16] if key == "pos_emb.weight" else current)
    assert torch.equal(model.pos_emb.weight[16:],new_positions)
    assert all(p.requires_grad for p in model.parameters())
    for name,value in (("dim",64),("layers",4),("context",8)):
        invalid=SimpleNamespace(**dict(vars(args),**{name:value}))
        with pytest.raises(ValueError, match="initial snapshot|training context"):
            scaling.snapshot_geometry(saved,invalid)
    monarch = NexusExpanseUnified.restore(model_snapshot("causal_attention_monarch"))
    with pytest.raises(ValueError, match="contradicts"):
        scaling.snapshot_geometry(monarch,args)
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["train","--train-file","fit.bin","--directory","run","--init-skill","skill.pt",
                                 "--init-snapshot","snapshot.pt"])


def test_init_snapshot_checks_digest_on_resume_and_reproduces_uninterrupted_bytes(tmp_path, raw_paths, capsys):
    source = tmp_path / "source.pt"
    raw = model_snapshot("causal_attention_skill")
    source.write_bytes(raw)
    resumed, direct = tmp_path / "resumed", tmp_path / "direct"
    arguments = (*train_args(raw_paths,source,resumed), "--epochs", "2", "--lr-schedule", "cosine")
    invoke(capsys,*arguments,"--epoch-limit","1")
    intent = resumed / "gate17-epoch-01.intent.json"
    first = intent.read_bytes()
    assert json.loads(first)["identity"]["init_snapshot_sha256"] == hashlib.sha256(raw).hexdigest()
    first_state = json.loads((resumed / "gate17-epoch-01.json").read_bytes())
    assert first_state["telemetry"]["start_model_hash"] == state_hash(NexusExpanseUnified.restore(raw).backbone)
    source.write_bytes(model_snapshot("causal_attention_skill", seed=99))
    with pytest.raises(SystemExit) as reject:
        invoke(capsys,*arguments,"--epoch-limit","1")
    assert reject.value.code == 2 and "epoch identity or hash mismatch" in capsys.readouterr().err
    assert intent.read_bytes() == first and not (resumed / "gate17-epoch-02.intent.json").exists()
    source.write_bytes(raw)
    x,y = (tmp_path / name for name in ("resumed.pt","uninterrupted.pt"))
    invoke(capsys,*arguments,"--epoch-limit","1","--save-snapshot",x,"--snapshot-quant","int8")
    invoke(capsys,*train_args(raw_paths,source,direct),"--epochs","2","--epoch-limit","2","--lr-schedule","cosine",
           "--save-snapshot",y,"--snapshot-quant","int8")
    a,b=(json.loads((p/"gate17.json").read_bytes()) for p in (resumed,direct))
    assert a["cold_equal"] and b["cold_equal"] and a["final_hash"] == b["final_hash"]
    for left,right in zip(a["epochs"],b["epochs"],strict=True):
        assert left["sha256"] == right["sha256"] and left["losses"] == right["losses"]
    assert x.read_bytes() == y.read_bytes()


def test_curriculum_reproducible_and_disjoint_even_when_blocks_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(curriculum,"BLOCK_ROUNDS",7)
    a,b = (tmp_path / name for name in ("a","b"))
    reports = [curriculum.prepare(path,size_bytes=22_000,seed=37) for path in (a,b)]
    assert reports[0] == reports[1]
    manifest = reports[0]
    assert not any(manifest["shared_32_byte_windows"].values()) and manifest["donor_forward_calls"] == 0
    t,e=(f"stream-{suffix}.bin" for suffix in ("train","eval"))
    for name,meta in manifest["files"].items():
        data = (a/name).read_bytes()
        assert data == (b/name).read_bytes() and meta["sha256"] == hashlib.sha256(data).hexdigest()
        assert set(meta["per_category"]) == set(CATEGORIES) and len(set(meta["per_category"].values())) == 1
        assert meta["examples"] == sum(meta["per_category"].values()) > 5*meta["unique_per_category"]
    assert manifest["verifier_passed"] == sum(row["examples"] for row in manifest["files"].values())
    fit,held=(p.read_bytes() for p in (a/t,a/e))
    assert_disjoint((fit,),{"eval":held})
    assert_disjoint((fit,held),curriculum.builtin_references())
    assert len(fit)+len(held) <= 22_000
    for split,cap,raw in (("T",17_600,fit),("E",4_400,held)):
        assert set(grams(raw)) == set(grams(curriculum.plan(37,split,cap).coverage()))


def test_curriculum_atomic_fuse_publication_and_verifier_rejection(tmp_path, monkeypatch):
    original = curriculum.make_verifier
    def unsupported(*args,**kwargs):
        raise PermissionError(errno.EPERM,"FUSE denies hard links")
    monkeypatch.setattr(curriculum.os,"link",unsupported)
    path=tmp_path/"curriculum"
    receipt=curriculum.prepare(path,size_bytes=9500,seed=8)
    assert json.loads((path/"curriculum_manifest.json").read_bytes()) == receipt
    assert {p.name for p in path.iterdir()} == {"stream-train.bin","stream-eval.bin","curriculum_manifest.json"}
    with pytest.raises(FileExistsError):
        curriculum.prepare(path,size_bytes=9500,seed=8)
    monkeypatch.setattr(curriculum,"make_verifier",lambda *a,**k: lambda _: SimpleNamespace(passed=False))
    destination=tmp_path/"rejected"
    with pytest.raises(ValueError,match="verifier failed"):
        curriculum.prepare(destination,size_bytes=9500,seed=8)
    assert not destination.exists()
    monkeypatch.setattr(curriculum,"make_verifier",original)
