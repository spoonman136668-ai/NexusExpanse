"""Offline format, streaming causality, and CLI tests for SSD Engram."""

from collections import Counter
import errno
import hashlib
from pathlib import Path
import sqlite3

import pytest
import torch

from benchmarks.phase15_cosine_lr_and_sampling import call_cli, space_snapshot
from benchmarks.phase19_ssd_engram import BINARY_PREFIX, prefix_checks
from nexus_expanse import cli, ssd_engram
from nexus_expanse.ssd_engram import SSDEngramStore
from nexus_expanse.unified import NexusExpanseUnified


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


def test_binary_directory_counts_across_chunks_and_resets_at_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ssd_engram, "READ_SIZE", 7)
    files = (b"\x00\xff\x80\x00" * 11, b"\x19\x10\xff\x00" * 15)
    root = tmp_path / "input"
    root.mkdir()
    for i, raw in enumerate(files):
        (root / f"{i}.bin").write_bytes(raw)
    expected = {}
    for raw in files:
        for end in range(4, len(raw)):
            for order in range(4, min(end, 8) + 1):
                counts = expected.setdefault(raw[end - order:end], Counter())
                counts[raw[end]] += 1
    first, second = (tmp_path / f"{name} %.sqlite" for name in ("a", "b"))
    reports = [SSDEngramStore.build(root, path, max_order=8, min_support=1) for path in (first, second)]
    assert reports[0] == reports[1]
    assert reports[0]["entry_count"] == len(expected)
    with SSDEngramStore(first, expected_sha256=reports[0]["index_sha256"]) as store:
        for prefix, counts in expected.items():
            gain, match = store.lookup(prefix)
            assert gain > 0
            assert match.order == len(prefix) and dict(match.counts) == counts
            assert match.support == sum(counts.values())
        assert store.lookup(files[0][-4:]+files[1][:2]) == (0.0, None)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store._database.execute("DELETE FROM contexts")


def test_longest_supported_match_and_exact_probability_blend(tmp_path):
    root = tmp_path / "fit"
    root.mkdir()
    for i, raw in enumerate((b"xaabcdZ", b"yaabcdZ", b"zXabcdY")):
        (root / str(i)).write_bytes(raw)
    path = tmp_path / "counts.sqlite"
    SSDEngramStore.build(root, path, max_order=6)
    with SSDEngramStore(path) as store:
        g, longest = store.lookup(b"longer xaabcd")
        lower_g, fallback = store.lookup(b"xaabcd", min_support=3)
        assert (longest.order, longest.support, longest.counts) == (5, 2, ((ord("Z"), 2),))
        assert (fallback.order, fallback.support, fallback.counts) == (4, 3, ((ord("Y"), 1), (ord("Z"), 2)))
        assert g == 0.35 * (5 / 6)**4 * (2 / 6) and 0 < lower_g < g
        assert store.lookup(b"xaabcd", min_support=4) == (0.0, None)
        with pytest.raises(ValueError, match="build floor"):
            store.lookup(b"xaabcd", min_support=1)
        base = torch.full((256,), 1 / 256, dtype=torch.float64)
        assert cli.engram_forecast(base, b"missing", store, 0.35)[0] is base
        assert cli.engram_forecast(base, b"xaabcd", store, 0)[0] is base
        augmented, gain = cli.engram_forecast(base, b"xaabcd", store, 0.35)
        expected = base * (1 - gain)
        expected[ord("Z")] += gain
        assert torch.equal(augmented, expected / expected.sum())
        assert augmented.sum().item() == pytest.approx(1.0)


def test_invalid_metadata_sha_and_fuse_fallback(tmp_path, monkeypatch):
    raw = b"abc\xff\x00" * 12
    source = tmp_path / "source.bin"
    source.write_bytes(raw)
    path = tmp_path / "fit.sqlite"

    def fuse(*_args):
        raise PermissionError(errno.EPERM, "FUSE disallows hardlinks")

    monkeypatch.setattr(ssd_engram.os, "link", fuse)
    report = SSDEngramStore.build(source, path)
    before = path.read_bytes()
    assert report["index_sha256"] == hashlib.sha256(before).hexdigest()
    assert report["source_sha256"] == hashlib.sha256(raw).hexdigest()
    with pytest.raises(FileExistsError):
        SSDEngramStore.build(source, path)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        SSDEngramStore(path, expected_sha256="0" * 64)
    with sqlite3.connect(path) as database:
        database.execute("UPDATE metadata SET schema = 'invalid'")
    with pytest.raises(ValueError, match="invalid engram metadata"):
        SSDEngramStore(path)
    assert not list(tmp_path.glob("*.partial"))


def test_racing_publication_preserves_existing_destination(tmp_path, monkeypatch):
    source, index = tmp_path / "input.bin", tmp_path / "target.sqlite"
    source.write_bytes(b"train on this" * 6)

    def competing_writer(_, path):
        Path(path).write_bytes(b"other writer")
        raise OSError(errno.ENOTSUP, "links unsupported")

    monkeypatch.setattr(ssd_engram.os, "link", competing_writer)
    with pytest.raises(FileExistsError):
        SSDEngramStore.build(source, index)
    assert index.read_bytes() == b"other writer"
    assert not index.with_suffix(".sqlite.partial").exists()


def test_active_engram_is_bit_exact_under_100_prefix_perturbations(tmp_path):
    model = NexusExpanseUnified.restore(space_snapshot(19), fast_decode=True)
    assert prefix_checks(tmp_path, model) == dict(seeds=100, active=100, maximum=0.0, failures=[])


@pytest.mark.parametrize("quant", ("fp32", "fp16", "int8"))
def test_cli_generation_reuses_snapshot_paths_and_forecasts_exact_bytes(tmp_path, quant):
    source = tmp_path / "train.bin"
    prompt, completion = BINARY_PREFIX[:12], b"\xff\x00\x80\x19"
    source.write_bytes((prompt + completion) * 40)
    path = tmp_path / "fit.sqlite"
    report = call_cli("build-engram", "--input", source, "--out", path)
    assert report["index_bytes"] == path.stat().st_size and report["entry_count"] > 0
    assert report["donor_forward_calls"] == 0
    model = NexusExpanseUnified.restore(space_snapshot(19))
    snapshot = tmp_path / f"{quant}.pt"
    snapshot.write_bytes(model.snapshot(snapshot_quant=quant))
    common = ("generate", "--snapshot", snapshot, "--device", "cpu", "--always-wake", "--max-bytes", 4,
              "--prompt-hex", prompt.hex())
    index = ("--engram-index", path, "--engram-index-sha256", report["index_sha256"])
    off = call_cli(*common)
    assert "engram_hits" not in off
    zero = call_cli(*common, *index, "--engram-alpha", "0")
    assert zero["generated_hex"] == off["generated_hex"] and zero["engram_hits"] == 0
    on = call_cli(*common, *index, "--engram-alpha", ".9")
    assert on["generated_hex"] == completion.hex()
    assert on["engram_hits"] == 4 and on["engram_hit_rate"] == 1
    assert on["always_wake"] and on["engram_index_sha256"] == report["index_sha256"]
    assert on["donor_forward_calls"] == 0


def test_engram_cuda_dispatch_with_cpu_fixture(tmp_path, monkeypatch):
    source, index, snapshot = (tmp_path / name for name in ("source.bin", "store.sqlite", "weights.pt"))
    source.write_bytes(BINARY_PREFIX * 12)
    SSDEngramStore.build(source, index)
    snapshot.write_bytes(space_snapshot(8))
    restore = cli.NexusExpanseUnified.restore
    devices = []

    def capture(raw, **options):
        devices.append(options.pop("decode_device"))
        return restore(raw, **options)

    monkeypatch.setattr(cli.scaling, "device_for", lambda device, dtype: torch.device(device))
    monkeypatch.setattr(cli.NexusExpanseUnified, "restore", capture)
    output = call_cli("generate", "--snapshot", snapshot, "--device", "cuda", "--prompt-hex", BINARY_PREFIX.hex(),
                      "--engram-index", index, "--max-bytes", 1)
    assert devices == [torch.device("cuda")] and output["device"] == "cuda" and output["engram_hits"] == 1
