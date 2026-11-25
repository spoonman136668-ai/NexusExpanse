"""Deterministic raw corpus and compatibility with the training CLI."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from nexus_expanse.cli import corpus_paths
from scripts.prepare_gpu_corpus import assert_disjoint, grams, prepare


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_gpu_corpus.py"


def test_small_corpus_is_reproducible_disjoint_and_ready_for_cli(tmp_path):
    manifests = []
    for folder in (tmp_path / "first", tmp_path / "second"):
        run = subprocess.run(
            [sys.executable, str(SCRIPT), "--out-dir", str(folder), "--size-mb", "0.05", "--seed", "1501"],
            check=True, text=True, capture_output=True,
        )
        manifest = json.loads(run.stdout)
        assert manifest == json.loads((folder / "corpus_manifest.json").read_text())
        assert manifest["total_bytes"] == 50_000
        assert manifest["train_eval_32byte_overlap"] == 0
        paths = corpus_paths(folder)
        assert paths.keys() == manifest["files"].keys()
        assert sum(path.stat().st_size for path in paths.values()) == 50_000
        for name, path in paths.items():
            raw = path.read_bytes()
            assert manifest["files"][name] == dict(
                file=path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
            )
        for train in (path for name, path in paths.items() if name.startswith("train_")):
            all_train = set(grams(train.read_bytes()))
            for evaluation in (path for name, path in paths.items() if name.startswith("eval_")):
                assert all_train.isdisjoint(grams(evaluation.read_bytes()))
        manifests.append(manifest)
    assert manifests[0] == manifests[1]


def test_pack_rejects_overlap_and_existing_files(tmp_path):
    with pytest.raises(ValueError, match="32-byte window"):
        assert_disjoint([b"!" + b"x" * 32], {"prose-eval.bin": b"?" + b"x" * 32})
    existing = tmp_path / "math-eval.bin"
    existing.write_bytes(b"existing data")
    with pytest.raises(FileExistsError):
        prepare(tmp_path, 50_000, 11)
    assert set(tmp_path.iterdir()) == {existing}
    assert existing.read_bytes() == b"existing data"
