import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "qualification_preflight.py"
spec = importlib.util.spec_from_file_location("qualification_preflight", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_manifest_and_no_science(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"train")
    b.write_bytes(b"heldout")
    m = module.corpus_manifest(a, b)
    assert m["files"]["train"]["sha256"] == hashlib.sha256(b"train").hexdigest()
    assert m["files"]["validation"]["bytes"] == 7
    assert m["scientific_trial_executed"] is False
    assert m["overlap_audit"] == "NOT_PERFORMED"


@pytest.mark.parametrize("case", ["same_path", "identical", "empty", "missing"])
def test_reject_invalid_custody(tmp_path, case):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"train")
    if case == "same_path":
        b = a
    elif case == "identical":
        b.write_bytes(b"train")
    elif case == "empty":
        b.write_bytes(b"")
    with pytest.raises((ValueError, FileNotFoundError)):
        module.corpus_manifest(a, b)


def test_cli_does_not_overwrite(tmp_path):
    a, b, out = (tmp_path / name for name in ("a", "b", "manifest.json"))
    a.write_bytes(b"train")
    b.write_bytes(b"heldout")
    cmd = [sys.executable, str(SCRIPT), "--train", str(a), "--validation", str(b), "--output", str(out)]
    subprocess.run(cmd, check=True)
    original = out.read_bytes()
    assert subprocess.run(cmd, capture_output=True).returncode != 0
    assert out.read_bytes() == original
