import subprocess
import sys

from benchmarks.gate24_cli_smoke import ROOT, run_smoke


def test_all_cli_subcommands_in_fresh_processes(tmp_path):
    report = run_smoke(tmp_path)
    assert report["status"] == "PASS"
    assert {s["args"][0] for s in report["steps"]} == {"train", "verify", "stream", "solve"}
    assert report["stream"][-1]["exact_state"]


def test_bad_snapshot_digest_fails_before_stream_changes(tmp_path):
    untrusted = tmp_path / "data.pt"
    destination = tmp_path / "out.pt"
    untrusted.write_bytes(b"invalid checkpoint bytes")
    result = subprocess.run([sys.executable, "-m", "nexus_expanse.cli", "stream", "--snapshot",
                             str(untrusted), "--snapshot-sha256", "0" * 64, "--input-hex", "00",
                             "--save-snapshot", str(destination)], cwd=ROOT, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 2
    assert "snapshot SHA-256 mismatch" in result.stderr
    assert not destination.exists()
