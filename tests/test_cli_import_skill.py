"""Unit tests for `nexus_expanse.cli import-skill`."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from nexus_expanse.donor_assimilation import DenseTokenDonorLM


def _make_compatible_donor(path: Path, seed: int = 21) -> None:
    torch.manual_seed(seed)
    donor = DenseTokenDonorLM(256, 32, 2, 4, 2, 64)
    torch.save(dict(weights=donor.state_dict(), byte_to_row=torch.arange(256)), path)


def test_cli_import_skill_width1_and_width2():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        donor = tmp / "donor.pt"
        _make_compatible_donor(donor)
        for wf in (1, 2):
            out = tmp / f"skill_w{wf}.pt"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "nexus_expanse.cli",
                    "import-skill",
                    "--donor",
                    str(donor),
                    "--method",
                    "residual8",
                    "--width-factor",
                    str(wf),
                    "--out",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(proc.stdout.strip())
            assert report["command"] == "import-skill"
            assert report["status"] == "PASS"
            assert report["donor_forward_calls"] == 0
            assert report["width_factor"] == wf
            assert report["prefix_max_abs_diff"] == 0.0
            assert report["round_trip_max_abs_diff"] == 0.0
            assert out.exists()
