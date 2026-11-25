"""Six fresh CPU subprocesses exercising the packaged developer workflow."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from nexus_expanse.checkpoint_transport import ImportedSkill
from nexus_expanse.donor_assimilation import DenseTokenDonorLM
from nexus_expanse.multi_skill_router import MultiSkillBank, save_bank


ROOT = Path(__file__).resolve().parents[1]


def _invoke(*args):
    process = subprocess.run(
        [sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert process.returncode == 0, f"{args}\n{process.stdout}\n{process.stderr}"
    return [json.loads(line) for line in process.stdout.splitlines() if line.startswith("{")]


def test_six_command_cli_pipeline(tmp_path):
    cli = ("-m", "nexus_expanse.cli")
    data = tmp_path / "corpus"
    (manifest,) = _invoke(
        "scripts/prepare_gpu_corpus.py", "--out-dir", data, "--size-mb", "0.05", "--seed", 42,
    )
    assert manifest["total_bytes"] == 50_000
    assert manifest["train_eval_32byte_overlap"] == 0
    assert manifest == json.loads((data / "corpus_manifest.json").read_text())

    snapshot = tmp_path / "unified.pt"
    training = tmp_path / "training"
    trained = _invoke(
        *cli, "train", "--corpus-dir", data, "--directory", training, "--device", "cpu",
        "--dtype", "fp32", "--dim", 32, "--context", 16, "--epochs", 1, "--steps-per-epoch", 1,
        "--batch-per-domain", 1, "--eval-blocks", 1, "--epoch-limit", 1,
        "--save-snapshot", snapshot,
    )[-1]
    assert trained["command"] == "train" and trained["exact_state"]
    assert trained["round_trip_probabilities"] == 0.0
    receipt = json.loads((training / "gate17.json").read_text())
    assert receipt["cold_equal"]
    for logical_name, item in manifest["files"].items():
        split, domain = logical_name.split("_")
        assert receipt["identity"]["corpora"][split][domain] == {
            "bytes": item["bytes"], "sha256": item["sha256"],
        }
    sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert trained["snapshot_sha256"] == sha

    (verified,) = _invoke(*cli, "verify", "--snapshot", snapshot, "--snapshot-sha256", sha)
    assert verified["status"] == "PASS" and verified["exact_state"]
    assert verified["snapshot_sha256"] == sha and verified["round_trip_probabilities"] == 0.0
    assert all(value == 0.0 for value in verified["parity"].values())

    # Import consumes a static donor pack. The unified training snapshot has its own format.
    donor_path = tmp_path / "donor.pt"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        donor = DenseTokenDonorLM(256, 32, 2, 4, 2, 64)
        torch.save(dict(weights=donor.state_dict(), byte_to_row=torch.arange(256)), donor_path)
        del donor

    skill_path = tmp_path / "skill.pt"
    (imported,) = _invoke(*cli, "import-skill", "--donor", donor_path, "--method", "monarch", "--out", skill_path)
    assert imported["status"] == "PASS" and imported["donor_forward_calls"] == 0
    assert imported["prefix_max_abs_diff"] == imported["round_trip_max_abs_diff"] == 0.0
    assert imported["skill_sha256"] == hashlib.sha256(skill_path.read_bytes()).hexdigest()

    package = torch.load(skill_path, weights_only=True, map_location="cpu")
    skill = ImportedSkill(width=package["width"])
    skill.load_state_dict(package["state"])
    bank_path = tmp_path / "bank.pt"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        save_bank(MultiSkillBank(ImportedSkill(), {"imported": skill}), bank_path)
    (routed,) = _invoke(
        *cli, "route-stream", "--bank", bank_path, "--router", "hmm_bma",
        "--input-hex", "000102", "--input-hex", "fdfeff",
    )
    assert routed["router"] == "hmm_bma"
    assert routed["prefix_max_abs_diff"] == 0.0 and routed["donor_forward_calls"] == 0
    assert [chunk["end_offset"] for chunk in routed["chunks"]] == [3, 6]
    for chunk in routed["chunks"]:
        assert set(chunk["routing_weights"]) == {"base_python", "imported"}
        assert sum(chunk["routing_weights"].values()) == pytest.approx(1.0)
        assert 0 <= chunk["next_byte"] <= 255

    (solved,) = _invoke(*cli, "solve", "--source-hex", "010203", "--target-hex", "040302", "--depth", 2)
    assert solved["solved"] and solved["output_hex"] == "040302" and solved["residual_hex"] == "000000"
