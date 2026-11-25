"""CLI cold-load, causal forecast alignment and JSON contracts for raw hex chunks."""
import json
import subprocess
import sys

import pytest
import torch

from nexus_expanse.checkpoint_transport import ImportedSkill
from nexus_expanse.multi_skill_router import MultiSkillBank, save_bank


@pytest.mark.parametrize("router", ("bayes_bma", "bayes_argmax", "surprise_sparse"))
def test_route_stream_round_trip_chunks(tmp_path, router):
    torch.manual_seed(1520)
    bank = MultiSkillBank(ImportedSkill(), {
        name: ImportedSkill() for name in ("rule_A", "rule_B", "markdown_prose")
    })
    snapshot = tmp_path / "bank.pt"
    save_bank(bank, snapshot)
    process = subprocess.run(
        [sys.executable, "-m", "nexus_expanse.cli", "route-stream", "--bank", str(snapshot),
         "--router", router, "--input-hex", "0001ff", "--input-hex", "f8100017"],
        check=True, capture_output=True, text=True,
    )
    report = json.loads(process.stdout)
    assert set(report) == {
        "command", "router", "prefix_max_abs_diff", "avg_active_skills_per_byte", "donor_forward_calls", "chunks"
    }
    assert report["command"] == "route-stream"
    assert report["router"] == router
    assert report["prefix_max_abs_diff"] == 0.0
    assert report["donor_forward_calls"] == 0
    assert [chunk["end_offset"] for chunk in report["chunks"]] == [3, 7]
    raw = torch.tensor([list(bytes.fromhex("0001fff8100017"))])
    forecasts, decisions, *calls = getattr(bank, f"forward_{router}")(raw, include_next=True)
    assert report["avg_active_skills_per_byte"] == (calls[0] if calls else 4 * 7 / 8)
    for chunk in report["chunks"]:
        assert set(chunk) == {"end_offset", "active_skill", "routing_weights", "next_byte"}
        offset = chunk["end_offset"]
        assert chunk["next_byte"] == int(forecasts[0, offset].argmax())
        assert 0 <= chunk["next_byte"] <= 255
        weights = chunk["routing_weights"]
        assert tuple(weights) == bank.skill_names
        assert sum(weights.values()) == pytest.approx(1)
        if router == "bayes_bma":
            assert list(weights.values()) == pytest.approx(decisions[0, offset].tolist(), abs=1e-6)
            assert chunk["active_skill"] == max(weights, key=weights.get)
        else:
            assert chunk["active_skill"] == bank.skill_names[int(decisions[0, offset])]
            assert weights[chunk["active_skill"]] == 1.0
