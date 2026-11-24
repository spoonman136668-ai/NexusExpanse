#!/usr/bin/env python3
"""Gate 6A — Holographic Vector-Symbolic Relational Memory on Gate 4A/5A fixture."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.gate4a_dev import SEEDS, cold, run, torch
from nexus_expanse.holographic_memory import HolographicRelationalMemory


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if len(sys.argv) >= 3 and sys.argv[1] == "--cold":
        cold(Path(sys.argv[2]), candidate_cls=HolographicRelationalMemory)
    else:
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
        runs = [run(seed, candidate_cls=HolographicRelationalMemory, cold_script=__file__) for seed in SEEDS]
        for r in runs:
            cand = r["arms"]["latent_wta"]
            sparse = r["arms"]["sparse_support"]
            tokenless_win = (
                cand["scores"]["unseen"]["accuracy"] >= 0.75
                and (sparse["scores"]["unseen"]["bpb"] - cand["scores"]["unseen"]["bpb"]) >= 2.0
            )
            r["tokenless_breakthrough"] = tokenless_win
        full_pass = all(r["status"] == "PASS" for r in runs)
        tokenless_pass = all(r["tokenless_breakthrough"] for r in runs)
        status = "PASS" if full_pass else ("MIXED" if tokenless_pass else "FAIL")
        out_path = Path(sys.argv[1])
        with out_path.open("x", encoding="utf-8") as out:
            json.dump(
                dict(
                    schema="nexus-gate6a-holographic-dev-v1",
                    qualification="DEV",
                    source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    torch=str(torch.__version__),
                    status=status,
                    runs=runs,
                ),
                out,
                allow_nan=False,
                indent=2,
            )
            out.write("\n")
        for r in runs:
            c = r["arms"]["latent_wta"]
            sp = r["arms"]["sparse_support"]
            sym = r["arms"]["symbolic_chain"]
            print(
                f"seed={r['seed']} | unseen_acc={c['scores']['unseen']['accuracy']*100:5.1f}% | "
                f"unseen_bpb={c['scores']['unseen']['bpb']:.4f} (sparse={sp['scores']['unseen']['bpb']:.4f}, sym={sym['scores']['unseen']['bpb']:.4f}) | "
                f"shifted_acc={c['scores']['shifted_primitive']['accuracy']*100:5.1f}% | "
                f"cold_acc={c['scores']['cold_return']['accuracy']*100:5.1f}% | "
                f"worst_drift={c['worst_drift']:.4f} | bytes={c['max_bytes']} | status={r['status']}"
            )
        print(f"OVERALL GATE 6A STATUS: {status}")
