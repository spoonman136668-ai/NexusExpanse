"""Engineering-only corpus custody preflight; no model training or scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def corpus_manifest(train: Path, validation: Path) -> dict:
    """Identify explicit byte files and reject trivial leakage/invalid input."""
    if train.resolve() == validation.resolve():
        raise ValueError("train and validation must be different files")
    records = {}
    for name, path in (("train", train), ("validation", validation)):
        data = path.read_bytes()
        if len(data) < 2:
            raise ValueError(f"{name} requires at least two bytes")
        records[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    if records["train"]["sha256"] == records["validation"]["sha256"]:
        raise ValueError("train and validation contents are identical")
    return {"schema": "nexus-corpus-preflight-v1", "files": records,
            "qualification": "ENGINEERING_PREFLIGHT_ONLY",
            "overlap_audit": "NOT_PERFORMED", "scientific_trial_executed": False}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--validation", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    manifest = corpus_manifest(args.train, args.validation)
    # Exclusive create preserves existing evidence, including the original report.
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
