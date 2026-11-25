"""Portable file publication for the resumable DEV benchmarks."""
import hashlib
import json
import os

import torch


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def atomic(path, value):
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("x") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.link(partial, path)
    partial.unlink()


def save_pt(path, value):
    partial = path.with_suffix(".partial")
    if path.exists():
        raise FileExistsError(path)
    with partial.open("xb") as out:
        torch.save(value, out)
        out.flush()
        os.fsync(out.fileno())
    os.link(partial, path)
    partial.unlink()
