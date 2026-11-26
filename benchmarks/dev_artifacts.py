"""Portable file publication for the resumable DEV benchmarks."""
import errno
import hashlib
import json
import os

import torch


def _publish(partial, path):
    try:
        os.link(partial, path)
    except OSError as exc:
        if not isinstance(exc, PermissionError) and exc.errno not in {errno.EPERM, errno.EXDEV, errno.ENOTSUP}:
            raise
        # The exclusively created partial still reserves the write during fallback.
        if os.path.lexists(path):
            raise FileExistsError(path) from exc
        os.replace(partial, path)
    else:
        partial.unlink()


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def atomic(path, value):
    if os.path.lexists(path):
        raise FileExistsError(path)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("x") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    _publish(partial, path)


def save_pt(path, value):
    partial = path.with_suffix(path.suffix + ".partial")
    if os.path.lexists(path):
        raise FileExistsError(path)
    legacy_partial = path.with_suffix(".partial")
    if os.path.lexists(legacy_partial):
        raise FileExistsError(legacy_partial)
    with partial.open("xb") as out:
        torch.save(value, out)
        out.flush()
        os.fsync(out.fileno())
    _publish(partial, path)


def atomic_bytes(path, payload):
    if os.path.lexists(path):
        raise FileExistsError(path)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("xb") as out:
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())
    _publish(partial, path)
