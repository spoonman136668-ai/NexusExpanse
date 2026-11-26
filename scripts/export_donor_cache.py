"""Export a local donor into immutable, checksummed raw-byte probability shards."""

import argparse
import json
import os
from pathlib import Path
import pickle
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus_expanse.donor_cache_io import EXPORT_DTYPES, export_donor_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nexus_donor", "hf"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="raw bytes in complete fixed-length windows")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window-bytes", type=int, choices=range(2, 65), default=64, metavar="2..64")
    parser.add_argument("--shard-windows", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4, help="attention heads for a DenseTokenDonorLM checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=EXPORT_DTYPES, default="fp32")
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        data = args.input.read_bytes()
        if not data or len(data) % args.window_bytes:
            raise ValueError("input must contain only complete raw-byte windows")
        windows = [data[i:i + args.window_bytes] for i in range(0, len(data), args.window_bytes)]
        manifest = export_donor_cache(
            args.out_dir, windows, backend=args.backend, checkpoint=args.model,
            shard_windows=args.shard_windows, n_heads=args.n_heads, device=args.device, dtype=args.dtype,
        )
    except (ValueError, KeyError, OSError, RuntimeError, pickle.UnpicklingError) as exc:
        parser.exit(2, f"donor cache export: {exc}\n")
    print(json.dumps(manifest, allow_nan=False))


if __name__ == "__main__":
    main()
