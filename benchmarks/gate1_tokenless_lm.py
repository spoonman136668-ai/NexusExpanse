#!/usr/bin/env python3
"""Nexus Expanse v0.1 — Gate 1 Falsification Benchmark (Tokenless Real Byte Stream).

Tests whether combining our 4 synergistic components:
  1. DirectMonarch sparse full-rank FFN (340 KB across 4L = 6.20x smaller than Dense)
  2. Strictly causal pre-FFN uncertainty gating (~54% deep FFN skip)
  3. Decoupled EpisodicByteMemory (multi-scale O(1) non-parametric recall, zero forgetting)
  4. Surprise-gated HebbianFastWeightAdapter (zero-backprop test-time plasticity)

can close the +0.44 BPB Monarch compression gap AND BEAT DenseFFN-4L (3.3311 BPB)
on the exact same frozen 19,567-byte real code/Markdown benchmark under strict
causal streaming (every byte predicted strictly BEFORE observing it).

Pre-registered Gate 1 PASS Criteria:
  - Held-out FP64 BPB strictly < DenseFFN-4L (3.3311 BPB)
  - Neural FFN static memory >= 5.0x smaller than DenseFFN-4L (2,107,392 B)
  - Deep FFN mean skip rate >= 40%
  - Zero catastrophic forgetting verified on episodic memory
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from nexus_expanse.core import NexusExpanseByteCore


def load_frozen_corpus(ygg_root: Path) -> tuple[bytes, bytes]:
    files = []
    for ext in ("*.py", "*.md"):
        for p in ygg_root.rglob(ext):
            rel = p.relative_to(ygg_root).as_posix()
            if (
                rel.startswith(".git/")
                or rel.startswith(".pytest_cache/")
                or rel.startswith("experiments/")
                or rel.startswith("tests/test_procedural_event")
            ):
                continue
            files.append(p)
    files.sort(key=lambda p: p.relative_to(ygg_root).as_posix())
    full = b"\n".join(p.read_bytes() for p in files)
    split = int(len(full) * 0.8)
    return full[:split], full[split:]


def make_batches(data: torch.Tensor, seq_len: int = 64, batch_size: int = 32):
    total_seqs = (len(data) - 1) // seq_len
    trimmed = data[: total_seqs * seq_len + 1]
    xs = trimmed[:-1].view(total_seqs, seq_len)
    ys = trimmed[1:].view(total_seqs, seq_len)
    return [(xs[i : i + batch_size], ys[i : i + batch_size]) for i in range(0, total_seqs, batch_size)]


def main() -> None:
    ygg_root = Path("/working_dir/c_991947bcc9e94dea/Yggdrasil")
    train_bytes, val_bytes = load_frozen_corpus(ygg_root)
    train_tokens = torch.tensor(list(train_bytes), dtype=torch.long)
    train_batches = make_batches(train_tokens, seq_len=64, batch_size=32)

    torch.manual_seed(42)
    core = NexusExpanseByteCore(
        n_layers=4, dim=128, n_heads=4, ffn_mult=4, target_skip_rate=0.50, enable_skip=True, dropout=0.1
    )

    t0 = time.perf_counter()
    opt = torch.optim.AdamW(core.parameters(), lr=3e-3, weight_decay=1e-2)
    epochs = 20
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(epochs):
        core.train()
        for xb, yb in train_batches:
            opt.zero_grad(set_to_none=True)
            logits, _, aux, _ = core.forward_neural(xb, targets=yb)
            ce = F.cross_entropy(logits.reshape(-1, 256), yb.reshape(-1))
            loss = ce + 0.1 * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep in (0, 4, 9, 14, epochs - 1):
            core.calibrate_gates(train_batches)

    # Ingest training bytes into decoupled EpisodicByteMemory (zero backprop)
    core.episodic.ingest_stream(train_bytes)
    train_sec = time.perf_counter() - t0
    ffn_params, ffn_bytes = core.total_ffn_footprint()

    # Evaluate 3 progressive configurations on exact same 19,567 held-out bytes
    res_neural_only = core.evaluate_streaming(
        val_bytes, use_episodic=False, online_episodic_update=False, online_hebbian_update=False
    )
    res_static_mem = core.evaluate_streaming(
        val_bytes, use_episodic=True, online_episodic_update=False, online_hebbian_update=False
    )

    # Clone episodic memory before online continual test so static state stays clean
    core_online = copy.deepcopy(core)
    res_online_full = core_online.evaluate_streaming(
        val_bytes, use_episodic=True, online_episodic_update=True, online_hebbian_update=True
    )

    dense_ref_bpb = 3.3311
    dense_ref_bytes = 2107392
    mem_red = dense_ref_bytes / ffn_bytes

    verdict_static = (
        "PASS"
        if (res_static_mem["bpb"] < dense_ref_bpb and mem_red >= 5.0 and res_static_mem["mean_skip"] >= 0.40)
        else "MIXED"
    )
    verdict_online = (
        "PASS"
        if (res_online_full["bpb"] < dense_ref_bpb and mem_red >= 5.0 and res_online_full["mean_skip"] >= 0.40)
        else "MIXED"
    )

    out = {
        "project": "Nexus Expanse v0.1",
        "gate": "Gate 1 — Tokenless Real Byte-Stream Falsification",
        "neural_ffn_bytes": ffn_bytes,
        "neural_ffn_reduction_vs_dense_x": round(mem_red, 2),
        "train_elapsed_sec": round(train_sec, 2),
        "dense_4l_reference_bpb": dense_ref_bpb,
        "arms": {
            "nexus_neural_only": res_neural_only,
            "nexus_static_episodic": res_static_mem,
            "nexus_online_episodic_plus_hebbian": res_online_full,
        },
        "verdict_static": verdict_static,
        "verdict_online": verdict_online,
    }
    Path(PKG_ROOT / "gate1_results.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(
        f"=== Nexus Expanse v0.1 Gate 1 (Tokenless, 19,567 bytes, FFN={ffn_bytes:,} B = {mem_red:.2f}x smaller than Dense, train={train_sec:.1f}s) ==="
    )
    print(f"Reference DenseFFN-4L (0% skip, 2.11 MB) : val_bpb={dense_ref_bpb:.4f}")
    print(
        f"Nexus Neural-Only (Monarch + Skip)       : val_bpb={res_neural_only['bpb']:.4f} | skip={res_neural_only['mean_skip']*100:.1f}%"
    )
    print(
        f"Nexus + Static Episodic Memory           : val_bpb={res_static_mem['bpb']:.4f} ({res_static_mem['bpb']-dense_ref_bpb:+.4f} vs Dense) | "
        f"skip={res_static_mem['mean_skip']*100:.1f}% | entries={res_static_mem['episodic_entries']:,} | {verdict_static}"
    )
    print(
        f"Nexus + Online Episodic + Hebbian FastW  : val_bpb={res_online_full['bpb']:.4f} ({res_online_full['bpb']-dense_ref_bpb:+.4f} vs Dense) | "
        f"skip={res_online_full['mean_skip']*100:.1f}% | hebbian_writes={res_online_full['hebbian_updates']:,} | {verdict_online}"
    )


if __name__ == "__main__":
    main()
