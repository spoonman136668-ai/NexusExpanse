#!/usr/bin/env python3
"""Gate 6B — O(T) Kuramoto-Monarch Streaming Benchmark on frozen Gate 4B/5B corpus."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.gate4b_dev import MANIFEST, read_split
from nexus_expanse.arbitration import ReliabilityArbitrator
from nexus_expanse.consolidation import ConsolidatingMemory
from nexus_expanse.kuramoto import KuramotoMonarchByteCore
from nexus_expanse.patching import CompactBigramPredictor


def verify_kuramoto_exactness(seeds: int = 20) -> Dict[str, object]:
    """Verify 0.0 bit-exact streaming-vs-batch AND prefix invariance."""
    max_stream_diff = 0.0
    max_prefix_diff = 0.0
    sample = torch.tensor(list(b"def kuramoto_step(z, x): return z\n"), dtype=torch.long)[None]
    for s in range(seeds):
        torch.manual_seed(s)
        m = KuramotoMonarchByteCore(n_layers=2, dim=32, ffn_mult=2, enable_skip=False, dropout=0.0).eval()
        with torch.inference_mode():
            full_logits = m.forward_neural(sample)[0]
            pref_logits = m.forward_neural(sample[:, :13])[0]
            max_prefix_diff = max(max_prefix_diff, float((full_logits[:, :13] - pref_logits).abs().max()))

            # Compare O(1) step-by-step streaming probabilities against full_logits softmax
            states = m.init_stream_state(1)
            for idx in range(sample.shape[1]):
                p_step, states = m.step_streaming(int(sample[0, idx].item()), states)
                p_batch = full_logits[0, idx].double().softmax(0)
                max_stream_diff = max(max_stream_diff, float((p_step - p_batch).abs().max()))
    return {
        "seeds": seeds,
        "max_prefix_diff": max_prefix_diff,
        "max_stream_vs_batch_diff": max_stream_diff,
    }


def train_kuramoto(seed: int, train_spans: List[bytes], dim: int = 64, epochs: int = 12) -> KuramotoMonarchByteCore:
    torch.manual_seed(seed)
    model = KuramotoMonarchByteCore(n_layers=2, dim=dim, ffn_mult=2, enable_skip=False, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    spans = [torch.tensor(list(s), dtype=torch.long)[None] for s in train_spans if len(s) > 1]
    model.train()
    for _ in range(epochs):
        for tok in spans:
            opt.zero_grad(set_to_none=True)
            logits = model.forward_neural(tok[:, :-1])[0]
            loss = F.cross_entropy(logits.reshape(-1, 256), tok[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model.eval()


def eval_kuramoto_policy(
    model: KuramotoMonarchByteCore,
    train_spans: List[bytes],
    eval_spans: List[bytes],
    policy: str,
    online: bool,
) -> Dict[str, object]:
    """Evaluate Kuramoto O(1) streaming or Kuramoto + compact BytePatchReader."""
    mem = ConsolidatingMemory(40000, "sparse_support")
    for span in train_spans:
        for i in range(1, len(span)):
            mem.observe_transition(span[max(0, i - 32) : i], span[i])

    bigram_counts = np.zeros((256, 256), dtype=np.int64)
    for span in train_spans:
        for i in range(len(span) - 1):
            bigram_counts[span[i], span[i + 1]] += 1
    compact = CompactBigramPredictor.from_counts(bigram_counts)
    min_prob = 2.0 ** -4.0

    arb = ReliabilityArbitrator()
    total_bits = 0.0
    count = 0
    deep_calls = 0

    t0 = time.perf_counter()
    with torch.inference_mode():
        for span in eval_spans:
            states = model.init_stream_state(1)
            prev_byte = None
            for i in range(len(span) - 1):
                obs = span[i]
                target = span[i + 1]

                if policy == "kuramoto_stream":
                    p_neural, states = model.step_streaming(obs, states)
                    deep_calls += 1
                else:
                    # kuramoto_patch: wake Kuramoto only on high-surprise bytes
                    wake = prev_byte is None or (compact.probability(prev_byte, obs) < min_prob)
                    prev_byte = obs
                    if wake:
                        p_neural, states = model.step_streaming(obs, states)
                        deep_calls += 1
                    else:
                        p_neural = torch.from_numpy(compact.forecast(obs))

                ctx = span[max(0, i + 1 - 32) : i + 1]
                p_mem, _ = mem.predict_probs(ctx)
                p_mix, _ = arb.mix(p_neural, p_mem)
                total_bits += -math.log2(max(float(p_mix[target].item()), 1e-300))
                count += 1

                arb.update(float(p_neural[target].item()), float(p_mem[target].item()))
                if online:
                    mem.observe_transition(ctx, target)
    elapsed = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "policy": policy,
        "mode": "online" if online else "static",
        "bpb": round(total_bits / count, 4),
        "bytes": count,
        "ms_per_byte": round((elapsed / count) * 1000.0, 4),
        "deep_calls": deep_calls,
        "deep_fraction": round(deep_calls / count, 4),
        "peak_rss_mb": round(rss_mb, 2),
    }


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)

    parity = verify_kuramoto_exactness(20)
    assert parity["max_prefix_diff"] == 0.0
    assert parity["max_stream_vs_batch_diff"] == 0.0

    data = read_split()
    train_spans = data["train"]
    eval_spans = data["eval"]

    runs = []
    for seed in (42, 43, 44):
        model = train_kuramoto(seed, train_spans, dim=64, epochs=12)
        arms = [
            eval_kuramoto_policy(model, train_spans, eval_spans, policy, online)
            for policy in ("kuramoto_stream", "kuramoto_patch")
            for online in (False, True)
        ]
        runs.append({"seed": seed, "param_bytes": model.total_param_bytes(), "arms": arms})

    # Aggregate 3-seed means per (policy, mode)
    summary = {}
    for policy in ("kuramoto_stream", "kuramoto_patch"):
        for mode in ("static", "online"):
            vals = [a for r in runs for a in r["arms"] if a["policy"] == policy and a["mode"] == mode]
            summary[f"{policy}_{mode}"] = {
                "mean_bpb": round(sum(v["bpb"] for v in vals) / len(vals), 4),
                "mean_ms_per_byte": round(sum(v["ms_per_byte"] for v in vals) / len(vals), 4),
                "mean_deep_fraction": round(sum(v["deep_fraction"] for v in vals) / len(vals), 4),
                "max_rss_mb": max(v["peak_rss_mb"] for v in vals),
            }

    # Gate 5B DirectMonarch cached always-on baselines: 4.9391 static / 4.4062 online, 3.821 ms/byte online
    ref_static_bpb = 4.9391
    ref_online_bpb = 4.4062
    ref_ms = 3.821
    best_static = summary["kuramoto_stream_static"]
    best_online = summary["kuramoto_stream_online"]
    speed_ok = best_static["mean_ms_per_byte"] <= ref_ms / 2.0 and best_online["mean_ms_per_byte"] <= ref_ms / 2.0
    bpb_ok = (best_static["mean_bpb"] <= ref_static_bpb + 0.05) and (best_online["mean_bpb"] <= ref_online_bpb + 0.05)
    status = "PASS" if (speed_ok and bpb_ok) else ("MIXED" if (speed_ok or bpb_ok) else "FAIL")

    report = {
        "schema": "nexus-gate6b-kuramoto-dev-v1",
        "qualification": "REUSED_MANIFEST_DEV",
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "torch": str(torch.__version__),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "parity": parity,
        "gate5b_ref_direct_monarch_cached": {
            "static_bpb": ref_static_bpb,
            "online_bpb": ref_online_bpb,
            "online_ms_per_byte": ref_ms,
        },
        "summary": summary,
        "status": status,
        "runs": runs,
    }
    out_path = Path(sys.argv[1])
    with out_path.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")

    print(f"Kuramoto Parity: {parity}")
    for k, v in summary.items():
        print(
            f"{k:26s} | mean_bpb={v['mean_bpb']:.4f} | mean_ms/byte={v['mean_ms_per_byte']:.4f} "
            f"({ref_ms / max(v['mean_ms_per_byte'], 1e-6):.2f}x vs Gate5B Monarch) | deep={v['mean_deep_fraction']*100:.1f}%"
        )
    print(f"OVERALL GATE 6B STATUS: {status}")


if __name__ == "__main__":
    main()
