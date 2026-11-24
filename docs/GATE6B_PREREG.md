# Gate 6B — O(T) Kuramoto Phase-Coupled Oscillator Mixer (Frozen DEV Protocol)

Replace quadratic `O(T^2)` `MultiheadAttention` + rolling KV-cache rebuilds with `KuramotoMonarchByteCore` (`nexus_expanse/kuramoto.py`, `K = dim // 2` unit-norm 2D phase oscillators on `S^1`, O(1) per-byte streaming state, zero KV cache, zero window rollovers, feeding `DirectMonarchLinear` FFNs).

Evaluate on the exact frozen Gate 4B/5B real code-to-prose manifest (`docs/GATE4B_MANIFEST.json`, 31 Python training spans, 8 held-out Markdown evaluation spans, 2,040 scored transitions, `seeds 42, 43, 44`, `dim=64`, `n_layers=2`, `ffn_mult=2`, `epochs=12`, matched optimizer/steps) against Gate 5B's recorded baselines (`DirectMonarch cached always-on`: `3.821 ms/byte`, `4.9391` frozen / `4.4062` prequential BPB; `Dense cached always-on`: `3.117 ms/byte`, `4.8955` / `4.4044 BPB`; `Memory only`: `0.189 ms/byte`, `4.6971` / `4.3925 BPB`).

Test two Kuramoto streaming policies (both fused via `ReliabilityArbitrator` with bounded `ConsolidatingMemory` matching Gate 5B):
1. `kuramoto_stream`: O(1) per-byte Kuramoto recurrence (zero KV cache).
2. `kuramoto_patch`: O(1) Kuramoto recurrence + compact `BytePatchReader` (`5,128 B` bigram wake/fold).

Pre-declared disposition criteria:
- **Engineering Gate:** `0.0` bit-exact streaming-vs-batch parity AND `0.0` bit-exact prefix invariance across 20 seeds.
- **Scientific PASS:** Mean real wall-clock `ms/byte` is **`>= 2.0x` faster** than Gate 5B `DirectMonarch cached always-on` (`<= 1.91 ms/byte` in both static and online modes) AND frozen & prequential BPB stay within **`<= +0.05 BPB`** of `DirectMonarch cached always-on` on all 3 seeds.
- **MIXED:** Meets speed bar (`>= 2.0x` faster) OR BPB bar (`<= +0.05 BPB`), but not both.
- **FAIL:** Fails both bars.

No post-result threshold tuning or code changes between preregistration commit and evidence commit.
