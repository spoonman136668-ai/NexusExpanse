# Nexus Expanse

Independent research prototype: raw-byte causal attention, DirectMonarch FFNs, uncertainty-gated computation, episodic suffix memory and local fast weights. This is a byte-level attention architecture, not a non-transformer model. Do not integrate into other projects without separate authorization and qualification.

## Baseline custody

Original archive SHA-256: `241b0c365a2d46eea49e06cf5337f06798250b7561de5ccce9574c59f684119e`.
Original import: `bb22c229aebae264d5b85fcc5e5a16901bbfeae8`, tree `fce3c3d7fd89f2996a15bfbc4685115353552b62`.
`gate1_results.json` is imported historical evidence, not a reproduced result. The original benchmark uses an unavailable local Yggdrasil path and an external hard-coded dense score. It is retained unchanged.

## Engineering

Python with PyTorch and pytest. Run `python -m pytest -q`. The new qualification preflight only checks explicit train/validation byte files and emits their SHA-256 identities; it does not train or consume a held-out scientific trial.

`python benchmarks/qualification_preflight.py --train PATH --validation PATH --output NEW.json`

## Research status

See `docs/RESEARCH.md`. No checkpoint, frozen source corpus, independently reproduced dense control or scientific first-use authorization is present. Keep original evidence immutable. No deployment, external runner, or scheduled execution authorized by this import.

## Matched development ablation

`python benchmarks/ablation_dev.py --output /path/to/new-evidence.json` runs a CPU-only, three-seed comparison on embedded, known synthetic fixtures. It trains dense and Monarch variants with and without gating, matches non-FFN initialization and optimizer steps, and clones trained state for six online/static ablations per variant plus memory-only controls. It requires a clean Git tree and exclusive output creation. The results are development diagnostics, not held-out evidence, equal-FLOP comparisons or a reproduction of Gate 1. Reported tensor bytes exclude episodic Python storage. The original default model and benchmark are preserved; callers may now explicitly select a dense FFN and use frozen fast weights without resetting them.

## Opt-in bounded episodic memory

`from nexus_expanse import BoundedEpisodicByteMemory`; assign `model.episodic = BoundedEpisodicByteMemory(max_contexts=4096)` to opt in. The context budget evicts least recently observed contexts, not least recently read contexts. `memory.save(Path('new-memory.json'), max_bytes=2_000_000)` writes an exclusive-create versioned JSON snapshot. `BoundedEpisodicByteMemory.load(path, max_bytes=2_000_000, max_contexts=4096)` checks the externally supplied limits. It restores episodic state only, not neural parameters or Hebbian weights. The byte limit applies to serialized data, not live allocator size, and the context cap applies after each observation. Eviction can cause forgetting; do not confuse exact checkpoint replay with learned retention under interference. No deployment or automatic checkpointing is enabled.

## Opt-in adaptive arbitration

Pass a fresh `ReliabilityArbitrator(bins=1)` as `arbitrator` to `evaluate_streaming`, with `online_arbitration_update=True` to adapt after each scored byte. The optional four-bin version separates reliability by memory confidence. Both start at equal odds and use a fixed 1% share to permit recovery from a shift. A caller may keep the instance or freeze updates; arbitration state is not part of the model checkpoint. The original fixed blend remains the default; `mem_blend_cap` applies only to that rule. Run `python benchmarks/arbitration_dev.py --output NEW.json` on the known synthetic development fixture. No held-out data, Hebbian adaptation, or scientific first-use is included in this comparison.

## Stronger training development diagnostic

`python benchmarks/complementarity_dev.py --output NEW.json` runs a bounded 12-epoch, three-seed comparison of dense/Monarch with/without gating, on the prior synthetic fixture and a constructed complementary fixture. Adaptive fusion beats the standalone controls on the latter, but dense full leads and the original gated Monarch still does not beat memory alone in mean BPB. The 40-epoch attempt timed out without complete evidence. This diagnostic deliberately favors different experts in different regions; it is not a held-out trial or a context-matched strongest-control comparison. See `docs/RESEARCH.md` and `evidence/complementarity-dev-001.json`.

## Opt-in fast-weight checkpoint (CPU float32)

`save_fast_weights(model, path, max_bytes=40000)` and `load_fast_weights(model, path, max_bytes=40000)` preserve the bounded fast tensor and update count. Supply the exact same neural backbone, inference settings, adapter configuration, source implementation and PyTorch version on restore. The checksum is corruption detection, not authentication. The file is exclusively created; no automatic checkpoint or overwrite. Restore checks the externally supplied size budget, identity, finite values and clamp range before mutating state. After restore use `evaluate_streaming(..., use_fast_weights=True, reset_fast_state=False)`; the original default still resets fast weights. This is not a complete model, episodic memory, or arbitration checkpoint. GPU and cross-platform restore are unqualified. Use a quiescent model during save/restore; concurrent writers are unsupported. Run `python benchmarks/fast_retention_dev.py --output NEW.json` for the known synthetic A/B/A diagnostic. Cold checkpoint replay works, but interference persists; see the research notes.

### v0.6 — interference decomposition (development only)

Optional `ContextBankFastWeightAdapter` uses disjoint hidden-feature slices and causal suffix hashing with the same fast tensor capacity. It did not pass retention on the known A/B fixture. Shared fast weights with decay disabled passed provisional retention and B-learning thresholds in all three seeds; default behavior is unchanged. See `evidence/interference-dev-001.json` and `docs/RESEARCH.md` for controls, caveats and next steps. Source-bound v0.5 fast checkpoints are not automatically compatible with v0.6.

### v0.7 longitudinal development diagnostic

`python benchmarks/longitudinal_dev.py --output results/longitudinal-new.json` (create `results/` first; use a clean tree and new output path). Eight regimes, two acquisition passes, three seeds, two cue conditions and four controls. Target-byte retention gate **FAIL** for all arms; no-decay improves average target prediction but does not prevent forgetting. No observed clamping does not establish absence of long-term saturation. See `docs/RESEARCH.md` and `evidence/longitudinal-dev-001.json` for exact provenance, limitations and full trajectories. No change to model defaults.
