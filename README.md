# Nexus Expanse v1.2.0-rc1

Nexus Expanse is an independent raw-byte research prototype. The release candidate includes a CPU streaming runtime, a bounded expression solver, resumable training and checkpoint assimilation. [Architecture](docs/ARCHITECTURE_V1.md) · [Checkpoint transfer findings](docs/CHECKPOINT_TRANSFER_FINDINGS.md) · [Owner Colab receipts](docs/LIVE_COLAB_RECEIPTS.md)

## Quick start

Use Python 3.12, NumPy, and PyTorch 2.10. Run from the repository root.

```sh
python -m nexus_expanse.cli verify --save-snapshot ../nexus-demo.pt
python -m nexus_expanse.cli stream --snapshot ../nexus-demo.pt --input-hex 48656c6c6f --continuation 8 --save-snapshot ../nexus-adapted.pt
python -m nexus_expanse.cli solve --source-hex 010203 --target-hex 040302 --depth 2
```

Omit `--input-hex` to submit hex byte chunks on stdin, one line at a time. The stream reports five next-byte candidates and optional greedy continuation. It learns from supplied observations. It generates continuations on a separate copy of the stream state. The snapshot is created at the end of the session, when requested. Supply `--snapshot-sha256` to check the digest from a previous verification or export. Use checkpoints from a trusted local source.

The solver searches one- to three-step pipelines over six fixed, pure AST expressions on raw bytes `00` through `07`. It reports grammar calibration separately from the six-call search budget and returns the checked byte residual. Use `--save-agent-snapshot` and `--agent-snapshot` to carry verified skills between sessions.

## CLI commands

Run `python -m nexus_expanse.cli <command> --help` for arguments and defaults.

| Command | What it does |
| --- | --- |
| `train` | Trains or warm-starts from a local skill, resumes from immutable epoch checkpoints and can export a verified unified snapshot. |
| `verify` | Checks prefix, streaming and snapshot round-trip parity. Accepts a snapshot and optional SHA-256. |
| `stream` | Observes raw hex byte chunks, reports next-byte candidates and optionally generates a continuation. |
| `generate` | Generates raw bytes from a unified snapshot with greedy or seeded top-p decoding on `cpu`, `cuda`, or `auto`. |
| `solve` | Searches a bounded expression library and checks the output against the target bytes. |
| `import-skill` | Converts a compatible static donor checkpoint into a verified native skill. |
| `route-stream` | Forecasts from a saved multi-skill bank using the selected router. |
| `assimilate` | Trains a native byte skill from static donor-cache shards. |
| `assimilate-zero-exec` | Initializes a native skill from local or streamed safetensors without running the donor. Offers configurable student width and depth. |
| `weight-edit` | Inspects or ablates an attention subspace in a saved skill. |

## Phase 13 warm-start and generation

```sh
python -m nexus_expanse.cli train --device cpu --corpus-dir ./data --directory ../nexus-warm --init-skill ../skill.pt --dim 64 --layers 4 --context 64 --until-complete --save-snapshot ../nexus-warm/unified.pt
python -m nexus_expanse.cli generate --snapshot ../nexus-warm/unified.pt --prompt-text "Hello" --max-bytes 128 --temperature 0.0 --top-p 0.9 --seed 42
```

Resume with the same initialization archive and training settings. The run binds the archive SHA-256 in epoch custody. Any stored static prior decays to zero during the first epoch. The exported unified snapshot contains the native model and its native memory state.

Use `--prompt-hex` for arbitrary bytes or `--prompt-text` for UTF-8. Generation returns JSON with exact output bytes in hex and a UTF-8 display string. Use a positive temperature for seeded nucleus sampling. Online adaptation is opt-in with `--adapt`. For up to 32 output bytes, default greedy generation matches `stream --no-adapt --continuation` from the same snapshot and prompt.

In the preregistered CPU DEV comparison, warm-start beat both scratch and shuffled initialization on pooled held-out BPB in 5/5 seeds. Independent scientific qualification remains open. [Preregistration](docs/PHASE13_WARMSTART_AND_GENERATE_PREREG.md) · [Unrounded results](evidence/phase13-warmstart-train-dev-001.json)

## Phase 14 fast generation and real corpus packaging

```sh
python -m nexus_expanse.cli generate --snapshot ../nexus-warm/unified.pt --prompt-text "Hello" --device cpu --max-bytes 128
python scripts/prepare_real_corpus.py --prose-dir docs --code-dir . --math-fallback-synthetic --size-mb 1.5 --out-dir ../nexus-real-corpus
```

Generation defaults to CPU. Use `--device auto` for CUDA when available. The JSON includes the selected device, prompt prefill time and decode time per byte. CPU greedy output matched `stream --no-adapt` in all five scored profiles. Decode was 18.71× faster at the median for those CPU rollover profiles. [Preregistration](docs/PHASE14_FAST_GENERATE_AND_REAL_CORPUS_PREREG.md) · [Unrounded results](evidence/phase14-fast-generate-dev-001.json)

Use the packaged directory as `train --corpus-dir`. The packager keeps input files or dataset records whole, separates train and eval, and verifies zero shared 32-byte windows across the six outputs. It records document counts, byte counts and SHA-256 digests in `corpus_manifest.json`. Supply `--math-dir` for real math text. The explicit fallback uses deterministic arithmetic records when local math is missing or insufficient. The optional `--hf-prose`, `--hf-code` and `--hf-math` flags accept `repo[:config]` and require the `datasets` package.

Phase 14B selects holdout documents from large pools using a seeded ranking based on shared 32-byte windows. A smaller pool can use a bounded search when ranking leaves too few bytes. Each candidate is a whole document, and publication requires an exact overlap check across all train and eval files. The offline regression covers 5,000 Hub-like records with repeated story prefixes and local files with shared imports.

## Phase 15 cosine training and repetition control

```sh
python -m nexus_expanse.cli train --corpus-dir ./data --directory ../nexus-cosine --init-skill ../skill.pt --dim 64 --layers 4 --context 64 --epochs 4 --steps-per-epoch 32 --lr 0.003 --lr-schedule cosine --until-complete --save-snapshot ../nexus-cosine/unified.pt
python -m nexus_expanse.cli generate --snapshot ../nexus-cosine/unified.pt --prompt-text "Hello" --seed 42 --temperature 0.7 --top-p 0.9 --repetition-penalty 1.5 --repeat-window 64
```

Training defaults to a constant rate of `0.003`. Cosine uses the committed optimizer step to resume the same schedule after a restart. Keep the rate and schedule fixed across invocations. Training now accepts widths from 32 through 512 in increments of 32, 2/4/6/8 layers, up to 1,024 steps per epoch, and up to 32 blocks per domain per batch. Offline checks cover constructing and restoring the largest supported snapshot geometry. Full-scale GPU throughput remains unverified.

Generation disables the penalty at `1.0`. Higher values scale the log scores of bytes present in the last `--repeat-window` bytes, including the prompt tail. The default window is 64 bytes and the supported range is 1–512. The five CPU DEV seeds passed exact resume and generation checks. Cosine lowered pooled held-out BPB in 2/5 seeds and raised the mean by 0.01255 BPB. [Preregistration](docs/PHASE15_COSINE_LR_AND_SAMPLING_PREREG.md) · [Raw results](evidence/phase15-cosine-lr-dev-001.json)

`generate --always-wake` keeps the neural path active for every prompt and generated byte. Use it when predictable transitions such as repeated indentation keep the default byte reader on its bigram fallback. It composes with the repetition penalty and reports `always_wake` in the generation JSON. With the flag omitted, default greedy generation retains byte-for-byte parity with `stream --no-adapt`.

## Release-candidate evidence

The local Linux CPU run using bare `pytest -q` passed **708/708** tests with zero failures, skips or warnings. The prefix benchmark reported **0/100** failures. The owner separately verified TinyStories Hub streaming and reported that forcing wake cleared the repeated-space loop on four Colab prompts. Independent scientific qualification, the live wikitext rerun, multi-GB GPU pretraining, full-scale 256×6 and 512×8 training on L4/A100 GPUs, and live Engram extraction remain open. [Owner Colab receipts](docs/LIVE_COLAB_RECEIPTS.md)

## Windows consumer-GPU launch

With CUDA-enabled PyTorch, place `math-train.bin`, `math-eval.bin`, `code-train.bin`, `code-eval.bin`, `prose-train.bin`, and `prose-eval.bin` in `data`. Use separate train and evaluation files. The example covers roughly one pass over up to 4 MiB per training domain. Use a local NTFS run directory outside the clean source checkout.

```powershell
python -m nexus_expanse.cli train --device cuda --dtype bf16 --corpus-dir .\data --directory ..\nexus-gpu-run --dim 64 --context 256 --batch-per-domain 4 --steps-per-epoch 64 --epochs 64 --epoch-limit 1 --until-complete --save-snapshot ..\nexus-gpu-run\unified.pt
```

Use `--dtype fp32` if the device lacks native BF16. Repeat an interrupted run with identical source, bytes, device and training settings to resume from the last fully committed epoch. The exact training receipt and unified CPU snapshot are written when training completes. The owner recorded a Windows FP32 512×8 run on an RTX 5070 Ti Laptop GPU. The v1.2.1 autocast fix has offline CPU coverage. A Windows CUDA BF16 rerun remains open. [Scaling protocol](docs/GATE17_PREREG.md) · [Owner Windows receipt](docs/LIVE_COLAB_RECEIPTS.md#windows-rtx-5070-ti-laptop-gpu-training-and-generation)

## Snapshot weight storage

Use `train --save-snapshot PATH --snapshot-quant fp16` or `int8` to reduce the exported snapshot size. The default `fp32` retains the existing archive bytes and reads as `fp32` in verification JSON. Compressed archives record their storage mode. FP16 stores matrix and higher-rank neural weights in half precision. INT8 uses symmetric per-output-channel scales. Biases, norms, adaptive memories, and the stream cursor retain their original precision. Both encodings restore to FP32 for inference, including `generate --always-wake`.

```sh
python -m nexus_expanse.cli verify --snapshot ../unified.pt --save-snapshot ../unified-int8.pt --snapshot-quant int8
```

The export reports exact parity for the stored representation after restore. Quantization can change probabilities relative to the source FP32 weights. The training export's `neural_state_sha256` identifies the original training checkpoint, and `snapshot_sha256` identifies the exported archive.

On the offline trained 256×6 fixture, the FP16 archive was 1.94× smaller and the INT8 archive was 3.63× smaller. The largest absolute held-out BPB change across the three domains was 0.000089 for FP16 and 0.002095 for INT8.

<details>
<summary>Historical research and API notes</summary>

## Baseline custody

Original archive SHA-256: `241b0c365a2d46eea49e06cf5337f06798250b7561de5ccce9574c59f684119e`.
Original import: `bb22c229aebae264d5b85fcc5e5a16901bbfeae8`, tree `fce3c3d7fd89f2996a15bfbc4685115353552b62`.
`gate1_results.json` is imported historical evidence, not a reproduced result. The original benchmark uses an unavailable local Yggdrasil path and an external hard-coded dense score. It is retained unchanged.

## Engineering

Python with PyTorch and pytest. Run `python -m pytest -q`. The new qualification preflight only checks explicit train/validation byte files and emits their SHA-256 identities; it does not train or consume a held-out scientific trial.

`python benchmarks/qualification_preflight.py --train PATH --validation PATH --output NEW.json`

## Research status

The initial import lacked a checkpoint, frozen source corpus, independently reproduced dense control and scientific first-use authorization. Keep original evidence immutable. [Historical research notes](docs/RESEARCH.md)

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

### v0.8 — selective consolidation (development only)

`nexus_expanse.consolidation.ConsolidatingMemory` is an opt-in episodic-memory policy with exact serialized byte accounting, full-context LRU/support eviction, and sparse suffix support. It is not wired into default model behavior. Run `python benchmarks/consolidation_dev.py --output /path/to/new-output.json` from a clean source checkout (exclusive new output). See `docs/RESEARCH.md` and `evidence/consolidation-dev-002.json` for source-pinned results. Sparse support improves final synthetic target-byte accuracy substantially against full-budget LRU, but all arms fail the predeclared worst-case retention limit. No scientific qualification or deployment.

### v0.9 — projected neural plasticity (development only)

Optional `ProjectedFastWeightAdapter(dim, lr=..., rank=8)` can replace `model.fast_weights`. It freezes confident observed directions and projects later no-decay Hebbian updates orthogonally, with bounded scalar updates. Checkpoint stores basis and weights; prior source-bound checkpoints are not migrated. `python benchmarks/plasticity_dev.py --output /path/to/new-output.json` reuses the common comparison harness. Both ranks fail acquisition/retention and predict worse than decay-free shared weights; no default changes. See `evidence/plasticity-dev-001.json` and `docs/RESEARCH.md`.

**Historical engineering blocker before the Gate 1 repair:** existing gated prefix invariant failed final regression; deterministic 100-seed diagnostic reproduces 16 identical failures on v0.8 and v0.9. Run `python benchmarks/prefix_causality_dev.py` (expected nonzero until repair). No scientific or deployment acceptance. See `evidence/prefix-causality-dev-001.json`.

</details>
