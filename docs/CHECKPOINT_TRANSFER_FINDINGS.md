# Checkpoint-only transfer: findings and handoff

Everything in this report is DEV. The published starting point is d48b2df. The local Gates 30–32 branch was not in the available Git history. Imported skills are selected explicitly. Donor training and reference collection run in isolated producer processes. The converter reads static tensors/vocabulary, and the receiver learns only from ground-truth bytes. No teacher cache or teacher logits enter the optimizer.

## What crossed

The two independently fine-tuned donors start from one parent and solve incompatible secret cue/byte rules. Both swaps use identical queries, vocabulary and receiver code. The first test uses fresh byte surfaces on previously learned key/cue combinations. The Markdown test measures all 256 next-byte probabilities on a file separate from training and validation. All old-mode tests use disjoint Python files.

| Interface | Cases | Near before tuning | Near after stopping | Mean zero-update top-1 | Worst zero-update KL (bits) |
|---|---:|---:|---:|---:|---:|
| 32 → 32 rule | 10 | 10 | 9 | 100.00% | 0.0000029 |
| 32 → 32 Markdown | 5 | 5 | 5 | 99.86% | 0.0000053 |
| 32 → 64 rule | 10 | 7 | 6 | 100.00% | 0.0000084 |
| 32 → 64 text | 5 | 5 | 5 | 99.90% | 0.0000040 |

Multi-byte BPE donor → raw-byte receiver: 0/5 near-lossless after validation-selected stopping; mean zero-update agreement 26.88%. Best bridge: position_lift. This comparator uses a canonical-prefix-boundary byte posterior. Equal hidden dimensions alone do not qualify this interface.

### What a single weight-group swap changed

| Replaced from opposite donor | Own-rule accuracy | Opposite-rule accuracy | Cases meeting the stand-alone carrier criterion |
|---|---:|---:|---:|
| none | 98.2% | 0.2% | 0/10 |
| embeddings | 93.7% | 0.4% | 0/10 |
| attention | 9.3% | 47.8% | 0/10 |
| norms | 97.7% | 0.2% | 0/10 |
| ffn | 72.2% | 4.2% | 0/10 |
| readout | 93.5% | 1.3% | 0/10 |
| all | 0.2% | 98.2% | 10/10 |

## Time to the common task target

Includes checkpoint loading, conversion, tuning and validation. Excludes common recipient pretraining and independent auditing. Only pairs meeting both validation and test targets are included in savings. A censored run establishes no exact saving. Small-model wall times are local to this single-thread CPU workflow. Medians of savings are taken pairwise.

| Challenge | Comparator | Matched / total | Median receiver s | Median comparator s | Median s saved | Median % saved |
|---|---|---:|---:|---:|---:|---:|
| 32 → 32 rules | scratch | 5/10 | 0.0140 | 0.6545 | 0.6407 | 97.9% |
| 32 → 32 rules | parent | 8/10 | 0.0140 | 0.8609 | 0.8468 | 98.4% |
| 32 → 32 text | scratch | 0/5 | — | — | censored | — |
| 32 → 32 text | parent | 5/5 | 0.0187 | 0.4977 | 0.4793 | 96.2% |
| 32 → 64 rule | scratch | 1/10 | 0.0133 | 0.9617 | 0.9484 | 98.6% |
| 32 → 64 rule | parent | 1/10 | 0.0133 | 0.5882 | 0.5749 | 97.7% |
| 32 → 64 text | scratch | 0/5 | — | — | censored | — |
| 32 → 64 text | parent | 2/5 | 0.0189 | 0.3230 | 0.3041 | 94.1% |

If creating the compatible 32-wide donor purely for one receiver, median net time saved on the 5 scratch-matched rule pairs was -1.4405 seconds. Producer creation and reference scoring are counted. An existing donor is the use case in the main timing table.

## Limits and disposition

The winning correction keeps quantized information about the omitted dense weights and computes an additional dense linear operation at inference. The original compatible screen measured a 179,859-byte imported skill and a 150,613-byte donor checkpoint. Closer predictions did not imply a smaller model. The wider transfer embeds a 32-dimensional function into a 64-dimensional repeated-coordinate subspace and increases storage again. None of these assays establishes broad knowledge transfer across unrelated model families.

The independently seeded FP16 fallback reached zero-update near-losslessness on 10/10 rules and 5/5 Markdown cases. Its complete joint time/fidelity gate was False.

## Continue or reproduce

1. Clone the complete research branch (the corpus reader uses the historical base Git object):
   `git clone --branch dev/checkpoint-transfer-fidelity https://github.com/spoonman136668-ai/NexusExpanse.git`
2. Use Python 3.12, CPU PyTorch 2.10.0, NumPy and pytest. Entry points, workflow invocations, and thresholds are documented in CHECKPOINT_TRANSFER_PROTOCOL.md, CHECKPOINT_BPE_PREREG.md and CHECKPOINT_WIDTH_PREREG.md. CHECKPOINT_TRANSFER_EXECUTION.md records the cold-reference correction.
3. The seven JSON files under evidence/checkpoint-transfer-* are the machine-readable receipts (screen, confirmation, precision fallback, BPE screen, BPE confirmation, component swaps, width). Each includes the measured source commit. The component, BPE and width tests use separate fresh seeds. Keep test results out of training and winner selection.
4. For the ready-made receiver-only demonstration, download the checkpoint-transfer-width-demo workflow artifact, extract it and run `python benchmarks/checkpoint_width_lift.py demo <directory-containing-example.json>`. A successful run verifies two cold restored skills and both original-route hashes and prints the two different outputs for one raw prefix. To rebuild the kit from source, run `python benchmarks/checkpoint_width_lift.py results/width-replay.json results/demo` after creating results/.
5. The highest-value open problem is preserving causal histories across different symbol boundaries. Use the fixed 8-bit correction as a control, fresh donor pairs and an independent corpus. Measure full-distribution divergence, donor-swap specificity, held-out retention, payload, inference cost, and time to identical performance. Avoid claiming an exact time saving for a baseline that failed to hit the target.

## Phase 2: Multi-Byte BPE Follow-Up (Causal Folding & Offline Byte-KL)

All three prospective follow-up experiments below branch from `92d43cb` on the same isolated producer/receiver split (`docs/GATE17_PREREG.md` + `docs/GATE20_PREREG.md` fit, `GATE22_PREREG.md` val, `GATE23_PREREG.md` test), with `donor_forward_calls == 0` in every receiver and zero BPE tokenizer/merges at inference.

| Experiment | Branch / Evidence | Outcome | Key Finding |
| :--- | :--- | :--- | :--- |
| **Causal-History Folding** (`docs/CHECKPOINT_BPE_FOLDING_PREREG.md`) | `dev/checkpoint-transfer-bpe-causal-folding` (`c32717d`), `evidence/checkpoint-transfer-bpe-fold-confirm.json` | **ENG PASS / SCI FAIL** | `static_trie_fold` beat `shuffled` and `scratch` in 5/5 seeds (`1001–1005`), but lost to `position_lift` on zero-update KL in 5/5 seeds (`1.23–1.35` vs `1.16–1.30` bits). Hard chunk-end masking + EMA folding drops fine-grained byte context when heuristics disagree with greedy BPE. |
| **On-Fit Offline Byte-KL** (`docs/CHECKPOINT_BPE_OFFLINE_KL_PREREG.md`) | `dev/checkpoint-transfer-bpe-offline-kl` (`59dcaae`), `evidence/checkpoint-transfer-bpe-offline-kl.json` | **ENG PASS / SCI FAIL** | Exporting a `96,768 B` top-`k=16` cache on `fit` and running 160 steps overfit the 16-window `fit` split (`1.68–1.86` bits test KL vs `1.09–1.30` for `position_lift` at 0–8 steps), though it still beat `shuffled_offline_kl` on top-1 agreement in 5/5 seeds. |
| **Fresh-Corpus Validation-Stopped Offline Byte-KL** (`docs/CHECKPOINT_BPE_TRANSFER_CORPUS_KL_PREREG.md`) | `dev/checkpoint-transfer-bpe-offline-kl` (`0f05017`), `evidence/checkpoint-transfer-bpe-transfer-corpus-kl.json` | **ENG PASS / SCI MIXED** | Exporting a `196,608 B` top-`k=16` cache on 64 disjoint transfer windows (`GATE15`/`16`/`18`/`19_PREREG.md`) with `val`-KL early stopping (`24–40` steps) beat `position_lift`, `shuffled_transfer_kl`, and `transfer_kl_from_scratch` on held-out `test` KL and agreement in **5/5** seeds (`1201–1205`). |

### Fresh-Corpus Offline Byte-KL (`seeds = [1201–1205]`, `test` KL bits / Top-1 Agreement)

| Seed | `transfer_kl_from_position_lift` | `transfer_kl_from_scratch` | `shuffled_transfer_kl` | `position_lift` |
| ---: | ---: | ---: | ---: | ---: |
| **1201** | **1.0379 / 24.90%** (32) | 1.1902 / 20.63% (88) | 1.1660 / 14.68% (16) | 1.3720 / 13.99% (8) |
| **1202** | **1.0875 / 27.98%** (32) | 1.2272 / 21.23% (88) | 1.2019 / 20.14% (24) | 1.4507 / 12.70% (8) |
| **1203** | **0.9402 / 28.08%** (24) | 1.1229 / 23.91% (80) | 1.0802 / 19.35% (8) | 1.1306 / 25.10% (0) |
| **1204** | **0.9744 / 28.17%** (40) | 1.1232 / 24.80% (72) | 1.0726 / 20.83% (8) | 1.0722 / 27.48% (0) |
| **1205** | **1.1739 / 25.89%** (32) | 1.3305 / 19.54% (96) | 1.3375 / 14.29% (32) | 1.5426 / 10.52% (8) |
| **Mean** | **1.0428 / 27.00%** | 1.1988 / 22.02% | 1.1716 / 17.86% | 1.3136 / 17.96% |

**Disposition:** `position_lift` warm-start + disjoint-corpus validation-stopped offline byte-KL is the strongest BPE -> raw-byte transfer recipe tested so far (`5/5` wins over every baseline and control), but at `dim=32` / 64 transfer windows it stops at `0.94–1.17` bits KL (`~27%` agreement), short of the `< 0.85` / `>= 99%` near-lossless bar. Scaling the unlabeled transfer corpus and student width is the remaining lever.
