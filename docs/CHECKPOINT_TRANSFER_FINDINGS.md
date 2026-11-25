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
