# Checkpoint transfer handoff

Isolated DEV series based on d48b2df. All decisions and thresholds were frozen in docs/CHECKPOINT_TRANSFER_PROTOCOL.md.
The donor producer is a separate process. The importer accepts tensors and byte-row metadata only. Audit probabilities are created by the producer and never supplied to the importer or optimizer.
The original Python route is frozen and explicitly selected. These results cover compatible attention geometry and 256 singleton-byte symbols. Both models perform neural inference when their own predictions are requested.

## Recovery and reproduction

Use the source SHA in each JSON receipt. Run the workflow with workflow_dispatch on this branch to resume at the first absent receipt. On Linux CPU with Python 3.12, PyTorch 2.10.0+cpu, NumPy and pytest, run the stage commands in .github/workflows/checkpoint-transfer.yml. Keep earlier result files in evidence/ for deterministic selection. Incremental partial receipts are diagnostic; a restarted stage recomputes its whole frozen cohort. The action archives partial results and test logs on failure.

## Screen

| Rank | Method | Worst KL, bits | Mean KL, bits | Skill checkpoint B |
|---:|---|---:|---:|---:|
| 1 | residual8 | 0.0000011 | 0.0000008 | 179859 |
| 2 | block2 | 0.0318441 | 0.0269340 | 183443 |
| 3 | spectral4 | 0.1150958 | 0.0519212 | 176979 |
| 4 | walsh8 | 0.1750173 | 0.1016969 | 183123 |
| 5 | monarch | 0.4021479 | 0.2138419 | 168171 |

## Confirmation

Source commit `217af0ac9a3dfcbe0ae6fda335b394b5adc40e47`.

| Challenge | Method | Near at zero | Near after tuning | Qualified | Matched time pairs | Median time saved | Range |
|---|---|---:|---:|---:|---:|---:|---:|
| Rule | residual8 | 10/10 | 9/10 | 10/10 | 5 | 0.6407 s (97.9%) | 97.5% to 98.3% |
| Rule | block2 | 4/10 | 4/10 | 10/10 | 5 | 0.6375 s (97.4%) | 96.9% to 98.0% |
| Markdown | residual8 | 5/5 | 5/5 | 5/5 | 0 | censored | — |

### Charging donor creation

On rule, median net seconds saved when building a new donor for one receiver: -1.4405. The above table assumes an existing checkpoint. Donor cost includes qualification and reference export. Base receiver preparation is common.

### Donor attribution at zero receiver updates

| Method | Correct rule | Opposite rule | Own FFNs + opposite remainder, accuracy on own rule | Opposite FFNs + own remainder, accuracy on own rule |
|---|---:|---:|---:|---:|
| residual8 | 99.2% | 0.0% | 4.1% | 72.9% |
| block2 | 99.1% | 0.1% | 3.7% | 76.6% |

Exact cold restores of imported and old-mode predictions: 70.
The text and rule evidence includes individual time traces, producer costs, checkpoint sizes, fingerprints and uncensored matched times. A missed target at the tuning ceiling establishes no exact time-to-target.

## Precision fallback

Source commit `637d9b25805ee5b1240ca8de192dcc9b3b99f9ea`.

| Challenge | Method | Near at zero | Near after tuning | Qualified | Matched time pairs | Median time saved | Range |
|---|---|---:|---:|---:|---:|---:|---:|
| Rule | residual16 | 10/10 | 8/10 | 10/10 | 5 | 0.6405 s (97.8%) | 97.6% to 98.6% |
| Markdown | residual16 | 5/5 | 5/5 | 5/5 | 0 | censored | — |

### Charging donor creation

On rule, median net seconds saved when building a new donor for one receiver: -0.4070. The above table assumes an existing checkpoint. Donor cost includes qualification and reference export. Base receiver preparation is common.

### Donor attribution at zero receiver updates

| Method | Correct rule | Opposite rule | Own FFNs + opposite remainder, accuracy on own rule | Opposite FFNs + own remainder, accuracy on own rule |
|---|---:|---:|---:|---:|
| residual16 | 98.3% | 0.4% | 5.0% | 66.2% |

Exact cold restores of imported and old-mode predictions: 60.
The text and rule evidence includes individual time traces, producer costs, checkpoint sizes, fingerprints and uncensored matched times. A missed target at the tuning ceiling establishes no exact time-to-target.

## Disposition

The frozen ladder is exhausted. Preserve the failed criteria. Before tuning another method, identify the failed seed, comparator and criterion in the raw receipts. Distinguish static fidelity from measured time savings; an uncrossed scratch target leaves the savings censored.

## Multi-byte donor follow-up

Protocol: docs/CHECKPOINT_BPE_PREREG.md. Reproduce with benchmarks/checkpoint_bpe_challenge.py bpe-screen or bpe-confirm and a new JSON output path. The winner was chosen before independent confirmation. The three-seed screen ranking (worst donor KL bits) was position_lift 1.34925, suffix_lift 1.38870, sum_head 1.55123.

Status: confirmation pending. The workflow resumes from the committed screen.
