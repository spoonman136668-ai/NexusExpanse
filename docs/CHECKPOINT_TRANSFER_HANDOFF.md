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

## Disposition

Confirmation is pending. Advance the first two rows of the frozen ranking, with the first row entering Markdown.
