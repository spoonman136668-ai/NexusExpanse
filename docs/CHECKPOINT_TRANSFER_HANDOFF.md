# Checkpoint transfer handoff

Stage: preregistered, awaiting CPU execution. The checkpoint-transfer workflow continues independently of the chat and pushes a separate commit per evidence stage to this research branch. The branch starts at Gate 26's published tip d48b2df. No local Gates 27–32 commits are dependencies.

Read CHECKPOINT_TRANSFER_PROTOCOL.md before interpreting any result. The converter is nexus_expanse/checkpoint_transport.py; the runner is benchmarks/checkpoint_transfer_study.py. A completed stage writes evidence/checkpoint-transfer-{screen,confirm,escalate}.json as applicable. The workflow regenerates this handoff after each stage. To resume, dispatch .github/workflows/checkpoint-transfer.yml on this branch. It skips committed stages, runs the frozen remainder, then verifies the suite and original prefix reproducer. Partial receipts can be recovered from the run's checkpoint-study artifact. No threshold edits across stages.
