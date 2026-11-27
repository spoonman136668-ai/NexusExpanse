# Phase 13 warm-start training and byte generation DEV preregistration

Freeze this protocol, implementation, tests and benchmark in the first Phase 13 commit directly after `bbdc3c3ae4f43e4677d111ba38bb7bccf4b3de9b`. Run the scored check once from that clean checkout at a fresh external output path. The intent binds the source and protocol before the first seed. Preserve a consumed or interrupted attempt for review.

## Hypothesis and data

A zero-execution assimilated 64-wide, four-layer skill will finish native Gate 17 training at lower pooled held-out bits per byte than matched scratch and shuffled initialization. Use fresh seeds 2701 through 2705. Freeze Python and Markdown source bytes at `105e6e754abe940de5c66ca41a75c3bf825aa1cb` and use the Phase 9 FIT and CHECK selector for code and prose. Use Gate 17's deterministic math corpus generator for math. Each domain supplies 96 FIT and 24 CHECK 64-byte blocks. The code and prose VAL selections stay reserved and unused. Reject any exact 32-byte overlap between the packed FIT and CHECK files. Record file hashes and held-out counts.

Train one local 128-wide, four-layer BPE-320 donor per seed in a separate producer process using the existing Phase 10B budget of 320 updates from all three FIT domains. Export static weights. Use the Phase 12 shard importer to make aligned and shuffled 64×4 skills from that same donor and tokenizer. Guard donor execution in the consumer. The producer forward count is reported separately.

## Comparison and checks

| Arm | Initialization |
| --- | --- |
| `warmstart_train` | 64×4 zero-execution transport, then the native training CLI with `--init-skill` |
| `scratch_train` | Same native geometry with its seeded random weights, selected with `--skill-backbone` |
| `shuffled_init_train` | Same donor, geometry, and static-prior path with transport permutations from seed plus 50000 |

All arms use four attention heads, context 64, dense native FFNs with multiplier 2, an untied native readout, CPU FP32, two epochs, 32 Gate 17 AdamW updates per epoch and one training block per domain per step. The optimizer uses learning rate .003, weight decay .01 and gradient norm cap 1. Matched initialization variants carry the same available FIT information and adaptation budget. Copy skill parameters and materialize its affine corrections once. Copy matching learned position rows and retain seeded target rows beyond the imported context. For imported arms, linearly decay any stored static byte-prior weight from its initial value to zero across the first epoch, with no prior contribution afterward. A one-step epoch uses zero. Scratch has no checkpoint prior.

Run one epoch per CLI invocation and compare the end and resumed model digests and the optimizer step cursor. Evaluate all 24 CHECK blocks per domain in the fresh evaluator, using only valid held target bytes. Report pooled BPB weighted by target byte counts plus all three domain BPBs. Verify the exported unified snapshot in a new CLI process and compare greedy bytes against `stream --no-adapt` from the same snapshot and prompt. `generate` defaults to fixed online memory, and `--adapt` enables learning inside its private restored copy. The legacy stream observes the prompt with adaptation by default. Report all five seeds, even if the effect is null or negative.

## Frozen verdicts

- **ENGINEERING PASS** requires at least 672 bare pytest passes with zero failures, skips or warnings, 0/100 prefix benchmark failures, exact 0.0 unified parity and round-trip difference, exact greedy generation parity, zero consumer donor calls and exact cold resume and restore in 5/5 seeds.
- **SCIENTIFIC PASS** requires the aligned warm-start arm to beat both controls on unrounded pooled held-out BPB in all 5/5 seeds.
- **SCIENTIFIC MIXED** applies when the aligned arm beats both controls in exactly 4/5 seeds.
- **SCIENTIFIC FAIL** applies below 4/5. Ties do not count as wins. This DEV comparison requires independent scientific qualification.

```bash
/workspace/.venv/bin/python benchmarks/phase13_warmstart_train.py --out /workspace/work/nexus-phase13-scored.json
/workspace/.venv/bin/pytest -q
/workspace/.venv/bin/python benchmarks/prefix_causality_dev.py
git fsck --full --strict
git diff --check bbdc3c3..HEAD
```
