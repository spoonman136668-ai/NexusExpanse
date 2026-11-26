# Phase 12 scaled zero-execution assimilation DEV preregistration

Freeze this protocol, implementation, tests and benchmark in the first Phase 12 commit directly after `145c00226fbcfd285efac04fad5ff7b322d7c2fe`. Run the scored check once from that clean tree. Use a fresh output path outside the checkout. The runner writes an intent before scoring, checks the source again at the end and preserves all five seeds in the receipt. Preserve any failed or partial attempt. An ambiguous consumed run requires separate review before another scored attempt.

## Hypothesis and data

The 64-wide, four-layer native byte student initialized from donor tensors will outperform two matched-width controls at zero updates and after 64 native updates. It will also improve on the 32-wide, two-layer initialized student after adaptation. Test the scaled student with and without the static Engram table.

Use seeds 2601 through 2605 and the Phase 9 frozen Python and Markdown split generator. Read files from source commit `105e6e754abe940de5c66ca41a75c3bf825aa1cb`. Each domain contributes 96 FIT, 24 VAL and 24 CHECK windows of 64 raw bytes. The existing selector excludes exact 32-byte overlap across selections. Train and adapt on FIT. Leave VAL unused and report CHECK only after both initialization and adaptation. Record source file lists and split hashes. Give every arm the same CHECK bytes and native FIT minibatch order.

For each seed a separate producer process trains one 128-wide, four-layer, four-head dense donor with BPE vocabulary 320, FFN multiplier 2 and context 64. Learn BPE on FIT. Use the Phase 10B producer's 320 AdamW steps, batch size 8, learning rate .004, weight decay .01, gradient norm cap 1 and existing per-seed offsets. The producer exports three local weights-only files and a tokenizer index. Attach a static 256×256 table from within-window byte transitions in FIT with one pseudocount per outcome and row-normalized log probabilities. Record the donor file hashes, producer forward count and runtime. The consumer executes zero donor forwards.

## Arms and measurement

| Arm | Geometry and initialization |
| --- | --- |
| `dim32_2L` | 32 wide, two layers, Phase 10B tensor transport without the byte table |
| `dim64_4L` | 64 wide, four layers, same tensor transport without the byte table |
| `dim64_4L_engram` | Same 64-wide transport with the FIT-only static byte table |
| `shuffled_dim64_4L` | 64 wide, four layers, same donor and byte table with seeded row and column permutations at the existing shuffle sites |
| `scratch_dim64_4L` | 64 wide, four layers, random native initialization with the same trainable spectral correction capacity |

Score held byte negative log likelihood in bits per byte, averaged over all next-byte predictions. Report pooled and per-domain BPB at zero updates and after 64 updates for every arm and seed. Lower BPB is better. Use the native adaptation routine with identical AdamW settings, minibatch order from seed plus 60000, context and gradient clipping. Initialized arms linearly decay the prior weight from .25 to zero. Scratch starts with zero prior weight. Report the same pool and schedule for every seed.

A fresh process restores and scores all ten archives in each seed. Compare its state and probe-forecast digests and prior weights with the parent. Test independent future changes and shortened inputs on 50 probe seeds per arm and stage. Record all differences, shard peaks, donor call counts and wall time. The scored donor uses a synthetic static byte table. The offline DeepSeek FP8 tests separately cover extraction of real-layout singleton hash rows and the grouped value projection. Live checkpoint import, Colab capacity and any claim of general learning or reasoning require their own evidence.

## Frozen verdicts

- **ENGINEERING PASS** requires at least 659 bare pytest passes with zero failures, skips or warnings, 0/100 prefix benchmark failures, one peak loaded shard per imported arm, at most one downloaded shard at a time, zero guarded donor calls, exact 0.0 prefix difference for all 50 probe seeds and exact cold restores in 5/5 scored seeds.
- **SCIENTIFIC PASS** requires one scaled arm to beat both scratch and shuffled on both stages in all 5/5 seeds and beat the 32-wide arm on post-adaptation BPB in at least 4/5 seeds.
- **SCIENTIFIC MIXED** applies when no arm passes and one scaled arm beats both controls on both stages in at least 4/5 seeds but beats the 32-wide arm after adaptation in fewer than 4/5 seeds.
- **SCIENTIFIC FAIL** covers remaining outcomes. A tie is no win. Apply thresholds to unrounded values. A DEV result awaits independent scientific qualification.

```bash
/workspace/.venv/bin/python benchmarks/phase12_scaled_zero_exec.py --out /workspace/work/nexus-phase12-scored.json
/workspace/.venv/bin/pytest -q
/workspace/.venv/bin/python benchmarks/prefix_causality_dev.py
git fsck --full --strict
git diff --check 145c002..HEAD
```
