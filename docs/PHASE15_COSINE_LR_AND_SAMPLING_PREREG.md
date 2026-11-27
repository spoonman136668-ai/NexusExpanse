# Phase 15 cosine learning rate and byte sampling DEV preregistration

Freeze the protocol, implementation, tests and benchmark in the first Phase 15 commit directly after `e124207417be94c3ea242a8c5042067860ac23f5`. Score once from that clean tree to a fresh external path. The intent binds the source and protocol before scoring. Keep interrupted or invalid attempts for review. Use Linux CPU FP32, one PyTorch thread and seeds 2901–2905. All inputs are local.

## Learning-rate comparison

For each seed, use the Phase 13 corpus builder and Phase 9 FIT and CHECK selector. The Python and Markdown source is frozen at `105e6e754abe940de5c66ca41a75c3bf825aa1cb`. Use deterministic arithmetic records for math. Each domain supplies 96 FIT and 24 CHECK 64-byte blocks. Preserve disjoint source files for Python and Markdown, separate math operand ranges, and verify zero exact 32-byte FIT/CHECK overlap across domains. The reserved VAL blocks remain unused.

Fit one 32-wide, two-layer native skill from seed plus 50,000 for 32 AdamW updates on FIT only, with a rate of .003, weight decay .01 and gradient norm cap 1. Use one context-32 training block per domain per update and the Gate 17 block order. Save the skill once. Both comparisons receive that exact archive, the same corpus and the same training seed. All training and sampling run under the donor guard.

Train the warm-start arms with `cli train --init-skill` at width 32, two layers, context 32, one block per domain per step, four epochs and 32 updates per epoch. Use AdamW with base rate .003, weight decay .01, and gradient norm cap 1. Keep CPU FP32, dense FFNs and the untied readout for both arms. The control rate remains .003. The cosine arm indexes the rate by the one-based optimizer update, warms linearly to .003 at update `max(1, total_steps // 20)` and decays to .0003 on the final update. A one-update schedule uses the final endpoint.

Run each arm once in one process and once with a fresh process at every epoch boundary. Compare the optimizer, model and RNG checkpoint hashes for every epoch, the step losses, the scheduled rates and the final model hash. Require exact fresh-process cold restore and 0.0 for all four prefix and streaming parity differences. Export and verify the uninterrupted cosine snapshot separately and require 0.0 round-trip difference and exact state. Evaluate all valid CHECK targets with the fresh evaluator. Report unrounded per-domain BPB and pooled BPB weighted by target byte counts.

## Generation checks

With penalty 1.0 and a greedy 16-byte continuation, compare actual `generate` bytes against `stream --no-adapt` from the evaluated cosine snapshot. Use the prompt of one period followed by eight spaces. Separately use a controlled native snapshot whose fitted table favors space over period for all contexts. With the same prompt, require plain greedy generation to produce 16 spaces and penalty 2.0 with a one-byte window to produce eight period-space pairs. Require two runs with that penalty, seed, temperature 0.7 and top-p 0.9 to return identical bytes. The offline tests independently check log scaling and the effect on the normalized probability of a single recent byte.

## Frozen verdicts

- **ENGINEERING PASS** requires exact restart and cold restore, 0.0 prefix and round-trip differences, disabled greedy parity and seeded collapse-breaking in all five seeds, and zero donor forward calls. Require a clean bare pytest run with zero failures, skips or warnings, 0/100 prefix benchmark failures and clean Git integrity and whitespace checks. A miss is **ENGINEERING FAIL**.
- **SCIENTIFIC PASS (DEV)** requires lower pooled held-out BPB with cosine in all five seeds and a mean reduction of at least 0.01 BPB. **SCIENTIFIC FAIL (DEV)** applies if cosine wins zero seeds and raises mean pooled BPB by at least 0.01. All other valid outcomes are **SCIENTIFIC MIXED (DEV)**. Ties do not count as wins. A run with failed training custody or parity cannot qualify a scientific result. Keep every seed and domain in the evidence.
- The comparison measures matched schedule behavior on this small CPU workload. Sampling checks exercise a controlled byte distribution. Independent scientific qualification and large GPU training are outside this run.

```bash
/workspace/.venv/bin/python benchmarks/phase15_cosine_lr_and_sampling.py --out /workspace/work/nexus-phase15-scored.json
/workspace/.venv/bin/pytest -q
/workspace/.venv/bin/python benchmarks/prefix_causality_dev.py
git fsck --full --strict
git diff --check origin/main..HEAD
```
