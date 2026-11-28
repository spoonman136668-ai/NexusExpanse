# Phase 17 focused domain and single-stream training DEV preregistration

Freeze this protocol, implementation, benchmark and tests in the first Phase 17 commit after `a4b1b390bcbf2a1fd82cdad33888d5b5b2b15af8`. Score once from that clean commit into a fresh external path. Record source, tree and protocol digest before scoring. Preserve interrupted or invalid attempts. Use offline Linux CPU FP32, one PyTorch thread and seeds 3101–3105.

## Comparison and byte budget

Use the Phase 13 corpus builder for each model seed. The Python and Markdown sources are pinned at `105e6e754abe940de5c66ca41a75c3bf825aa1cb`, and math uses deterministic arithmetic. Each domain starts with 96 FIT and 24 CHECK 64-byte windows. Code and prose use disjoint FIT and CHECK files, math uses distinct operand ranges, and reserved VAL windows remain unused. Concatenation can still introduce an exact shared window. Before training either arm, try up to eight candidate corpus seeds `seed + 10000 * attempt`, starting at attempt zero. Take the first candidate that passes the full exact overlap check and record all rejected candidates. Keep the model seed fixed. Trim each FIT stream to the largest whole number of 32-target blocks plus one initial byte and verify overlap again. Keep every CHECK byte.

Compare `train --focus-domain all` with `train --focus-domain prose` on the same files and scratch initialization. Both use the native 32-wide, two-layer skill backbone, dense FFNs, context 32, two epochs, 32 updates per epoch and `--batch-per-domain 1`. The mixed arm sees one sequence each from math, code and prose per update. The focused arm draws all three sequences from prose. Both train on exactly 6,144 target bytes in 64 updates, with the same seed, AdamW, gradient clipping and cosine schedule from base rate .003. Keep all settings fixed across arms. Use the same held-out prose bytes and score BPB in the fresh evaluator. Record all three domain scores for context.

## Engineering qualification

For focused prose and single-stream training, run one uninterrupted two-epoch invocation and a repeat with one epoch per fresh process. Require exact epoch checkpoint and final-state hashes, `cold_equal: true`, and identical INT8 export digests in both modes. Verify each resumed archive through the CLI and run `generate --always-wake` with a raw-byte prompt. Require four parity measurements of `0.0`, `0.0` probability round-trip difference, `exact_state: true`, and zero donor forwards. Bind the focused domain in custody when selected. Legacy `all` retains its existing manifest fields. Raw-file mode uses one stream batch of `--batch-per-domain` sequences and binds both file digests under `stream`.

For the single-stream case, generate a 256 KiB offline binary capture with Python's seeded byte generator at seed plus 100,000. Pack it via the same single-stream packager used by `--single-dir` and independently verify zero exact 32-byte train/eval overlap. The packager preserves the bytes of whole local files in a directory and uses contiguous spans of up to 64 KiB when supplied a single file. Hub input uses whole text records. Unit tests exercise a mocked Hub source, misconfigured sources, and changed domain/file manifests. They also cover CPU BF16, SDPA, warm-start and `--until-complete` in raw-file mode.

## Frozen verdicts

- **ENGINEERING PASS** requires exact cold resume and INT8 export in both modes for all five seeds, matched initial hash, target bytes and step count for the two comparison arms, and zero donor forward calls. It also requires the offline manifest-rejection and packager controls, a clean bare pytest run with zero failures, skips or warnings, the legacy prefix benchmark at 0/100 failures, and clean Git integrity and whitespace checks. A miss makes the engineering verdict **FAIL**.
- **SCIENTIFIC PASS (DEV)** requires strictly lower held-out prose BPB for focused training in 5/5 seeds. **MIXED (DEV)** applies for three or four wins and **FAIL (DEV)** for zero through two. Ties are misses. Failed scored engineering invalidates the comparison. Report every result. The bounded test measures target-domain specialization under a fixed byte budget. Live Hub packing, GPU performance and independent scientific qualification require separate evidence.

```bash
/workspace/.venv/bin/python benchmarks/phase17_domain_focus.py --out /workspace/work/nexus-phase17-scored.json
/workspace/.venv/bin/pytest -q
/workspace/.venv/bin/python benchmarks/prefix_causality_dev.py
git fsck --full --strict
git diff --check origin/main..HEAD
```
