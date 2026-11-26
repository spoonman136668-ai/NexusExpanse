# Phase 11 SDPA and three-stage composition preregistration

Freeze this protocol, code, tests, and benchmark on the first clean commit above `a01f4386715fd59c75535961194677847d862622`. Run scored confirmation once on Linux CPU with PyTorch FP32, deterministic algorithms, and one Torch thread. Put the raw receipt and documentation in the second commit and qualify that exact commit. Keep protected main at `bb21fbe53c0299d432e866eb4d016c25b81ee6eb`. No external downloads or paid APIs.

## Suite A — SDPA

Use a 64-wide native two-layer core, four heads, FFN multiplier 2, no dropout or FFN skipping, batch 2, and context lengths 512 and 1024. Start both paths with identical seed 2511 state. Generate byte transitions independently per context with seed 2511 plus context. Each arm performs two warm-up forward/backward passes and seven measured passes without optimizer updates. The forward measures native forecasts and cross-entropy. Backward measures gradient evaluation. Report median forward, backward, and combined times with all samples.

Record peak tracked CPU tensor allocation from the PyTorch profiler's CPU allocation events in a separate, untimed pass. This covers tensor allocations observed during the pass. Process peak RSS is recorded separately. Check at most 1e-5 maximum absolute difference in training logits, hidden states, and parameter gradients. Also report unmodified native MHA evaluation differences for transparency. Its fused evaluation kernel uses a different reduction order from SDPA. Verify exact 0.0 across SDPA evaluation on the full context, truncated prefixes, and perturbed suffixes. The unit suite additionally checks training and native evaluation against legacy at lengths 32, 128 and 512 and checks eligibility for strict state-dict loading both ways after training.

The opt-in CLI sets causal SDPA at the training and cold validation entry points, permitting contexts up to 4096. The CPU unified deployment archive loads the same tensors through its established bit-exact incremental attention runtime. CUDA FlashAttention or BF16 deployment require separate execution evidence.

## Suite B — one continuous lifecycle

Use seeds 2501–2505. Use the Phase 10B disjoint FIT, VAL, and CHECK construction pinned to authentic Python and Markdown source `105e6e754abe940de5c66ca41a75c3bf825aa1cb`. Use 96 FIT, 24 VAL, and 24 CHECK 64-byte text windows per domain. Rule A and Rule B keep the Phase 3 frozen synthetic rules, FIT size, and eight held windows per domain. All three additional residents use Phase 3 producer geometry and budget with the retained FIT files. Train the donors in the producer process. Only checkpoint bytes reach the consuming and evaluating code.

1. Run the Phase 10B 64-wide BPE-320 producer and transport the three local mock shards into a 32-wide raw-byte skill. Check shard hashes against the producer, count open shards, and execute the receiver under the donor-forward guard. Compare with the same initialized native geometry and trainable spectral residual budget from Phase 10B scratch. Both adaptation arms use 64 matched AdamW raw-byte updates on FIT. Donor-derived prior alpha decays from 0.25 to 0.0. Select the donor-derived native skill for the next stage regardless of its held-out BPB. Record static-only and adapted held-out BPB and exact snapshot checks.
2. Inject the Phase 9 layer 1 output-projection rank-one fault into the **adapted assimilated skill**. Reuse Phase 10B's one-sided verified repair, SVD and budget-matched random directions, frozen alpha grid, VAL choice and untouched CHECK. Accept a repair only with strict fault improvement, exact prefix and cold parity, and signed Markdown delta no higher than +0.02 BPB. Record signed held-out Markdown change, fault BPB, damage recovery and blocked autograd calls. Install the actual returned SVD skill after archive restore. If rejected, install its faulted fallback and record a composition failure.
3. Install that very skill as `base_python` alongside `rule_A`, `rule_B`, and `markdown_prose`. Measure pre-birth hierarchical HMM forecasts. Check false births on all held resident windows paired into disjoint 128-byte streams, totaling 32 probes per seed. Use existing Phase 4 Rule C construction with a disjoint alphabet, the Phase 4 observed-prefix scan of four to eight 64-byte blocks, the established novelty threshold, and `episodic_assisted_birth` via `OpenWorldSkillBank.observe`. Only observed encounter bytes reach birth. Save the actual expanded bank, then evaluate it in a new process on eight held Rule C streams. Measure cue-answer accuracy at every answer byte and routing across all positions. Compare warm and cold roster, states, and forecast digests exactly. Record per-domain signed BPB changes from the pre-birth hierarchical bank and the maximum signed change across repaired Python held windows. The retention measure is the maximum of zero, these per-domain means, and the worst repaired Python window. Also verify bit-exact resident weights and direct repaired-resident forecasts across birth.

## Frozen verdicts

**ENGINEERING PASS** requires 615 or more bare pytest passes and zero failures, skips, or warnings. Legacy prefix failures must be 0/100. The 50 SDPA unit seeds and 50 assembled-bank unit seeds must have 0.0 prefix difference. Suite A must pass its FP32 training parity and exact prefix bounds. Suite B requires at most one simultaneously loaded shard, exactly zero guarded donor calls and Stage 2 autograd calls, zero false birth flags on resident probes, and exact cold restores in all five seeds.

**SCIENTIFIC PASS** requires all three stage gates in every seed. Stage 1 must beat scratch on held combined BPB with zero donor calls. Stage 2 SVD must be accepted, recover at least 50% of positive injected Python damage, beat random on held Python BPB, stay within +0.02 held Markdown BPB, and use zero autograd. Stage 3 must birth K=5, achieve at least 85% held Rule C answer accuracy and 90% cold routing, and keep maximum hierarchical retention drift within +0.015 BPB.

**SCIENTIFIC MIXED** requires at least four seeds to meet all stage gates while all seeds compose with exact state, causality, and bounded retention. Any broken handoff, denied SVD repair, missing birth, exactness failure, retention violation, or fewer than four joint passes yields **SCIENTIFIC FAIL**. Preserve measurements even when a threshold fails.

```bash
/workspace/.venv/bin/python benchmarks/phase11_sdpa_and_3stage.py --out /workspace/work/nexus-phase11-scored.json
env -u PYTHONPATH /workspace/.venv/bin/pytest -q
/workspace/.venv/bin/python benchmarks/prefix_causality_dev.py
git fsck --full --strict
git diff --check a01f438..HEAD
```
