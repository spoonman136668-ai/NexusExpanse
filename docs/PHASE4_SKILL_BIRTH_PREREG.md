# Phase 4 online skill birth preregistration

**Hypothesis.** A window whose bytes surprise every resident skill can identify a strongly shifted raw-byte domain. A clone adapted only to the observed encounter can acquire its cue-to-answer rule. The expanded bank can then route to it without changing resident weights or materially worsening prediction on original domains.

## Frozen source and streams

Run once on Linux CPU from the clean preregistration commit directly above Phase 3C tip `76ab0e94f814a79fcbf4e78701772197efc630ed`. Use the workspace virtual environment, PyTorch 2.15.0+cpu and one deterministic FP32 Torch thread. Use seeds 1701–1705. Reuse the Phase 3A text and Rule A/B fixtures, the four separate 192-step static donor fits, and `residual8` import. The pinned Python and Markdown byte source is `105e6e754abe940de5c66ca41a75c3bf825aa1cb`. No network data or donor execution enters the receiver.

Rule C uses the same four-byte cue-pair, delimiter and answer structure as the other rules, with a disjoint rule alphabet. Each record for key *i* has first byte 176 + (*i* modulo 8), second byte 192 + floor(*i*/8), delimiter 208, and an answer from 224–231. The seed-specific mapping is a distinct permutation within each group of eight keys. Rule A/B use different cue and answer bytes. Each 64-byte Rule C window samples 16 distinct keys. Encounter and CHECK use separate draws from the frozen generator. Exact 32-byte overlap is forbidden against original FIT and CHECK, between the two Rule C splits, and at the concatenated encounter boundaries. Preserve corpus and consumed encounter digests. This confirmation measures strongly shifted byte-domain novelty.

## Online encounter and matched comparisons

The base bank starts with frozen Python, Rule A, Rule B and Markdown receivers. Keep labels outside the autonomous prediction and adaptation paths. Present eight ordered 64-byte Rule C blocks at most. Start checking after four blocks (256 observed bytes), then after each subsequent block until the first birth or the 512-byte stop. For the most recent 64 observed bytes, measure each resident's mean BPB using the same strictly causal conditional forecasts available to HMM BMA. A birth triggers only if the minimum of those resident averages strictly exceeds 8.0 BPB. No probability from the next unseen byte is used.

At a trigger, clone the frozen Python receiver and fit only the clone. Run 32 AdamW steps at learning rate 0.004, weight decay 0.01, and gradient norm cap 1.0. Cycle through observed next-byte pairs with a prefix context of at most 64 bytes. Validate all original weights before registration, freeze the new receiver, and stop adapting. Use the normal bank snapshot format for a fresh-process restore and autonomous HMM routing. Negative probes take the eight original CHECK windows per domain in disjoint ordered pairs, yielding four 128-byte probes per domain and 16 per seed at the same threshold. Record every positive flag as a false birth. Do not use negative or CHECK feedback to adjust the threshold, train, or select a model.

| Arm | Fit and prediction |
| --- | --- |
| `open_world_birth` | Triggered clone trained on the observed encounter prefix. Cold-restored HMM BMA over the resulting bank. |
| `fixed_k4_bank` | Original untouched receivers, HMM BMA over four skills. |
| `continual_finetune_base` | Independently copied bank with its Python receiver updated in place by the same optimizer, observed bytes and 32 steps. HMM BMA over four skills. |
| `oracle_5_skill` | Cold-restored five-skill bank with true domain provided for each byte. Use the same born receiver if available. If the trigger never fires, fit the oracle's extra clone on the full observed encounter using the same 32-step rule. |

Compare all arms on eight held-out 64-byte windows per domain. Record rule top-1 accuracy on answer positions, BPB over all positions, original-domain pooled BPB, and per-domain BPB drift from the fixed bank. Measure Rule C routing on every byte, including the initial prior, using the top-weight skill. Record exact resident state and individual forecasts across expansion, and exact fresh-process state and output digests. A seed must verify both the autonomous snapshot and an expanded snapshot. Record donor guards, training costs, evaluated bytes, null births and missed gates.

## Frozen verdicts

- **ENGINEERING PASS** requires at least 307 pytest cases with zero failures or skips, 0/100 failures on the legacy prefix check, 0.0 expanded HMM BMA prefix difference across 50 seeds, zero false births, zero receiver donor forwards, and 5/5 exact resident preservation and cold restores.
- **SCIENTIFIC PASS** requires birth and strict improvements over the fixed bank on both held-out Rule C BPB and rule accuracy in all five seeds. It also requires lower pooled original-domain BPB than continual base fine-tuning in all five seeds, with no original domain increasing by more than 0.02 BPB versus the fixed bank. At least four seeds must route at least 80% of held-out Rule C bytes to the new cold-restored skill.
- **SCIENTIFIC MIXED** requires birth and both prediction and retention comparisons to succeed jointly in at least four seeds, with zero seeds beyond the original-domain forgetting limit and at least one PASS requirement unmet.
- **SCIENTIFIC FAIL** applies when fewer than four seeds meet the joint comparison or an original domain exceeds the forgetting limit.

Reject source or data drift and preserve negative outcomes. Commit the unrounded report as `evidence/phase4-skill-birth-dev-001.json` after scoring. Qualify the final packaging integration separately.
