# Phase 6 resident-anchored K=7 scaling preregistration

**Hypothesis.** Reserving fixed initial and switching probability for the born group can reduce original-domain BPB drift as the bank grows. The same expanded bank under the uniform HMM isolates the routing intervention. Three sequential births test whether earlier bindings survive later acquisitions and cold restore.

## Frozen source and data

Execute once on Linux CPU from the clean preregistration commit immediately above Phase 5 tip `362b8ccc84fde134e890b5862ad6a1760e279776`. Use the workspace Python environment, PyTorch 2.15.0+cpu and one deterministic FP32 Torch thread. Use seeds 1901–1905. Use the same four static donor fits, `residual8` conversion, starting skills and held-out resident suites as Phase 5. The original text fixture bytes remain pinned at `105e6e754abe940de5c66ca41a75c3bf825aa1cb`. Only the isolated producer executes donors. Receiver import, birth, inference and cold restore use the donor-forward guard and local data.

Use four-byte records for all three novel domains. For key *i*, the first two bytes are the bases below plus *i* modulo 8 and floor(*i*/8), then the delimiter and an answer drawn from the specified eight-byte range by a fixed seed permutation per group of eight keys.

| Domain | First cue base | Second cue base | Delimiter | Answer bytes |
| --- | ---: | ---: | ---: | --- |
| Rule C | 176 | 192 | 208 | 224–231 |
| Rule D | 160 | 168 | 172 | 184–191 |
| Rule E | 200 | 209 | 213 | 232–239 |

Use separate candidate draws for eight 64-byte encounter windows and eight 64-byte CHECK windows per novel domain. Alphabets are mutually disjoint. Exclude exact 32-byte overlap across encounter and CHECK, previous novel domains and the original fixtures. Include windows crossing concatenated encounter boundaries in the overlap check. Each test uses fresh surrounding bytes with the same cue bindings.

## Sequential growth and controls

Present Rule C, then D, then E. Start screening each encounter after 256 observed bytes, then every 64 bytes up to 512. The best existing skill must exceed 8 BPB over the latest 64 observed bytes. A positive trigger clones Python and consolidates with the Phase 5 episodic mode using 32 neural updates and the frozen three-byte suffix table. Freeze each slot at registration. Persist the starting four names and preserve every pre-existing state through subsequent births. No labels enter the novelty detector or forecast. Domain names label the slots for scoring after registration.

Compare the four frozen arms below.

| Arm | Treatment |
| --- | --- |
| `hierarchical_episodic_k7` | Apply the resident-anchored HMM to the cold-restored sequentially grown bank. |
| `uniform_episodic_k7` | Apply the standard uniform-initial HMM to the exact same cold-restored bank and streams. |
| `fixed_k4_bank` | Keep only the original four receivers with standard HMM routing. |
| `continual_finetune_base` | Fit Python in place in an independent K=4 bank, in C → D → E order, using the same observed prefixes and 32 steps per encounter. Use standard HMM routing. |

For the hierarchical arm, freeze switch probability at 0.05, total born initial mass at 0.02, and the born share of resident switches at 0.10. Split remaining initial mass equally among residents. Split each resident's switch budget between other residents and born states, then equally within each destination group. Born rows distribute their switch probability equally to all other states. Keep diagonal mass at 0.95. Apply filtering in log space from observed bytes. With no born slots, use the standard HMM exactly.

Score the original held-out 128-byte negative probes at K=4 and again after cold restore. Revisit Rule C in four disjoint 128-byte pairs per seed after cold restore. Measure routing to Rule C and call the novelty/birth entry point using a fresh candidate name on disposable copies. A new slot on a revisit counts as a duplicate birth. Preserve counts and exceptions separately.

Measure mean and minimum answer accuracy and routing accuracy across C, D and E. Include every test byte in routing accuracy and only answer positions in rule accuracy. Record first assigned position, original-domain pooled BPB, and maximum per-domain BPB increase versus fixed K=4. Compare warm and fresh-process state, resident names and exact forecast digests for both K=7 routers. Measure resident forecasts independently of mixture scores and compare each earlier-born state against its state at first registration. Preserve per-domain values, timings, seed-level outcomes and custody.

## Frozen verdicts

- **ENGINEERING PASS** requires at least 419 tests passing with bare pytest and zero failures, skips or warnings, 0/100 legacy-prefix failures, exact 0.0 K=7 prefix difference across 50 seeds, zero false and duplicate births, zero receiver donor forwards, and 5/5 exact resident preservation and K=7 cold restores.
- **SCIENTIFIC PASS** requires at least 85% held-out answer accuracy and 90% cold routing on every new domain in every seed. In all five seeds, maximum original-domain BPB drift must be at most 0.015 and strictly lower than the drift of both uniform K=7 and continual fine-tuning.
- **SCIENTIFIC MIXED** requires at least 85% answer accuracy on all three new domains in all five seeds and lower maximum drift than uniform K=7 in at least four seeds, with a PASS requirement unmet.
- **SCIENTIFIC FAIL** applies when neither PASS nor MIXED criteria are met.

Record the stronger 0.005 BPB drift target and first assigned positions as diagnostics. Reject source or fixture drift and preserve null and negative results. Commit the unrounded output as `evidence/phase6-hierarchical-scaling-dev-001.json` after scoring and qualify the final evidence-and-documentation tip separately.
