# Phase 5 born-skill mastery preregistration

**Hypothesis.** The Phase 4 receiver routed strongly shifted Rule C bytes to a new skill and reached 10.16–14.06% answer accuracy. An exact suffix memory learned from the same encounter can bind repeated cues to the correct answers in fresh held-out streams. The matched longer neural fit tests whether additional gradient updates achieve the same result.

## Frozen source, fixtures and exposure

Run once on Linux CPU from the clean preregistration commit directly above Phase 4 tip `61be699e0c254c6361e3a3cd1cb2a80350e6ef85`, using the workspace virtual environment, PyTorch 2.15.0+cpu and one deterministic FP32 Torch thread. Use fresh seeds 1801–1805. Reuse Phase 4's four-donor producer, static `residual8` receiver imports, authentic pinned text source `105e6e754abe940de5c66ca41a75c3bf825aa1cb`, rule generators, FIT/CHECK geometry and source verification. Donors execute only in the separate producer. Receiver import, adaptation, inference and cold restore run under the donor-forward guard, with no network data.

Freeze Rule C exactly as in Phase 4. The novel rule uses two cue bytes, a delimiter, and an answer among eight raw bytes. Encounter and CHECK use separate generator draws; exact 32-byte overlap is forbidden, including at joined encounter boundaries. Cue triplets can recur in otherwise fresh held-out streams. The experiment tests recall of the same cue-to-answer binding across fresh surrounding bytes. Held-out positions whose exact cue triplet has no stored transition are included in the primary score.

Use the common Phase 4 novelty decision once on the identical initial bank. Begin after 256 observed bytes, then check every 64 bytes up to 512. Trigger only when the best resident averages more than 8 BPB over the latest 64 observed bytes. All birth arms receive the same observed prefix and trigger. They add no slot when the trigger fails. The continual control receives that same observed prefix whether the trigger fires or the encounter reaches its limit. Probe the four original held-out domains in the same disjoint 128-byte pairs, yielding 16 negative controls per seed. No labels or CHECK feedback reach birth, adaptation or routing.

## Frozen birth paths and comparisons

All gradients use the Phase 4 AdamW settings and repeat the same observed context blocks. The four original receivers stay frozen in each birth arm. Every forecast predicts the next raw byte using only earlier bytes.

| Arm | Consolidation and prediction |
| --- | --- |
| `episodic_assisted_birth` | Clone Python and run 32 neural steps. Record every observed three-byte suffix and successor in the encounter. Retain a suffix only when all observations agree on its successor. On an exact hit, mix 90% probability on the recorded successor with 10% of the neural distribution. On misses and conflicts, use the neural forecast. Freeze the table at birth and use autonomous HMM BMA from the cold-restored bank. |
| `consolidated_neural_birth` | Clone Python and run 192 neural steps with the same observed bytes. No episodic head. Cold-restored HMM BMA. |
| `phase4_baseline_birth` | Clone Python and run the original 32 neural steps with the same bytes. No episodic head. Cold-restored HMM BMA. |
| `fixed_k4_bank` | Original untouched skills and HMM BMA over four slots. |
| `continual_finetune_base` | Update Python in an independent K=4 copy for 192 steps on the same observed bytes and use HMM BMA. |

The episodic table uses sorted 24-bit keys stored as int32 and uint8 successors. It contains no scored bytes or domain tags. Its snapshot stores the table alongside the native neural receiver; legacy pure-neural banks retain their original format.

Measure Rule C BPB over all held-out bytes, cue-to-answer top-1 accuracy at the known answer positions, and routing as the top-weight new skill over all bytes, including the initial prior. Measure original-domain pooled BPB and the maximum per-domain BPB increase versus fixed K=4. Also record shared novelty detection time, per-arm consolidation wall time, logical tensor payload bytes for the new or updated slot, serialized bank bytes, held-out answer-suffix coverage, exact resident forecasts and state hashes, and fresh-process state and output digests for all three birth paths.

## Frozen verdicts

- **ENGINEERING PASS** requires at least 364 tests passing with bare pytest and zero failures, skips or warnings, 0/100 legacy-prefix failures, exact 0.0 expanded-bank prefix difference across 50 seeds, zero false births, zero receiver donor forwards, and 5/5 exact resident preservation and fresh-process restores for the birth arms.
- **SCIENTIFIC PASS** requires the same candidate arm to achieve at least 75% held-out Rule C answer accuracy and strictly beat both Phase 4 baseline birth and fixed K=4 on that metric in every seed. It must also route at least 90% of the held-out Rule C bytes to the new cold-restored slot in all five seeds, and beat continual fine-tuning on pooled original-domain BPB while keeping every original domain's increase at or below 0.02 BPB in every seed.
- **SCIENTIFIC MIXED** requires a candidate to beat Phase 4 baseline answer accuracy in at least four seeds, stay within the per-domain 0.02 BPB retention bound in all five, and miss 75% accuracy in one or more seeds.
- **SCIENTIFIC FAIL** applies when no candidate meets the PASS or MIXED criteria. Record all unmet gates and the absolute answer scores.

Reject source or fixture drift and preserve null and negative results. Record the unrounded data in `evidence/phase5-skill-mastery-dev-001.json` following the scored run. Qualify the integrated evidence-and-documentation tip separately.
