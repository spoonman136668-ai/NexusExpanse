# Phase 2D: Scaled Disjoint Transfer Corpus Pure-KL + CLI `import-skill`

Prospective follow-up to `docs/CHECKPOINT_BPE_SCALED_WIDTH_KL_PREREG.md` (`a362a93`). Phase 2C proved that scaling the disjoint unlabeled transfer corpus 3x (64 -> 192 windows) dropped BPE -> raw-byte `test` KL monotonically from `1.0428` to `0.9401` bits (`best seed 0.8721`) and showed corpus volume—not hidden width—is the active lever.

Phase 2D tests:
1. Scaling the disjoint unlabeled transfer corpus to `>= 768` 64-byte windows (drawn from disjoint `docs/*.md` and `nexus_expanse/*.py` at `study.SOURCE`, strictly excluding any 32-byte overlap with `fit`, `val`, or `test`), with `top_k=32` offline byte-marginalized caching and cosine LR decay (`<= 320` steps, validation-KL early stopping).
2. Comparing pure cached KL (`kl_weight=1.0`) vs. mixed KL (`kl_weight=0.90`) vs. `position_lift_32` vs. `shuffled_scaled_kl_32` on fresh seeds `1401–1405`.
3. Adding first-class `import-skill` to `nexus_expanse/cli.py` with mechanical `donor_forward_calls == 0`, exact `0.0` prefix causality, and exact `0.0` cold-restore round-trip verification.

## Frozen Verdict Rules

- **ENGINEERING PASS**: 100% pytest pass (`190+` tests, `0` failures/skips), `0/100` on `benchmarks/prefix_causality_dev.py`, `donor_forward_calls == 0` in all receivers, `20/20` cold restores exact, and `cli.py import-skill` tests pass.
- **SCIENTIFIC PASS**: `scaled_pure_kl_32` OR `scaled_mixed_kl_32` beats `position_lift_32` and `shuffled_scaled_kl_32` on held-out `test` KL and top-1 agreement in `5/5` seeds, AND reaches mean `test` KL `< 0.85` bits (with `>= 40%` mean top-1 agreement).
- **SCIENTIFIC MIXED**: beats `position_lift_32` and `shuffled_scaled_kl_32` in `>= 4/5` seeds without clearing `< 0.85` bits / `>= 40%` agreement.
- **SCIENTIFIC FAIL**: fails to beat `position_lift_32` or `shuffled_scaled_kl_32`.
