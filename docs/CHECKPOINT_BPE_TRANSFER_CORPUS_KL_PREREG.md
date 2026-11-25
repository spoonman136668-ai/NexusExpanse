# Fresh-Corpus, Validation-Stopped Offline Byte-KL for BPE -> Raw-Byte Transfer

Prospective follow-up to `docs/CHECKPOINT_BPE_OFFLINE_KL_PREREG.md` (`59dcaae`). On `59dcaae`, 160 steps of offline KL on the donor's own 16-window `fit` split overfit the donor's `fit` memorization and raised held-out `test` KL from `~1.16` to `~1.78` bits. This experiment tests two fixes:
1. Export the one-time offline `top_k=16` byte-marginalized cache on a **disjoint unlabeled transfer split** (`docs/GATE15_PREREG.md`, `docs/GATE16_PREREG.md`, `docs/GATE18_PREREG.md`, `docs/GATE19_PREREG.md`) that the donor never trained on, AND
2. Use **validation-stopped selection** (evaluate `val` KL every 8 steps through 96 steps; select the best `val` KL checkpoint) rather than blindly running 160 steps.

## Arms & Fresh Seeds (`1201–1205`)

1. `transfer_kl_from_position_lift`: `position_lift` warm-start + fresh-corpus offline `top_k=16` KL (`kl_weight=0.85`, `ce_weight=0.15`), validation-KL stopped (`<= 96` steps).
2. `transfer_kl_from_scratch`: `scratch` student + fresh-corpus offline `top_k=16` KL, validation-KL stopped (`<= 96` steps).
3. `position_lift`: `92d43cb` baseline (CE on `fit`, validation-BPB stopped).
4. `shuffled_transfer_kl`: `position_lift` warm-start + row-permuted fresh-corpus cache (falsification control).

## Frozen Verdict Rules

- **ENGINEERING PASS**: 100% pytest pass, `0/100` on `benchmarks/prefix_causality_dev.py`, `donor_forward_calls == 0` in all receivers, and `20/20` exact cold-restore SHA matches.
- **SCIENTIFIC PASS**: `transfer_kl_from_position_lift` beats `position_lift` and `shuffled_transfer_kl` on `test` KL (bits) and top-1 agreement in `5/5` seeds, and cuts `test` KL below `0.85` bits.
- **SCIENTIFIC MIXED**: `transfer_kl_from_position_lift` beats `position_lift` and `shuffled_transfer_kl` on `test` KL in `>= 4/5` seeds without reaching `< 0.85` bits.
- **SCIENTIFIC FAIL**: fails to beat `position_lift` or `shuffled_transfer_kl`.
