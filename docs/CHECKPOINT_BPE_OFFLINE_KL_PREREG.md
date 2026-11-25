# One-Time Offline Byte-Marginalized KL for Multi-Byte BPE -> Raw-Byte Transfer

Follow-up prospective experiment to `docs/CHECKPOINT_BPE_PREREG.md` and `docs/CHECKPOINT_BPE_FOLDING_PREREG.md`. Both static checkpoint-only bridges (`position_lift` and `static_trie_fold`) hit a ~1.16–1.35 bit KL floor across the multi-byte BPE boundary with zero donor runs. This gate tests whether **one-time offline caching** of the donor's canonical-prefix byte-marginalized next-byte distribution on the `fit` split (stored as a compact top-k=16 cache, with the donor deleted before any receiver init/training) breaks that floor on held-out `test`.

## Protocol & Isolation

- Corpus & splits: exact `benchmarks/checkpoint_bpe_challenge.py` splits (`fit`: `GATE17_PREREG.md`, `GATE20_PREREG.md`; `val`: `GATE22_PREREG.md`; `test`: `GATE23_PREREG.md`).
- Producer runs ONCE offline in an isolated subprocess to:
  1. Train `DenseTokenDonorLM` (V=320, dim=32, 2 layers, 96 steps) on `fit` and export `donor.pt`.
  2. Export `fit_cache.pt` containing ONLY `top_k=16` `uint8` byte indices + `float16` probabilities per position on the `fit` split windows (computed via `canonical_byte_bpe` prefix marginalization), plus cache byte size and SHA-256.
  3. Export `val`/`test` `reference.pt` for scoring only.
- Receiver runs inside `forbid_donor_forward()` (`donor_forward_calls == 0` verified during all receiver init, adaptation, and eval; zero BPE tokenizer/merges at inference).

## Arms & Seeds

- Fresh confirmation seeds: `1101, 1102, 1103, 1104, 1105`.
- Fixed adaptation budget: up to `160` steps on `fit` (early stopping when `val BPB <= donor val BPB + 0.02`, checked every 8 steps; or full 160 steps when `kl_weight > 0` to measure distribution recovery).Wait: to keep stopping fair and comparable to `92d43cb`, use the exact same validation-selected stopping rule (`<= 160` steps, stop on `val BPB <= donor + 0.02`), AND also log the `step=160` full-budget KL/agreement for the offline-KL arms.
- Four matched arms:
  1. `offline_kl_from_position_lift`: `position_lift` warm-start + `0.5 * CE + 0.5 * KL(P_cache || P_student)` from `fit_cache.pt`.
  2. `offline_kl_from_scratch`: `scratch` student + `0.5 * CE + 0.5 * KL(P_cache || P_student)` from `fit_cache.pt`.
  3. `position_lift`: `92d43cb` baseline (CE only, no cache).
  4. `shuffled_offline_kl`: `position_lift` warm-start + shuffled/permuted `fit_cache.pt` targets (falsification control).

## Frozen Verdict Rules

- **ENGINEERING PASS**: 100% of existing + new tests pass (`0` failures, `0` skips), `0/100` on `benchmarks/prefix_causality_dev.py`, `donor_forward_calls == 0` in all receiver processes, and 100% cold-restore digest equality.
- **SCIENTIFIC PASS**: `offline_kl_from_position_lift` beats `position_lift` and `shuffled_offline_kl` on held-out `test` KL (bits) and top-1 agreement in `5/5` seeds, beats `offline_kl_from_scratch` in `>= 4/5` seeds, and reduces test KL below `0.50` bits (or achieves `>= 75%` top-1 agreement).
- **SCIENTIFIC MIXED**: `offline_kl_from_position_lift` beats `position_lift` and `shuffled_offline_kl` on test KL in `>= 4/5` seeds without reaching `< 0.50` bits / `>= 75%` agreement.
- **SCIENTIFIC FAIL**: fails to beat `position_lift` or `shuffled_offline_kl`.
