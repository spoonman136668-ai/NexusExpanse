# Width-Lifted (32 -> 64) + Scaled Disjoint-Corpus Offline Byte-KL

Prospective follow-up to `docs/CHECKPOINT_BPE_TRANSFER_CORPUS_KL_PREREG.md` (`0f05017`). On `0f05017`, a 32-wide receiver distilled on 64 disjoint transfer windows beat `position_lift`, `scratch+KL`, and `shuffled` on 5/5 seeds (`1.0428` mean bits KL, `27.00%` agreement) without clearing `< 0.85` bits. This gate tests the two remaining levers in isolation and together:
1. **3× larger disjoint transfer corpus (`192` 64-byte windows)** drawn from `GATE10`–`GATE16`, `GATE18`, `GATE19`, `GATE21` preregistration docs (strictly excluding any 32-byte overlap with `val` or `test`).
2. **Static `32 -> 64` width-lifted `position_lift` warm-start** (`lift_dense_state(state, 2)`), giving the raw-byte receiver 2× hidden width to represent multi-byte intra-token state while preserving exact donor initialization.

## Arms & Fresh Seeds (`1301–1305`)

1. `width64_transfer_kl`: 64-wide `position_lift` warm-start + 192-window offline `top_k=16` KL (`kl_weight=0.90`), validation-KL stopped (`<= 160` steps, checked every 8 steps).
2. `width32_transfer_kl`: 32-wide `position_lift` warm-start + 192-window offline `top_k=16` KL (`kl_weight=0.90`), validation-KL stopped (`<= 160` steps) — isolates corpus scaling from width doubling.
3. `position_lift_32`: `92d43cb` locked baseline (CE on `fit`, validation-BPB stopped).
4. `shuffled_width64_transfer_kl`: 64-wide `position_lift` warm-start + row-permuted 192-window cache (falsification control).

## Frozen Verdict Rules

- **ENGINEERING PASS**: 100% pytest pass, `0/100` on `benchmarks/prefix_causality_dev.py`, `donor_forward_calls == 0` in all receivers, and `20/20` exact cold-restore SHA matches.
- **SCIENTIFIC PASS**: `width64_transfer_kl` beats `position_lift_32` and `shuffled_width64_transfer_kl` on held-out `test` KL and top-1 agreement in `5/5` seeds, AND achieves mean `test` KL `< 0.85` bits.
- **SCIENTIFIC MIXED**: `width64_transfer_kl` beats `position_lift_32` and `shuffled_width64_transfer_kl` in `>= 4/5` seeds on `test` KL without clearing `< 0.85` bits.
- **SCIENTIFIC FAIL**: fails to beat `position_lift_32` or `shuffled_width64_transfer_kl`.
