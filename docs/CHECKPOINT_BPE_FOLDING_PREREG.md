# Causal-History Folding for Multi-Byte Checkpoint-Only Transfer

Follow-up prospective experiment to `docs/CHECKPOINT_BPE_PREREG.md` targeting the open problem in `docs/CHECKPOINT_TRANSFER_FINDINGS.md`: preserving causal histories across mismatched symbol boundaries (multi-byte BPE donor -> raw-byte receiver) with zero donor forward calls and zero BPE tokenizer/merges at receiver inference.

## Data, Oracle, and Locked Baseline

- Corpus & splits: exact `benchmarks/checkpoint_bpe_challenge.py` splits (`fit`: `GATE17_PREREG.md`, `GATE20_PREREG.md`; `val`: `GATE22_PREREG.md`; `test`: `GATE23_PREREG.md`; `python` retention + `retention` probes).
- FFN converter: locked `residual8` from `evidence/checkpoint-transfer-screen.json`.
- Locked comparator: `position_lift` from `92d43cb`.
- Strict tokenless-receiver rule: zero BPE merge table, zero tokenizer calls, and `donor_forward_calls == 0` inside the receiver at conversion, training, and inference.

## Screen & Confirmation Arms

- **Screen** (`seeds = [901, 902, 903]`): compare `position_lift` (baseline) vs. `entropy_fold` (purely causal byte-transition chunk folding, zero vocab table) vs. `static_trie_fold` (static `vocab_bytes` pairwise continuation table + causal chunk-end attention masking + chunk-index position clock). Freeze the single best folding method on zero-update worst KL (then mean KL, then name) before running confirmation.
- **Confirmation** (`seeds = [1001, 1002, 1003, 1004, 1005]`): compare `winner_fold`, `position_lift`, `shuffled_donor_swap` (donor-swap specificity control), and `scratch` across zero-update and validation-selected stopping (`<= 160` updates, target `val BPB <= donor val BPB + 0.02`).

## Frozen Verdict Rules

- **ENGINEERING PASS**: 100% of existing + new tests pass (`0` failures, `0` skips), `0/100` on `benchmarks/prefix_causality_dev.py`, `0.0` prefix causality diff on the folded receiver across 50 seeds, `0` donor forward calls during conversion/training/eval, and 100% cold-restore digest equality.
- **SCIENTIFIC PASS (Breakthrough)**: `winner_fold` beats `position_lift` on zero-update KL and agreement in 5/5 confirmation seeds, beats `shuffled_donor_swap` in 5/5 seeds, and clears near-lossless (`>=99%` agreement, `<=0.02`-bit KL, `<=+0.02` BPB) with `>=25%` matched time savings.
- **SCIENTIFIC MIXED (Progress)**: `winner_fold` beats `position_lift` and `shuffled_donor_swap` in `>=4/5` seeds on zero-update KL/agreement and held-out BPB without clearing the `99%` / `0.02`-bit near-lossless bar.
- **SCIENTIFIC FAIL**: `winner_fold` fails to beat `position_lift` or `shuffled_donor_swap`.
