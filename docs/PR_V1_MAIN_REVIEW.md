# Nexus Expanse v1.0-rc1 + Checkpoint Transfer — Owner PR Review Summary

Everything below is DEV-verified on Linux CPU (`PyTorch 2.10/2.15+cpu`). Protected `main` remains untouched at `bb21fbe53c0299d432e866eb4d016c25b81ee6eb` pending owner authorization.

## Verified Architecture & Milestone Arc

| Track / Gates | Branch / Tip | Engineering | Scientific (DEV) | Verified Highlights |
| :--- | :--- | :--- | :--- | :--- |
| **Core Kernel & Memory (Gates 1–8A)** | `dev/gate8-kuramoto-complete-and-hopfield-vsa` (`447b8f0`) | **PASS** | **PASS (4B/5B/6B/8A comp)** | 100% tokenless raw bytes (`0..255`), `DirectMonarchLinear`, `BytePatchReader` (`>=13.2%` speedup), `4,800/4,800` `0.0` KV-cache parity, `HopfieldVSAMemory` (`78.65%` unseen 2-hop composition, `128 KiB` bipolar codebooks). |
| **Capacity & Unified Kernel (Gates 9–14)** | `dev/gate12-14-deltanet-jepa-integration` (`57eded6`) | **PASS** | **PASS (Attention+Monarch) / FAIL (Kuramoto/JEPA)** | `Causal Attention + DirectMonarch` reached `100%` Math & Causal (`3/3` seeds); `Kuramoto` (`25%/50%`) retired for symbolic reasoning. `NexusExpanseUnified` verified (`442 KB` snapshot, `0.0` parity, `3/3` cold restore). |
| **Agency, Sleep & Verifiers (Gates 15–23)** | `dev/gate23-24-real-verifier-and-v1-rc1` (`1137a68`) | **PASS** | **PASS (Gate 16) / MIXED (18–23)** | `NexusActiveAgent` beat all controls `3/3` (`93.1%` cold return); Gate 19 PO agent beat Markov `4x`; Gate 21/23 verifier-gated synthesis solved `96/96` & `192/192` scored tasks; Gate 22 `Quine Sleep` beat Gate 20 sleep & random pruning `3/3`. |
| **v1.0-rc1 CLI & GPU Pack (Gate 24)** | `dev/gate23-24-real-verifier-and-v1-rc1` (`1137a68`) | **PASS (CPU)** | **UNVERIFIED (GPU)** | Fresh-process `train -> resume -> verify -> stream -> solve` smoke verified (`0.0` parity across 4 checks, exact `000000` residual). |
| **Checkpoint Transfer Phase 1 (Gates 25–26 + Fidelity)** | `dev/checkpoint-transfer-fidelity` (`92d43cb`) | **PASS** | **PASS (32->32 & 32->64) / FAIL (BPE zero-run)** | Near-lossless zero-update transfer on `30/30` compatible & width-doubled cases (`99.86–100%` top-1, `<=8.4e-6` bits KL, `94.1–98.6%` time saved). |
| **Checkpoint Transfer Phase 2 (BPE Folding & Offline KL)** | `dev/checkpoint-transfer-bpe-offline-kl` (`a362a93`) | **PASS** | **FAIL (Folding/On-fit KL) / MIXED (Disjoint-Corpus KL)** | Disjoint-corpus validation-stopped offline byte-KL (`192` windows, `0` receiver donor calls) beat `position_lift` and `shuffled` on `5/5` seeds (`1.2911 -> 0.9401` bits mean KL, `15.73% -> 33.57%` agreement); corpus scale—not width—is the active lever. |

## Regression & Custody Verification (Current Tip)

- `pytest -q`: **190 passed**, `0` failed, `0` skipped.
- `benchmarks/prefix_causality_dev.py`: **0/100** failures (`fails: []`).
- `git diff --check`: clean (`0` issues).
- Protected `origin/main`: untouched at `bb21fbe53c0299d432e866eb4d016c25b81ee6eb`.

## Open Items Requiring Owner Hardware / Approval

1. **Windows Consumer GPU Multi-MB Scaling Run (from `1137a68` or current tip):**
   ```powershell
   python -m nexus_expanse.cli train --device cuda --dtype bf16 --corpus-dir .\data --directory ..\nexus-gpu-run --dim 64 --context 256 --batch-per-domain 4 --steps-per-epoch 64 --epochs 64 --epoch-limit 1 --until-complete --save-snapshot ..\nexus-gpu-run\unified.pt
   ```
2. **PR Merge to `main`:** Ready for owner review once the GPU run receipt is logged or if you wish to merge the CPU-verified v1.0-rc1 + Checkpoint Transfer suite first.
