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

## Phase 2D Addendum (`dev/checkpoint-transfer-phase2d-scale-and-cli`)

- **CLI `import-skill`**: Added `python -m nexus_expanse.cli import-skill --donor <donor.pt> --method residual8 --width-factor <1|2> --out <skill.pt>` with mechanical `donor_forward_calls == 0`, `0.0` prefix causality, and `0.0` cold-restore round-trip verification (`tests/test_cli_import_skill.py`).
- **768-Window Disjoint-Corpus Offline Byte-KL (`93dc0fc`)**: Cleared `< 0.85` bits mean KL on both `scaled_pure_kl_32` (`0.8384` bits) and `scaled_mixed_kl_32` (`0.8393` bits, `35.32%` mean agreement, `4.6705` BPB), beating `position_lift_32` (`1.3003` bits) and `shuffled_scaled_kl_32` (`1.0890` bits) in `5/5` seeds (`1401–1405`).
- **Updated Regression Count**: **191 passed**, `0` failed, `0` skipped; `0/100` prefix causality failures.

## Phase 3A/3B DEV review

The preregistered CPU experiment is recorded in [Phase 3A evidence](../evidence/phase3-multi-skill-routing-dev-001.json). It compares six arms on file-disjoint Python and Markdown, two incompatible byte rules, and unlabeled 32-byte regime switches. Each of the five seeds uses four separately fitted donors and zero donor forwards in the receiver. The bank's frozen base state, saved weights, router settings, and forecasts were exact after fresh-process restore.

**Scientific verdict is MIXED.** Both dense routes beat uniform mixture and base-only on single-domain rule accuracy and overall BPB in 5/5 seeds. Both routed at 76.12–89.90% after warmup and met the eight-byte switch recovery target in 0/5 seeds. Sparse routing met the 40% skill-call reduction target in both suites in 5/5 seeds and beat both prediction controls in 4/5 seeds. The trade-off and misses are preserved in the raw evidence.

The integrated Linux CPU regression has **250 tests**, comprising 191 Phase 2D tests, 54 Phase 3A tests, and five Phase 3B tests. It also uses the separate 100-seed legacy prefix check. The Phase 3A exact prefix test covers 50 seeds across all three routers.

### Prepare a corpus and launch on Windows CUDA

Run these commands from the repository root on a Windows owner machine with a CUDA-capable PyTorch environment.

~~~powershell
$py = ".\.venv\Scripts\python.exe"
& $py .\scripts\prepare_gpu_corpus.py --out-dir .\data --size-mb 4.0 --seed 42
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -m nexus_expanse.cli train --device cuda --dtype bf16 --corpus-dir .\data --directory ..\nexus-gpu-run --dim 64 --context 256 --batch-per-domain 4 --steps-per-epoch 64 --epochs 64 --epoch-limit 1 --until-complete --save-snapshot ..\nexus-gpu-run\unified.pt
~~~

The generated workload is 4,000,000 raw bytes across math, code and prose train/eval files. The manifest carries exact file sizes and SHA-256 digests. The builder verifies zero exact 32-byte overlaps between train and eval before publication. Existing corpus files are preserved. The Windows CUDA launch awaits owner execution and a GPU receipt.

## Phase 3C Sticky-HMM DEV review

**DEV science MIXED.** On fresh seeds 1601–1605, HMM BMA lowered post-switch BPB in 5/5 seeds and also improved routing in 4/5 versus discounted BMA. It reached the eight-byte recovery bar on all three transitions in 0/5 seeds. HMM sparse used 2.401 to 2.555 skill calls per byte (36.13–39.98% below the four-skill reference, short of the 40% target) and improved post-switch routing over surprise sparse in 3/5 seeds and uniform mixture in 5/5. Dense HMM recovered Python to Markdown sooner in all five seeds. Rule A to Rule B missed the eight-byte bar in all five seeds for both HMM arms.

Post-warmup single-domain routing ranged from 76.28% to 86.61% for HMM BMA and 73.44% to 86.22% for HMM sparse. The 90% bar was reached in 0/5 and 0/5 seeds, respectively. The Linux CPU experiment recorded 0 forbidden donor calls and 5/5 exact cold restores.

Focused tests cover 50 exact prefix checks on both routers, cold compatibility with the existing snapshot format, CLI smoke, and a controlled abrupt switch. The combined regression contains 307 cases. Scientific qualification and Windows CUDA deployment remain separate owner decisions.

[Preregistration](PHASE3C_STICKY_HMM_ROUTER_PREREG.md) · [Raw five-seed evidence](../evidence/phase3c-sticky-hmm-routing-dev-001.json)

## Phase 4 DEV review and CLI smoke

**DEV science PASS.** Birth triggered in 5/5 seeds after 256 observed bytes. The open-world arm beat fixed K=4 on Rule C BPB and rule accuracy in 5/5 seeds. It beat continual base fine-tuning on pooled original-domain BPB and stayed within the per-domain drift limit in 5/5. The largest original-domain BPB increase was +0.01903. Cold Rule C routing was 96.88% in all five seeds, above the 80% bar. Rule C answer accuracy ranged from 10.16% to 14.06%, near the 12.5% uniform reference across its eight answer bytes. The experiment provides limited evidence of learning the cue-to-answer mapping.

The scored Linux CPU run recorded 0/80 false births, 0 receiver donor calls, 5/5 exact resident weight and forecast comparisons, and 5/5 exact cold restores. The combined suite contains 364 tests and runs from the repository root with bare pytest. Pytest loads the local imports through the project configuration and disables the unused AnyIO plugin that emitted the prior warning.

The CPU smoke runs corpus preparation, train, verify, import-skill, HMM route-stream and solve in separate processes. It verifies the training corpus and snapshot digests, loads the actual imported skill into the routed bank, and checks the exact solve residual. The import command consumes a static donor pack, while verify consumes the unified training snapshot. Windows CUDA qualification remains an owner run.

[Preregistration](PHASE4_SKILL_BIRTH_PREREG.md) · [Raw evidence](../evidence/phase4-skill-birth-dev-001.json)

## Phase 5 born-skill mastery DEV review

**DEV science FAIL.** Episodic-assisted birth reached 95.31–97.66% Rule C answer accuracy and 96.88% Rule C routing after cold restore. It beat the 32-step baseline on answer accuracy in 5/5 seeds and its largest original-domain BPB increase was +0.02826. 192-step neural birth reached 7.81–12.50% Rule C answer accuracy and 91.99–95.12% Rule C routing after cold restore. It beat the 32-step baseline on answer accuracy in 2/5 seeds and its largest original-domain BPB increase was +0.02846. The episodic mode met the answer and routing thresholds in all five seeds. Its routed Python BPB rose by 0.02826 in seed 1802 and 0.02104 in seed 1805, above the 0.02 per-domain ceiling. The 32-step Phase 4 birth arm had identical original-domain drift in all five seeds.

Median consolidation took 0.260 s for episodic-assisted birth and 1.567 s for 192-step neural birth. The suffix table added 615–670 tensor bytes and covered 95.31–97.66% of held-out answer positions in the banks that triggered.

Rule C CHECK uses fresh byte windows with recurring cue bindings. The accuracy measure includes answer positions with an unseen cue. The matched arms use the same observed encounter prefix and retain exact resident parameters and individual forecasts when expansion succeeds.

The scored CPU run logged 0/80 false births, 0 receiver donor forwards, 5/5 exact resident weight and forecast comparisons, and 5/5 exact cold restores across the three birth modes. The combined regression has 419 cases. Windows CUDA and independent scientific qualification await their own receipts.

[Preregistration](PHASE5_SKILL_MASTERY_PREREG.md) · [Unrounded five-seed evidence](../evidence/phase5-skill-mastery-dev-001.json)

## Phase 6 hierarchical prior and K=7 DEV review

**DEV science PASS.** The bank reached K=7 in 5/5 seeds. Hierarchical routing reached 88.28–94.53% minimum answer accuracy across the three new domains and 96.09–96.88% minimum cold routing. Maximum original-domain drift was 0.00402–0.00672 BPB for hierarchical routing and 0.03102–0.04778 BPB for uniform routing on the same bank. Hierarchical routing stayed within 0.015 BPB in 5/5 seeds and reduced maximum drift versus uniform routing in 5/5 and continual fine-tuning in 5/5.

The stronger 0.005 BPB target held in 1/5 seeds. Mean first routing to a new skill occurred after 2.00–2.50 observed bytes across the per-seed domain averages. The experiment used three mutually disjoint novel byte alphabets and tested the same cue bindings on fresh surrounding bytes.

The scored Linux CPU run recorded 0/160 false birth flags and 0/20 duplicate births. Rule C received at least 90% routing on 20/20 cold revisits. Receiver donor forwards were 0. Resident state and forecasts and earlier-born states were exact in 5/5 seeds, with 5/5 exact K=7 cold restores. The combined regression has 479 cases. Windows CUDA and independent scientific qualification await separate receipts.

[Preregistration](PHASE6_HIERARCHICAL_SCALING_PREREG.md) · [Unrounded evidence and per-domain results](../evidence/phase6-hierarchical-scaling-dev-001.json)

## Phase 7 sparse compute and same-alphabet DEV review

**Suite A science FAIL.** Sparse routing used 3.162–3.434 calls per byte, saving 50.95–54.82% against nominal K=7. It met the call budget in 0/5 seeds. The minimum C/D/E answer accuracy was 85.94–93.75% across seeds, meeting the mastery bound in 5/5. Maximum resident drift was 0.16429–0.29205 BPB and held within 0.015 BPB in 0/5 seeds.

**Suite B science FAIL.** The validation-only tail detector triggered in 0/5 seeds, and the mean detector triggered in 0/5. The tail arm reached 5.47–14.06% novel answer accuracy and 0.00% cold routing, with 0/5 seeds meeting both birth and mastery. Maximum original-domain drift was 0.00000 BPB, within 0.02 BPB in 5/5 seeds. Fixed K=4 scored 2.255–2.402 BPB on the novel domain.

The guarded run measured 0 receiver donor calls and 5/5 exact K=7 and 5/5 exact Suite B restores. False-birth flags were 0/160 in Suite A, 1/80 for the tail detector and 0/80 for the mean detector. The tail detector flagged one held-out Markdown probe in seed 2002, which fails the zero-false-birth engineering gate. Neither detector added a novel slot in any seed, so their zero retention drift measures the unchanged K=4 bank. The combined bare-pytest regression has 532 cases. Windows CUDA and independent acceptance await separate receipts.

[Preregistration](PHASE7_HARD_NOVELTY_SPARSE_K7_PREREG.md) · [Unrounded evidence and per-domain metrics](../evidence/phase7-hard-novelty-sparse-k7-dev-001.json)

## Phase 8 confidence and per-skill novelty DEV review

**Suite A science MIXED.** Confidence-gated routing used 6.891 calls per byte and met the 2.8 budget in 0/5 seeds. Minimum C/D/E answer accuracy was 89.06–96.88%, within the mastery gate in 5/5 seeds. Maximum original-domain drift was 0.00508–0.00647 BPB, within 0.015 BPB in 5/5 seeds. The confidence-gated path used all seven experts on every measured noninitial forecast and matched dense forecasts bit-for-bit in all five seeds.

**Suite B science FAIL.** Per-skill calibration triggered a birth in 0/5 seeds and global mean triggered in 0/5. Per-skill answer accuracy was 7.03–13.28%, and new-slot cold routing was 0.00%. The per-skill arm met the combined birth and answer gate in 0/5 seeds and the 0.02 BPB drift limit in 5/5 seeds. The retained four-slot bank produced zero drift. The closest novel probe in each seed was still 0.291–0.473 BPB below the routed skill's ceiling.

The Linux CPU run recorded 0/160 Suite A false flags and 0/80 per-skill Suite B false flags. The global mean control flagged 0/80. Receiver donor forwards were 0. Warm-to-cold forecasts and calibration identity matched in 5/5 Suite A and 5/5 Suite B seeds. The combined bare regression contains 586 cases. Windows CUDA and independent qualification await separate receipts.

[Preregistration](PHASE8_PER_SKILL_NOVELTY_PREREG.md) · [Unrounded evidence and probe diagnostics](../evidence/phase8-per-skill-novelty-dev-001.json)

## Phase 9 offline assimilation and surgical editing DEV review

**DEV science FAIL.** On held-out Python bytes, the SVD edit beat both the matched random edit and faulted baseline in 0/5 seeds. It passed validation in 0/5 and recovered at least half the injected damage in 0/5. Absolute held-out Markdown drift stayed within 0.02 BPB in 5/5. The static-cache student beat its matched CE-only control on combined held-out BPB in 5/5.

All 20 SVD candidates lowered Python validation BPB. Every candidate also changed Markdown validation BPB by more than 0.02, so the validation gate rolled back all five seeds. The zero held Markdown drift therefore measures an unchanged faulted model. The injected fault itself increased held Markdown BPB by 0.638–2.077, so the perturbation was local to a projection but its effects crossed domains.

The scored Linux CPU run recorded 0 guarded donor forwards, 0 edit autograd calls and 5/5 exact cold restores. The focused regression checked 50/50 edited and 50/50 assimilated seeds at exact 0.0 prefix difference. The experiment used a controlled rank-one fault. Independent acceptance, a real local Hugging Face donor and CUDA await separate qualification.

| Seed | Clean Python BPB | Faulted Python BPB | SVD Python BPB | Random Python BPB | SVD Markdown drift BPB | Assimilated BPB | CE-only BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2201 | 4.089749 | 4.983994 | 4.983994 | 4.983994 | 0.000000 | 4.060603 | 4.142310 |
| 2202 | 3.948360 | 4.836868 | 4.836868 | 4.836868 | 0.000000 | 4.024095 | 4.076162 |
| 2203 | 4.052474 | 4.840283 | 4.840283 | 4.842054 | 0.000000 | 4.208807 | 4.274201 |
| 2204 | 3.844440 | 4.694911 | 4.694911 | 4.694911 | 0.000000 | 4.125413 | 4.181183 |
| 2205 | 4.116732 | 4.788059 | 4.788059 | 4.782175 | 0.000000 | 4.262366 | 4.375644 |

[Preregistration](PHASE9_ASSIMILATE_AND_WEIGHT_EDIT_PREREG.md) · [Unrounded evidence and five-seed diagnostics](../evidence/phase9-weight-edit-and-assimilate-dev-001.json)

## Phase 10B shard streaming and surgical editing DEV review

**Suite A science PASS.** The one-sided eligibility gate accepted the SVD repair in 5/5 seeds. It beat both the faulted skill and the matched random repair on held Python BPB in 5/5. It recovered at least half of the injected Python damage in 5/5 and stayed within the +0.02 BPB held Markdown non-degradation bound in 5/5.

| Seed | Clean Python BPB | Faulted Python BPB | SVD Python BPB | Random Python BPB | SVD recovery | SVD Markdown delta BPB | SVD alpha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2401 | 3.921085 | 5.947686 | 3.939476 | 5.949308 | 99.09% | -0.940449 | 1.0 |
| 2402 | 3.997742 | 4.405985 | 4.042999 | 4.401545 | 88.91% | -0.880919 | 1.0 |
| 2403 | 3.873199 | 4.357072 | 4.103839 | 4.357072 | 52.33% | -0.170764 | 1.0 |
| 2404 | 4.010902 | 4.946006 | 4.110168 | 4.946006 | 89.38% | -2.330173 | 1.0 |
| 2405 | 4.039915 | 5.001277 | 4.190823 | 4.936779 | 84.30% | -1.868794 | 1.0 |

**Suite B science PASS.** The Phase 10B stack beat the shuffled donor by at least 0.50 BPB at zero updates in 5/5 seeds. After 64 matched native byte updates it beat both shuffled and scratch in 5/5. The scored stack was implemented from the named techniques because the earlier winning artifact was unavailable.

| Seed | Stack zero | Shuffled zero | Scratch zero | Stack 64 steps | Shuffled 64 steps | Scratch 64 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2401 | 6.624916 | 7.801758 | 8.199890 | 4.259813 | 4.556117 | 4.601938 |
| 2402 | 7.177815 | 7.728235 | 8.275571 | 4.307707 | 4.510014 | 4.641131 |
| 2403 | 6.606032 | 7.706646 | 8.244898 | 4.333474 | 4.561784 | 4.617523 |
| 2404 | 6.676952 | 8.030158 | 8.252018 | 4.360224 | 4.583545 | 4.660551 |
| 2405 | 6.761800 | 7.796144 | 8.226901 | 4.284863 | 4.465672 | 4.515952 |

The scored Linux CPU run measured a peak of 1 loaded shard per import, 0 guarded donor calls and 0 edit autograd calls. Cold restores were exact in 5/5 Suite A and 5/5 Suite B seeds. The focused regression measured 50/50 exact seeded shard-stream prefix checks. The scored consumer used local mock shards. The real safetensors path was covered at the reader interface with a local double.

[Preregistration](PHASE10B_SHARD_STREAM_AND_EDIT_PREREG.md) · [Unrounded five-seed evidence](../evidence/phase10b-shard-stream-and-edit-dev-001.json)

## Phase 11 SDPA and three-stage lifecycle DEV review

**DEV composition science PASS.** Assimilation beat matched scratch in 5/5 seeds. The SVD repair cleared its held-out recovery, matched-random, and retention gates in 5/5. Rule C birth, answer accuracy, cold routing, and hierarchical retention cleared together in 5/5. All three stages cleared in 5/5.

The run used one shard at a time and measured 0 guarded donor calls and 0 surgical edit autograd calls. It recorded 0/160 resident novelty flags and 5/5 exact lifecycle cold restores. The focused suite tested 50 SDPA and 50 assembled-bank seeds at exact 0.0 prefix difference. CUDA FlashAttention and independent scientific acceptance await separate qualification.

| Context | Legacy forward ms | SDPA forward ms | Legacy backward ms | SDPA backward ms | Legacy peak CPU bytes | SDPA peak CPU bytes | Training logit difference | Native eval logit difference | SDPA prefix difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 12.851 | 11.614 | 19.166 | 16.490 | 14235656 | 12496904 | 0.0 | 1.52587890625e-05 | 0.0 |
| 1024 | 42.766 | 23.747 | 57.221 | 38.234 | 33382408 | 25483272 | 0.0 | 1.52587890625e-05 | 0.0 |

| Seed | Assimilated BPB | Scratch BPB | SVD Python BPB | Random Python BPB | SVD damage recovered | Rule C accuracy | Cold routing | Max retention drift BPB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2501 | 4.307342 | 4.583466 | 4.273775 | 5.312965 | 84.99% | 100.00% | 96.88% | 0.007465 |
| 2502 | 4.316885 | 4.577774 | 4.356621 | 5.697977 | 95.73% | 100.00% | 96.88% | 0.006848 |
| 2503 | 4.247035 | 4.539935 | 4.112065 | 5.181601 | 96.32% | 100.00% | 96.88% | 0.007183 |
| 2504 | 4.442791 | 4.674963 | 4.784317 | 5.730679 | 66.03% | 96.88% | 96.88% | 0.008380 |
| 2505 | 4.516686 | 4.667958 | 4.658300 | 5.185444 | 86.75% | 100.00% | 96.88% | 0.007071 |

[Preregistration](PHASE11_SDPA_AND_3STAGE_PREREG.md) · [Unrounded profiles and five-seed evidence](../evidence/phase11-sdpa-and-3stage-dev-001.json)
