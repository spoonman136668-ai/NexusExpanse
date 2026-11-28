# Nexus Expanse v1.2.0-rc1 architecture and DEV scorecard

**Verified and kept.** The default RC training route uses causal attention and DirectMonarch. The CPU unified stream uses the exact KV path, a 5,128-byte compact BytePatchReader, bounded ConsolidatingMemory and ReliabilityArbitrator. The Gate 14 unified snapshot measured 442,344 bytes. HopfieldVSAMemory uses fixed 128 KiB codebooks and remains a decoupled relation store. NexusActiveAgent, its PO mode, VerifierGatedNexusAgent and Quine Sleep remain opt-in. The Gate 17 harness owns corpus training, resume and telemetry. The CLI exposes these pieces through train, stream, solve and verify.

**Falsified or retired for the tested role.** Pure DeltaNet and ungated/selective Kuramoto reached 25% math and 50% causal on the decisive supervised fixtures. Kuramoto FFN skipping failed its quality bar. Single-vector linear HRR used three 8 MiB FP64 codebooks and suffered interference on the relational fixture. Prior code and artifacts remain available as DEV evidence. Hybrid v1 and v2 missed their capacity gate, so the RC training path uses attention.

**Open and unverified.** Windows CUDA, consumer-GPU throughput, VRAM scaling, multi-MB learning curves and independent scientific qualification. Gate 17 qualifies the Linux CPU FP32 path. Publication of this branch is a DEV review handoff.

| Gate | DEV finding and disposition |
|---:|---|
| 1 | Historical BPB import remains unverified. Repaired prefix test reached 0/100 failures. [Import](../gate1_results.json) · [Repair](../evidence/prefix-causality-dev-002.json) |
| 2 | Bounded memory and arbitration available. Memory alone led the matched ablation, and projected fast weights missed retention. [Ablation](../evidence/ablation-dev-002.json) · [Arbitration](../evidence/arbitration-dev-001.json) · [Plasticity](../evidence/plasticity-dev-002.json) |
| 3 | Compositional local learner matched accuracy and missed symbolic-control BPB and retention. [Evidence](../evidence/relational-dev-001.json) |
| 4 | Noisy shifted composition failed. Real-corpus streaming supplied the later comparison manifest. [Composition](../evidence/gate4a-relational-dev-001.json) · [Corpus](../evidence/gate4b-real-corpus-dev-001.json) |
| 5 | Aligned relational route failed. Keep exact KV and compact 5,128-byte wake/fold. [Relational](../evidence/gate5a-relational-dev-001.json) · [KV](../evidence/gate5b-kv-patching-dev-001.json) |
| 6 | Single-vector HRR failed. Kuramoto streaming passed its separate DEV speed/quality check. [HRR](../evidence/gate6a-holographic-dev-001.json) · [Streaming](../evidence/gate6b-kuramoto-dev-001.json) |
| 7 | Windowed HRR and Kuramoto FFN skipping failed. [HRR](../evidence/gate7a-holographic-v2-dev-001.json) · [Skipping](../evidence/gate7b-kuramoto-skip-dev-001.json) |
| 8 | Keep Hopfield VSA's measured 78.65% unseen two-hop and 100% primitive recall. The joint margin and retention gate failed. [Evidence](../evidence/gate8a-hopfield-vsa-dev-001.json) |
| 9 | Tokenless Kuramoto-JEPA failed the matched micro-fixture. [Evidence](../evidence/gate9-kuramoto-jepa-dev-001.json) |
| 10 | Scaled per-domain Kuramoto-JEPA failed the relational objective comparison. [Evidence](../evidence/gate10-kuramoto-jepa-dev-001.json) |
| 11 | Both Kuramoto variants stayed at 25% math and 50% causal. Attention reached 100% on both expanded DEV tasks. [Evidence](../evidence/gate11-selective-kuramoto-complete-summary-dev-001.json) |
| 12 | Pure DeltaNet streamed 11.52–13.44× faster than the repository KV cache and stayed at chance. MIXED. [Evidence](../evidence/gate12-deltanet-dev-001.json) |
| 13 | Attention JEPA and hybrid objectives missed the joint variance and retrieval bars. FAIL. [Evidence](../evidence/gate13-jepa-proven-backbone-dev-001.json) |
| 14 | Unified engineering PASS, exact stream and restore, 100% math/causal, and 442,344-byte snapshot. Memory-only BPB led fusion. [Evidence](../evidence/gate14-unified-v1-dev-001.json) |
| 15 | Hybrid v1 reduced cache bytes and seed 44 math fell to 50%. FAIL. [Evidence](../evidence/gate15-hybrid-dev-001.json) |
| 16 | Persistent active planner beat all controls on the synthetic goals. 93.1% cold return. PASS. [Evidence](../evidence/gate16-agency-dev-001.json) |
| 17 | Keep resumable scaling harness. Linux CPU FP32 smoke PASS. Windows and CUDA open. [Evidence](../evidence/gate17-harness-smoke-dev-001.json) |
| 18 | Hybrid v2 was 10.88–13.55× faster on the repository CPU KV comparison. Seed 44 causal fell to 50%. MIXED. [Evidence](../evidence/gate18-hybrid-v2-dev-001.json) |
| 19 | PO planner beat Markov in all seeds and reached 88.9% cold return. Tied fresh PO on seed 42 totals. MIXED. [Evidence](../evidence/gate19-po-agency-dev-001.json) |
| 20 | Sleep beat no-sleep in all seeds. One seed cleared the margin against both controls. MIXED. [Evidence](../evidence/gate20-sleep-consolidation-dev-001.json) |
| 21 | Verifier synthesis solved 96/96 cold compositions. The ungated control matched all Phase 2 solve rates. MIXED. [Evidence](../evidence/gate21-verifier-synthesis-dev-001.json) |
| 22 | Quine Sleep kept all 65 witnesses and beat random prune and Gate 20 sleep. It missed the byte-matched no-sleep margin. MIXED. [Evidence](../evidence/gate22-quine-sleep-dev-001.json) |
| 23 | Bounded AST synthesis solved every scored problem on the first call in both persistent arms. Neither encountered an eligible failed trace. MIXED. [Evidence](../evidence/gate23-real-verifier-dev-001.json) |

## Phase 3 multi-skill routing and corpus preparation

MultiSkillBank freezes the base and imported checkpoint receivers. The two dense Bayes routers compare the likelihood of earlier bytes, then mix probabilities or select one skill. The sparse router evaluates its active skill between surprise-triggered and periodic probes. The bank consumes raw values 0–255 and forecasts before observing the target byte. Its tensor-only snapshot preserves ordered skill weights and routing parameters.

**Scientific verdict is MIXED.** Both dense routes beat uniform mixture and base-only on single-domain rule accuracy and overall BPB in 5/5 seeds. Both routed at 76.12–89.90% after warmup and met the eight-byte switch recovery target in 0/5 seeds. Sparse routing met the 40% skill-call reduction target in both suites in 5/5 seeds and beat both prediction controls in 4/5 seeds. The trade-off and misses are preserved in the raw evidence.

Linux CPU coverage includes **250 pytest cases** and the separate 100-seed legacy prefix check. The Phase 3A checks verify exact prefixes across 50 seeds and zero donor forwards. The five-seed experiment checks exact fresh-process restore. [Raw per-arm evidence](../evidence/phase3-multi-skill-routing-dev-001.json)

Build a bank from compatible static donor packs in the same process that will save it. The CLI can then cold-load that bank and forecast after each supplied chunk.

~~~python
from nexus_expanse.checkpoint_transport import load_skill
from nexus_expanse.multi_skill_router import MultiSkillBank, save_bank

base = load_skill("base_python.pt", method="residual8")
skills = {
    name: load_skill(name + ".pt", method="residual8")
    for name in ("rule_A", "rule_B", "markdown_prose")
}
save_bank(MultiSkillBank(base, skills), "bank.pt")
~~~

~~~powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m nexus_expanse.cli route-stream --bank .\bank.pt --router bayes_bma --input-hex 64656620 --input-hex 6628293a
~~~

The command reports the active skill, the named routing weights and one next-byte forecast per chunk in a JSON object. All chunks share one causal history, with a total input limit of 4,096 bytes. It verifies exact prefix causality before emitting results.

The offline corpus builder produces six Gate 17 raw files, verifies train/eval 32-byte disjointness, and records size and SHA-256 in the manifest. The default pack contains 4 MB. Windows CUDA qualification requires a separate owner run. [Corpus build and Windows PowerShell sequence](PR_V1_MAIN_REVIEW.md#prepare-a-corpus-and-launch-on-windows-cuda)

## Phase 3C Sticky-HMM routing

The phase adds `hmm_bma` and `hmm_sparse` to `route-stream` with the existing tensor-only bank format. Both report predictive regime weights for the next byte. HMM BMA uses those weights to mix probabilities. HMM sparse forecasts through the top skill, propagates the sticky prior between probes, and refilters the last four observed bytes at a probe. The replay updates stored state for subsequent forecasts.

**DEV science MIXED.** On fresh seeds 1601–1605, HMM BMA lowered post-switch BPB in 5/5 seeds and also improved routing in 4/5 versus discounted BMA. It reached the eight-byte recovery bar on all three transitions in 0/5 seeds. HMM sparse used 2.401 to 2.555 skill calls per byte (36.13–39.98% below the four-skill reference, short of the 40% target) and improved post-switch routing over surprise sparse in 3/5 seeds and uniform mixture in 5/5. Dense HMM recovered Python to Markdown sooner in all five seeds. Rule A to Rule B missed the eight-byte bar in all five seeds for both HMM arms. Post-warmup single-domain routing ranged from 76.28% to 86.61% for HMM BMA and 73.44% to 86.22% for HMM sparse. The 90% bar was reached in 0/5 and 0/5 seeds, respectively. The Linux CPU experiment recorded 0 forbidden donor calls and 5/5 exact cold restores.

[Preregistration](PHASE3C_STICKY_HMM_ROUTER_PREREG.md) · [Raw five-seed evidence](../evidence/phase3c-sticky-hmm-routing-dev-001.json)

## Phase 4 online skill birth

OpenWorldSkillBank checks the last 64 observed bytes and triggers only when every resident exceeds eight bits per byte. The new slot is cloned from Python, gradient-adapted on observed raw bytes, and frozen before registration. Its snapshot is readable by the existing bank loader and HMM route-stream command. All four resident hashes are checked before registration.

**DEV science PASS.** Birth triggered in 5/5 seeds after 256 observed bytes. The open-world arm beat fixed K=4 on Rule C BPB and rule accuracy in 5/5 seeds. It beat continual base fine-tuning on pooled original-domain BPB and stayed within the per-domain drift limit in 5/5. The largest original-domain BPB increase was +0.01903. Cold Rule C routing was 96.88% in all five seeds, above the 80% bar. Rule C answer accuracy ranged from 10.16% to 14.06%, near the 12.5% uniform reference across its eight answer bytes. The experiment provides limited evidence of learning the cue-to-answer mapping.

The scored Linux CPU run recorded 0/80 false births, 0 receiver donor calls, 5/5 exact resident weight and forecast comparisons, and 5/5 exact cold restores. The combined suite contains 364 tests and runs from the repository root with bare pytest. Pytest loads the local imports through the project configuration and disables the unused AnyIO plugin that emitted the prior warning.

[Preregistration](PHASE4_SKILL_BIRTH_PREREG.md) · [Raw evidence](../evidence/phase4-skill-birth-dev-001.json)

## Phase 5 episodic-assisted consolidation

The episodic mode learns a frozen suffix table from the observed encounter and runs 32 neural steps on the original birth clone. A three-byte suffix with one consistent successor mixes 90% stored probability with 10% neural probability. Unseen and conflicting suffixes use the neural forecast. The 192-step consolidation mode uses the same encounter with a pure neural receiver. Native bank snapshots retain their existing format. The episodic snapshot stores sorted keys and raw successors in a versioned bank payload for cold routing.

**DEV science FAIL.** Episodic-assisted birth reached 95.31–97.66% Rule C answer accuracy and 96.88% Rule C routing after cold restore. It beat the 32-step baseline on answer accuracy in 5/5 seeds and its largest original-domain BPB increase was +0.02826. 192-step neural birth reached 7.81–12.50% Rule C answer accuracy and 91.99–95.12% Rule C routing after cold restore. It beat the 32-step baseline on answer accuracy in 2/5 seeds and its largest original-domain BPB increase was +0.02846. The episodic mode met the answer and routing thresholds in all five seeds. Its routed Python BPB rose by 0.02826 in seed 1802 and 0.02104 in seed 1805, above the 0.02 per-domain ceiling. The 32-step Phase 4 birth arm had identical original-domain drift in all five seeds.

Median consolidation took 0.260 s for episodic-assisted birth and 1.567 s for 192-step neural birth. The suffix table added 615–670 tensor bytes and covered 95.31–97.66% of held-out answer positions in the banks that triggered.

The scored CPU run logged 0/80 false births, 0 receiver donor forwards, 5/5 exact resident weight and forecast comparisons, and 5/5 exact cold restores across the three birth modes.

[Preregistration](PHASE5_SKILL_MASTERY_PREREG.md) · [Unrounded five-seed evidence](../evidence/phase5-skill-mastery-dev-001.json)

## Phase 6 resident-anchored routing and sequential birth

The hierarchical HMM fixes the born group's initial mass at 2% and its share of a resident's switch probability at 10%. Adding born slots divides that group budget. Without a born slot the router delegates to the standard HMM exactly. Snapshots record the resident names. Legacy snapshots with no group metadata treat every saved slot as resident. The CLI exposes the new path as hmm_bma_hierarchical.

**DEV science PASS.** The bank reached K=7 in 5/5 seeds. Hierarchical routing reached 88.28–94.53% minimum answer accuracy across the three new domains and 96.09–96.88% minimum cold routing. Maximum original-domain drift was 0.00402–0.00672 BPB for hierarchical routing and 0.03102–0.04778 BPB for uniform routing on the same bank. Hierarchical routing stayed within 0.015 BPB in 5/5 seeds and reduced maximum drift versus uniform routing in 5/5 and continual fine-tuning in 5/5.

The stronger 0.005 BPB target held in 1/5 seeds. Mean first routing to a new skill occurred after 2.00–2.50 observed bytes across the per-seed domain averages. The experiment used three mutually disjoint novel byte alphabets and tested the same cue bindings on fresh surrounding bytes.

The scored Linux CPU run recorded 0/160 false birth flags and 0/20 duplicate births. Rule C received at least 90% routing on 20/20 cold revisits. Receiver donor forwards were 0. Resident state and forecasts and earlier-born states were exact in 5/5 seeds, with 5/5 exact K=7 cold restores.

[Preregistration](PHASE6_HIERARCHICAL_SCALING_PREREG.md) · [Unrounded evidence and per-domain results](../evidence/phase6-hierarchical-scaling-dev-001.json)

## Phase 7 sparse routing and calibrated tail novelty

The hierarchical sparse HMM shares the dense initial and transition priors. It calls one skill between probes and refilters up to two observed bytes at a probe. Tail novelty keeps a frozen validation limit per resident and requires every score to exceed its own limit. Suite B applies an identical ASCII answer-byte bijection to the static A/B checkpoints and their data, then varies the novel cue-to-answer binding.

**Suite A science FAIL.** Sparse routing used 3.162–3.434 calls per byte, saving 50.95–54.82% against nominal K=7. It met the call budget in 0/5 seeds. The minimum C/D/E answer accuracy was 85.94–93.75% across seeds, meeting the mastery bound in 5/5. Maximum resident drift was 0.16429–0.29205 BPB and held within 0.015 BPB in 0/5 seeds.

**Suite B science FAIL.** The validation-only tail detector triggered in 0/5 seeds, and the mean detector triggered in 0/5. The tail arm reached 5.47–14.06% novel answer accuracy and 0.00% cold routing, with 0/5 seeds meeting both birth and mastery. Maximum original-domain drift was 0.00000 BPB, within 0.02 BPB in 5/5 seeds. Fixed K=4 scored 2.255–2.402 BPB on the novel domain.

The guarded run measured 0 receiver donor calls and 5/5 exact K=7 and 5/5 exact Suite B restores. False-birth flags were 0/160 in Suite A, 1/80 for the tail detector and 0/80 for the mean detector. The tail detector flagged one held-out Markdown probe in seed 2002, which fails the zero-false-birth engineering gate. Neither detector added a novel slot in any seed, so their zero retention drift measures the unchanged K=4 bank.

[Preregistration](PHASE7_HARD_NOVELTY_SPARSE_K7_PREREG.md) · [Unrounded evidence and per-domain metrics](../evidence/phase7-hard-novelty-sparse-k7-dev-001.json)

## Phase 8 confidence-gated routing and per-skill novelty

Confidence gating saves an empirical byte NLL limit per skill in the snapshot and uses one expert only when its predictive mass reaches the threshold. A full-bank step mixes all experts in log space and refilters the latest sparse emission when replaying. The maximum predictive mass under the default transition is 0.95. The novelty detector compares the observed bank BPB with the ceiling from the dominant skill's own ID validation windows.

**Suite A science MIXED.** Confidence-gated routing used 6.891 calls per byte and met the 2.8 budget in 0/5 seeds. Minimum C/D/E answer accuracy was 89.06–96.88%, within the mastery gate in 5/5 seeds. Maximum original-domain drift was 0.00508–0.00647 BPB, within 0.015 BPB in 5/5 seeds. The confidence-gated path used all seven experts on every measured noninitial forecast and matched dense forecasts bit-for-bit in all five seeds.

**Suite B science FAIL.** Per-skill calibration triggered a birth in 0/5 seeds and global mean triggered in 0/5. Per-skill answer accuracy was 7.03–13.28%, and new-slot cold routing was 0.00%. The per-skill arm met the combined birth and answer gate in 0/5 seeds and the 0.02 BPB drift limit in 5/5 seeds. The retained four-slot bank produced zero drift. The closest novel probe in each seed was still 0.291–0.473 BPB below the routed skill's ceiling.

The Linux CPU run recorded 0/160 Suite A false flags and 0/80 per-skill Suite B false flags. The global mean control flagged 0/80. Receiver donor forwards were 0. Warm-to-cold forecasts and calibration identity matched in 5/5 Suite A and 5/5 Suite B seeds.

[Preregistration](PHASE8_PER_SKILL_NOVELTY_PREREG.md) · [Unrounded evidence and probe diagnostics](../evidence/phase8-per-skill-novelty-dev-001.json)

## Phase 9 static donor caches and surgical editing

The local exporter partitions fixed-length raw windows into checksummed shards. Each observed prefix supplies a top-32 next-byte distribution, stored as unsigned byte IDs and normalized FP16 probabilities alongside the original bytes. The DenseTokenDonorLM backend reads weights-only checkpoints. The optional Hugging Face backend resolves local safetensors and tokenizer.json, then decodes GPT-2 byte-level or SentencePiece spellings. The exporter is the only inference path that loads a donor. The native `assimilate` CLI verifies shard byte counts and SHA-256 hashes, trains with cross-entropy and optional cached KL, and writes a standalone skill with exact CPU prefix and cold-archive checks.

The inference-only editor derives a direction from the top singular vector of paired differences in per-window mean attention contexts. It supports subspace ablation, rank-one steering and bias shifts. An edit to a structured FFN layer materializes just that target affine in the exported skill. The verified-edit API gates a copy on exact prefix and archive parity, lower fault BPB, and bounded retain BPB drift. The `weight-edit` CLI provides attention inspection and ablation with an exactness receipt. Its `--fault-hex` and `--retain-hex` options take equal-length observed byte windows.

**DEV science FAIL.** On held-out Python bytes, the SVD edit beat both the matched random edit and faulted baseline in 0/5 seeds. It passed validation in 0/5 and recovered at least half the injected damage in 0/5. Absolute held-out Markdown drift stayed within 0.02 BPB in 5/5. The static-cache student beat its matched CE-only control on combined held-out BPB in 5/5.

All 20 SVD candidates lowered Python validation BPB. Every candidate also changed Markdown validation BPB by more than 0.02, so the validation gate rolled back all five seeds. The zero held Markdown drift therefore measures an unchanged faulted model. The injected fault itself increased held Markdown BPB by 0.638–2.077, so the perturbation was local to a projection but its effects crossed domains.

[Preregistration](PHASE9_ASSIMILATE_AND_WEIGHT_EDIT_PREREG.md) · [Unrounded evidence and five-seed diagnostics](../evidence/phase9-weight-edit-and-assimilate-dev-001.json)

## Phase 10B tensor-only shard transport and one-sided editing

The sharded transport reads local safetensors and tokenizer indexes. It reduces requested tensors into a Procrustes-aligned native spectral basis while each shard is open. A shard is released before the next one opens. The recipient uses mined MLP key/value pairs, DirectMonarch projections with low-rank spectral corrections, and a static raw-byte prior derived from token-leading probability mass. An optional Engram byte logit table contributes to this prior. Shard and tokenizer identities, peak loaded shards, donor-call counts, exact prefix parity and cold-archive parity are recorded. The local weights-only mock format uses the same mapping for CPU regression.

The `assimilate-zero-exec` CLI writes the initialization and receipt. Native adaptation blends the static table at alpha 0.25 on the first step and decays its weight linearly to zero over the matched update budget. The one-sided edit gate accepts signed retain BPB changes at or below +0.02 when fault BPB improves and exact checks pass.

**Suite A science PASS.** The one-sided eligibility gate accepted the SVD repair in 5/5 seeds. It beat both the faulted skill and the matched random repair on held Python BPB in 5/5. It recovered at least half of the injected Python damage in 5/5 and stayed within the +0.02 BPB held Markdown non-degradation bound in 5/5.

**Suite B science PASS.** The Phase 10B stack beat the shuffled donor by at least 0.50 BPB at zero updates in 5/5 seeds. After 64 matched native byte updates it beat both shuffled and scratch in 5/5. The scored stack was implemented from the named techniques because the earlier winning artifact was unavailable.

[Preregistration](PHASE10B_SHARD_STREAM_AND_EDIT_PREREG.md) · [Unrounded five-seed evidence](../evidence/phase10b-shard-stream-and-edit-dev-001.json)

## Phase 11 causal SDPA and composed skill archives

Opt-in SDPA projects Q, K and V through the native attention tensors and uses implicit causality. A custom mask still follows the masked attention path. Evaluation uses fixed geometry for exact prefixes. The training CLI records the attention backend and accepts contexts through 4096 when SDPA is selected. The established CPU unified cache runs the same exported tensors with legacy attention.

The composed lifecycle adapts a tensor-only imported skill, verifies an edit on a detached copy, and restores the edited skill as the resident used for Rule C birth. Bank archives reuse the standalone skill reconstruction for structured FFN layouts, fixed evaluation geometry and static-prior tensors. The warm and fresh-process expanded banks are checked for exact states and forecasts. The hierarchical router measures resident retention against the pre-birth bank.

**DEV composition science PASS.** Assimilation beat matched scratch in 5/5 seeds. The SVD repair cleared its held-out recovery, matched-random, and retention gates in 5/5. Rule C birth, answer accuracy, cold routing, and hierarchical retention cleared together in 5/5. All three stages cleared in 5/5.

The run used one shard at a time and measured 0 guarded donor calls and 0 surgical edit autograd calls. It recorded 0/160 resident novelty flags and 5/5 exact lifecycle cold restores. The focused suite tested 50 SDPA and 50 assembled-bank seeds at exact 0.0 prefix difference. CUDA FlashAttention and independent scientific acceptance await separate qualification.

[Preregistration](PHASE11_SDPA_AND_3STAGE_PREREG.md) · [Unrounded profiles and five-seed evidence](../evidence/phase11-sdpa-and-3stage-dev-001.json)

## Phase 12 configurable shard transport and static Engram

The native importer accepts `--student-dim` values 32, 64, 128 and 256 and `--student-layers` values 2, 4 and 6. Defaults are 32 and two. Standalone and bank archives reconstruct depth from their FFN layouts. Large vocabularies still use a row-chunked FP32 Gram matrix for the shared embedding and readout basis.

`--use-engram` opts into indexed DeepSeek text Engram pairs at layers 1 and 14. The importer hashes each standalone byte token with padded history, reads and dequantizes only the selected table rows, and projects the shared value slice of `wkv` into the student basis. A quarter of the mean projected value is added before the static prior is normalized. The native skill carries only the resulting byte prior. Existing checkpoints with an explicit `engram.weight` byte table retain their previous default behavior.

The public DeepSeek-V4.1-Flash-1017 headers at `08c8657f02291e11b6f6b63e0e16014f8e748518` put the two Engram tables on separate shards of about 101.5 GB each. The Hub route releases a shard before downloading the next. Engram mode needs free space for the largest selected shard. The offline fixture verified bounding the stream to one shard. Live checkpoint import and memory use on Colab remain unverified. [DeepSeek Engram reference](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/inference/engram.py)

**DEV science PASS.** Both 64-wide arms beat both same-width controls on both stages in 5/5 seeds and beat the 32-wide baseline after adaptation in 5/5. The experiment used a local BPE donor and a FIT-only byte table. Independent scientific qualification remains open.

[Preregistration](PHASE12_SCALED_ZERO_EXEC_PREREG.md) · [Five-seed BPB tables](PR_V1_MAIN_REVIEW.md#phase-12-scaled-zero-execution-assimilation-dev-review) · [Unrounded receipt](../evidence/phase12-scaled-zero-exec-dev-001.json)

## Phase 13 native warm-start training and generation

`train --init-skill` restores a local standalone skill and verifies its width and depth against the training geometry. It retains the untied byte readout and materializes corrected FFN weights into native affine layers. Matching position rows transfer from the skill. Any remaining target rows use the seeded training initialization. The run and every epoch intent bind the initial skill's SHA-256. A stored byte prior contributes to the first epoch with a linear decay to zero. Export uses a regular unified snapshot with the trained native dense FFNs and untied readout.

`generate` feeds UTF-8 or hex prompt bytes through a fresh unified snapshot restore, then emits greedy or seeded nucleus-sampled continuation bytes. The cached and batch paths share the same readout. Hex represents every output byte exactly. Text decoding replaces malformed UTF-8. The optional `--adapt` flag applies online adaptation within the restored session.

**DEV science PASS.** With matched native geometry and 64 update steps, warm-start beat both scratch and shuffled on pooled held-out BPB in 5/5 seeds. It beat both controls in code and prose in 5/5 and in math in 2/5.

| Seed | Warm-start pooled BPB | Scratch pooled BPB | Shuffled pooled BPB |
| --- | ---: | ---: | ---: |
| 2701 | 4.554312 | 4.709394 | 4.844375 |
| 2702 | 4.351882 | 4.897314 | 4.567026 |
| 2703 | 4.617996 | 4.635069 | 4.766564 |
| 2704 | 4.507034 | 4.693228 | 4.591118 |
| 2705 | 4.331808 | 4.701732 | 4.889546 |

The frozen Linux CPU run recorded exact cold resume and restore, 0.0 unified parity and bit-exact greedy continuation in 5/5 seeds, with zero consumer donor forwards. Independent scientific qualification and GPU testing remain open. [Preregistration](PHASE13_WARMSTART_AND_GENERATE_PREREG.md) · [Unrounded pooled and per-domain results](../evidence/phase13-warmstart-train-dev-001.json)

## Phase 14 batched decode and document-level corpus packing

For fixed-memory generation, the patch reader records the prompt events and computes a forecast from the final retained window. Rollover rebuilds attention keys, values and feed-forward cursors at their rebased absolute positions in one batch. Adapted generation still observes each byte and uses the same batched rollover. CPU remains the default. CUDA moves the decode backbone and cache to the selected device while patch counts, episodic memory and arbitration remain on CPU.

The five CPU rollover profiles had zero byte mismatches against `stream --no-adapt` and zero donor forwards. Median decode speedup was 18.71×. Timings cover decode after prefill and exclude snapshot restore. The owner measured T4 CUDA decode on a separate Colab workload. [Owner Colab receipt](LIVE_COLAB_RECEIPTS.md#phase-14-real-text-training-verification-and-t4-cuda-generation)

| Seed | Prompt bytes | Legacy ms per byte | Batched ms per byte | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 2801 | 8 | 5.382 | 0.825 | 6.53× |
| 2802 | 26 | 23.434 | 1.572 | 14.91× |
| 2803 | 65 | 23.595 | 1.261 | 18.71× |
| 2804 | 148 | 68.527 | 2.297 | 29.83× |
| 2805 | 111 | 29.710 | 1.244 | 23.89× |

`scripts/prepare_real_corpus.py` packs whole documents under a total output cap with a 50,000-byte minimum per file. Selection discards documents that would introduce cross-split overlap. The final exact check includes windows spanning document boundaries and checks all training files against all evaluation files before publication. Files use the FUSE-safe immutable publisher, with the manifest written last. The frozen repo pack contained 798,670 bytes across six files with zero exact 32-byte overlap. Its math domain used the opt-in synthetic fallback. The owner later verified TinyStories streaming on the Phase 14B release. [Preregistration](PHASE14_FAST_GENERATE_AND_REAL_CORPUS_PREREG.md) · [Unrounded timings and corpus manifest](../evidence/phase14-fast-generate-dev-001.json) · [Owner Colab receipt](LIVE_COLAB_RECEIPTS.md#phase-14b-tinystories-hub-streaming-and-cuda-decode)

The initial candidate at `b0aa482b76707c8df56679ed6bd575be21b8e625` had 686 passing tests and one failure. Editing `patching.py` changed the frozen Gate 9 corpus digest. The final candidate keeps that file and its Gate 9 fixtures byte-identical to the Phase 13 base. The corrected implementation was frozen and scored once against the same protocol. The first run's 16.14× score is preserved separately (receipt SHA-256 `2e0d8f7c5f68c4d0051ba3d2f101f130f88280e1ae5302b9de4327770ba51661`).

## Phase 15 deterministic learning-rate schedule and byte repetition control

For cosine training, Gate 17 derives the one-based update index from the epoch cursor and checks the restored AdamW step count before updating weights. The rate warms linearly to the base rate over the first 5% of steps rounded down, with a minimum of one warmup step. Cosine decay ends at 10% of the base rate. A one-step run uses that endpoint. The identity records an explicit rate and schedule when either differs from the legacy constant `.003` configuration. Existing source, tree, corpus and initializer checks still apply on resume.

The unified archive reader caps input at 80,000,000 bytes. The expanded allowance applies to the supported attention layout with FFN multiplier 2 and up to width 512, eight layers and 4,096 learned positions. Other geometries retain the 32,000,000-byte limit. The reader still uses `weights_only=True`. Standalone initial skill inputs retain their previous limit.

With a penalty above `1.0`, generation multiplies the log score of each distinct recent byte by the penalty before selecting a greedy byte or applying temperature and nucleus sampling. A trailing raw-byte window starts with the prompt tail and rolls forward after each generated byte. Penalty `1.0` uses the original scoring and sampling path.

**DEV science MIXED.** On the same FIT-trained native skill and the same 128 updates per arm, cosine lowered pooled held-out BPB in 2/5 seeds and raised mean pooled BPB by 0.012550. It lowered mean math BPB from 4.944520 to 4.740830. Mean code BPB rose from 4.371421 to 4.486561, and mean prose BPB rose from 4.645026 to 4.771228.

| Seed | Constant pooled BPB | Cosine pooled BPB |
| --- | ---: | ---: |
| 2901 | 4.597222 | 4.657616 |
| 2902 | 4.673209 | 4.685508 |
| 2903 | 4.551266 | 4.578979 |
| 2904 | 4.787200 | 4.776940 |
| 2905 | 4.659382 | 4.631987 |

Both arms matched uninterrupted training at every epoch in all five seeds, including optimizer and RNG checkpoint hashes. Cold restore and unified verification were exact, and donor forward calls were zero. The disabled penalty matched `stream --no-adapt` in 5/5 seeds. Penalty `2.0` broke the controlled repeated-space loop in 5/5 seeds, and seeded sampling repeated exactly. Independent scientific qualification and full-scale GPU training remain open. [Preregistration](PHASE15_COSINE_LR_AND_SAMPLING_PREREG.md) · [Unrounded pooled and per-domain evidence](../evidence/phase15-cosine-lr-dev-001.json)

## Phase 15B always-awake generation

`BytePatchReader` wakes for the first byte and when the smoothed probability of the observed transition falls below its default `2^-4` threshold. A likely transition can keep autoregressive generation on the compact bigram fallback. In particular, repeatedly choosing a space can keep the reader asleep.

`generate --always-wake` sets the restored reader's threshold to `2.0` before prefill, above every valid transition probability. Both batched and adaptive prefill treat each prompt byte as a wake event, and each generated byte advances the neural path. The option applies to the current generation session. Seeded sampling stays deterministic in the offline test, and the default greedy path still matches `stream --no-adapt` exactly. The owner reported that the same threshold cleared the repeated-space loop on four Colab prompts. [Owner Colab receipt](LIVE_COLAB_RECEIPTS.md#phase-15-cosine-training-and-always-awake-generation)

## Phase 17 focused domain and single-stream training

In six-file mode, a focused update draws the usual `3 × batch_per_domain` sequences exclusively from the selected domain. The deterministic block permutation advances by that full batch width. All three held-out files still get per-domain BPB reports. The exported compact bigram and episodic inputs fit from the selected training bytes. The selected domain is part of the immutable epoch identity. Default `all` keeps the legacy batch order and identity fields.

Raw-file mode uses `batch_per_domain` sequences per update and stores separate train and eval digests under `stream`. File identity, minimum length, and content digest are checked before training. The packager can produce that pair from a directory, a file split into contiguous spans, or whole Hugging Face text records. It verifies exact 32-byte disjointness including windows across output joins before publishing. The exact index uses the smaller split for one-pair validation. Large single pools use a bounded sample for collision ranking when the first assignment leaves too little compatible data. A local synthetic binary run produced 99,936,768 bytes with zero shared windows at the 100 MB cap. Live Hub packing at that scale remains unverified.

**DEV science PASS.** With identical initialization and 6,144 valid training targets in 64 updates per arm, focused prose beat the three-domain baseline in 5/5 seeds. Mean prose BPB fell from 5.346958 to 4.843014. Mean math BPB rose from 5.043168 to 7.663808. Mean code BPB was 5.150753 for mixed and 5.100015 for focused. The first corpus candidate for seed 3103 failed the exact overlap check. The frozen data-only rule selected corpus seed 13103 and kept the training seed at 3103. Independent scientific qualification remains open.

| Seed | Mixed prose BPB | Focused prose BPB |
| --- | ---: | ---: |
| 3101 | 5.283214 | 4.830429 |
| 3102 | 5.399978 | 4.907772 |
| 3103 | 5.361705 | 4.844901 |
| 3104 | 5.264998 | 4.826621 |
| 3105 | 5.424893 | 4.805348 |

Both new modes matched their uninterrupted epoch and final hashes and INT8 exports in all five seeds. CLI verification had `0.0` round-trip and causal differences, and `generate --always-wake` accepted every tested snapshot with zero donor forward calls. [Preregistration](PHASE17_DOMAIN_FOCUS_PREREG.md) · [Unrounded scores and custody evidence](../evidence/phase17-domain-focus-dev-001.json)

## Phase 18 multimodal byte skills

Audio frames store a pitch marker and 64 μ-law sample levels. Image frames store a tile marker and 64 row-major grayscale values. Telemetry packets store a type marker, a positive little-endian counter delta, three correlated byte channels, and a checksum. The generators can reject overlapping 32-byte windows between FIT, CAL, and CHECK. All three use the existing 256-way byte head, including zero and bytes outside UTF-8.

`MultiSkillBank.from_unified_snapshots` accepts named unified archives with different widths, depths, and contexts. A receiver exposes the restored neural backbone with its own causal context. Its archived unified cursor and adaptive memory do not feed the router. Bank snapshots preserve the embedded archive bytes and load with weights-only deserialization. INT8 reduces storage and the receiver forecasts in FP32. The bank still restores legacy imported skills and episodic entries.

Sticky-HMM BMA updates routing weights from observed-byte likelihoods and keeps each resident model frozen. `OpenWorldSkillBank` tests novelty on a trailing window. On a trigger it copies the base receiver, adapts the copy on observed bytes and repacks its neural archive at the source precision before registering it with suffix memory. The relative mixture weights can change after registration even when the resident weights and component forecasts are exact.

The four `128×4` residents used a combined 16,384 training targets and 256 updates, matching the `128×4` monolith's budget. On the switched stream, Sticky-HMM BMA averaged 5.2269 BPB, the uniform mixture 6.2814, the best single model 5.9257, and the labeled oracle 5.1973. The switched-stream arm passed in all five seeds.

| Seed | HMM BPB | Uniform BPB | Best single BPB | Oracle BPB | Settled routing |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3201 | 5.3932 | 6.3941 | 6.0717 | 5.3870 | 87.2% |
| 3202 | 5.2975 | 6.4213 | 6.2466 | 5.3156 | 89.4% |
| 3203 | 5.2005 | 6.1526 | 5.6813 | 5.2122 | 86.6% |
| 3204 | 5.1913 | 6.3363 | 5.7605 | 5.1036 | 94.5% |
| 3205 | 5.0519 | 6.1026 | 5.8683 | 4.9682 | 90.2% |

The telemetry arm passed in 0/5 seeds. Seeds 3201 and 3205 triggered at byte 160 during the encounter. Their held telemetry BPB fell from 7.8046 to 7.2707 and from 8.9472 to 7.4520. Settled cold telemetry routing reached 27.7% and 37.5%, below the 80% per-domain target. Both seeds also flagged the held audio control. Maximum upward drift across residents was 0.0151 BPB, within the +0.05 limit. The other three seeds registered no new skill. The joint DEV verdict is **MIXED**.

All five seeds had zero shared 32-byte train/eval windows, exact `0.0` prefix differences, exact cold bank forecasts, unchanged resident weights, and zero donor forwards. Individual INT8 snapshots compressed by at least 3.104× versus FP32. The complete four-skill bank was about 4× the size of the single INT8 monolith. The two five-skill banks each used about 4.20 MB. These are Linux CPU results on repository prose/code and synthetic binary streams. External media, scale-up, and independent scientific qualification remain open. [Preregistration](PHASE18_MULTIMODAL_BYTE_SKILLS_PREREG.md) · [Unrounded results and custody](../evidence/phase18-multimodal-byte-skills-dev-001.json)

## Phase 19 disk-backed byte Engram

`SSDEngramStore` uses the standard library and SQLite. The builder scans raw bytes in bounded chunks and spills suffix/next-byte counts to a temporary local database. It publishes a compact read-only file with exact binary keys, total counts, and sparse byte counts. Directory source hashes bind sorted relative paths, file lengths, and contents. Each file has its own history. The destination is reserved by a partial file, fsynced, and published without overwriting a previous index. The same-directory rename fallback supports filesystems that disallow hardlinks. Lookup uses a 2 MiB SQLite page-cache target with memory mapping disabled. Opening hashes the index in a streaming pass. `--engram-index-sha256` compares that hash with a pinned digest.

For each generated byte, the Engram sees only the current prompt and the bytes already generated in that session. The longest exact suffix meeting the support floor supplies normalized counts. For match length `L`, index maximum order `M`, and support `n`, the gate is `alpha × (L/M)^4 × n/(n+4)`. A match blends those counts with the existing streaming forecast before repetition penalty or sampling. A miss returns the original tensor exactly. The index and its counts stay on CPU. Lookup and blending require no training or donor execution. The generation adapter shares the same probability blend with the teacher-forced neural comparison.

Both scientific DEV criteria passed. The baseline below is each seed's frozen mixed neural snapshot. Cross-domain rows use a separate domain-focused snapshot and the opposite domain's index. Negative deltas improve BPB.

| Seed | Neural pooled BPB | Engram pooled BPB | Code index on prose Δ BPB | Prose index on code Δ BPB |
| --- | ---: | ---: | ---: | ---: |
| 3301 | 4.8180 | 4.7551 | −0.00610 | −0.01234 |
| 3302 | 5.1981 | 5.1412 | −0.00828 | −0.00625 |
| 3303 | 4.8126 | 4.7262 | −0.01476 | −0.00641 |
| 3304 | 5.0388 | 4.9634 | −0.00628 | −0.00576 |
| 3305 | 4.9840 | 4.9112 | −0.00309 | −0.01290 |

Mean math BPB was 5.29218 without the Engram and 5.29204 with it. Mean code BPB fell from 4.73973 to 4.56074, and prose fell from 4.87893 to 4.84553. The cross-domain hit rates ranged from 7.2% to 15.6%. The interference criterion measures BPB under those matches and leaves broader collision safety open.

Active Engram prefix and cold-reopen differences were exactly `0.0`, with 100/100 active prefix checks and zero donor forwards. The isolated lookup fixture stored 589,823 suffixes in 12,152,832 bytes. Across two fresh CPU processes, the slower mean lookup was 0.0308 ms per byte and the larger peak RSS was 14.55 MiB. Opening and hashing took about 9.5 ms. These figures describe the fixture hardware and index size. Live CUDA generation, multi-GB builds, and independent scientific qualification remain open. [Preregistration](PHASE19_SSD_ENGRAM_MEMORY_PREREG.md) · [Unrounded evidence](../evidence/phase19-ssd-engram-dev-001.json)
