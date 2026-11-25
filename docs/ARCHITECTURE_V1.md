# Nexus Expanse v1.0-rc1 architecture and DEV scorecard

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
