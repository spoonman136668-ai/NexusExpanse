# Qualification roadmap

## Historical Gate 1 (unverified)

Imported report: neural only 3.767148 BPB; static memory 2.307542; online episodic + Hebbian 1.732717; dense reference 3.3311. Reported 340000 FFN bytes and 55.301% mean skip. These are not whole-system efficiency metrics. 150681 -> 176582 distinct suffix contexts; no byte accounting. The dense score is hard-coded, no checkpoint or corpus manifest is supplied, and the verdict does not evaluate the stated forgetting criterion. The four original tests passed on import. Disjoint-domain exact substring recall does not demonstrate durable learning or generalization. Counts are never overwritten but conflicting observations can change predictions. The fast weight scratchpad resets at the start of each evaluation.

## Next scientific gate: proposed, not executed or frozen

1. Recover the exact original corpus Git SHA/tree, file manifest, split hashes, training configuration, checkpoint custody, and dense benchmark provenance. If unavailable, label Gate 1 non-reproducible and use a new prospectively frozen split; never present a new split as a reproduction.
2. Freeze the dataset, split by source/domain to avoid near-duplicate overlap, seeds, budget, matched training and evaluation protocol, and decision thresholds before any held-out use. Include an explicit duplicate audit. Obtain separate first-use authorization.
3. Evaluate neural dense and Monarch controls (no skip / skip), memory-only static and adaptive n-gram/PPM, neural + static episodic, neural + online episodic without Hebbian, neural + Hebbian without episodic, neural + both. Dense controls also need equal access to memory and online adaptation. Clone identical trained state for each applicable arm. Prediction must precede update. Report prequential and frozen evaluation separately.
4. Measure BPB, complete model parameter/buffer bytes, memory allocated and serialized bytes, peak process/GPU memory, wall time, training and evaluation throughput, memory growth, and cold restart. No speed inference from skip fraction. Use matched time and compute comparisons where feasible.
5. Check conflicting-context interference, held-out domain transfer, retention after reset/reload, and explicit memory budgets. Compare against strongest matched control with predeclared seed-level uncertainty. Provisional Hebbian threshold: at least 0.05 BPB better than online episodic-only on each of three seeds and no >0.02 BPB degradation on familiar cold retention; thresholds need freezing before execution. No claim of intelligence or persistent cognition from BPB.

## Open implementation work

- Implement trained dense controls and isolated ablation runner; calibrate hyperparameters on train/development only.
- Implement bounded and versioned persistent memory serialization; current episodic memory has no memory budget or checkpoint persistence through module state_dict.
- Benchmark actual sparse compute and full-system cost, including Python memory overhead.
- Independent qualification, Windows consumer GPU acceptance and any deployment remain separate.

## Development checkpoint (2026-11-24)

- Matched dense/Monarch gated/ungated ablations implemented; see `evidence/ablation-dev-001.json` and notes. The memory-only online arm beats the combined model on the known tiny fixture; no positive Hebbian increment over online memory met the provisional threshold. Preserve this negative result. Stronger real-data training, independent corpus/overlap audit and a prospectively frozen evaluation remain outstanding.
- Opt-in bounded episodic memory now supports deterministic context eviction, versioned JSON serialization, external restore limits and exact cold-process prediction replay. Context count and serialized bytes are not live memory bytes. Interference and eviction remain known loss mechanisms. No trained model checkpoint or persistent fast-weight lifecycle is established.
- Next: develop a neural/memory arbitration comparator on development data (including memory-only, fixed blend, and a causal reliability estimator), then freeze a new real-data protocol with strongest trained controls. Do not consume a held-out trial without authority. Independently qualify retention under a fixed memory budget, full-system compute and consumer GPU behavior.
