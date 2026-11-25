# Gate 11 — selective-gated Kuramoto and InfoNCE-JEPA (frozen DEV protocol)

Parent a36a49d54f0b674718b7c050093a7769ab55e6c8; tree f6478b7f4d4e275f549b5c41601f0cb5caf3b445. Preserve Gate 10 source, manifest and negative evidence unchanged. No protected holdout, external dataset, paid service, default model change, promotion, or deployment. Gate 10's authored evaluation is previously exposed DEV, not an independent scientific trial. Failure mechanism hypothesis: fixed dt and no input-selective phase write gate cause early operands/premises to wash out. Alternative explanations include insufficient training coverage, optimizer and readout limitations.

## Architecture and engineering first

SelectiveKuramotoLayer adds one sigmoid per oscillator, initialized to 0.5 (zero projection weights/bias). Same shared weights as ungated initialization, dim/2 FP64 2D phase pairs, dt=.5. Project forcing plus coupling onto the phase tangent, multiply by dt*gate, add and normalize with +1e-12. Zero gate is approximate locking due to normalization epsilon. No KV cache, recurrent state 2*dim*8 bytes for two layers at batch 1, constant with span length; this is O(1) in sequence length, not in model dimension. Same always-on Monarch FFNs, no tokenization. Train with inherited batched FFN path; eval uses identical per-byte readout geometry to streaming. Check exact prefix logits and hidden at cuts 1/7/13/25/31, streaming probabilities at every byte and final streaming hidden, 32 random bytes, seeds 0..49 at each dim=64,128, CPU single thread, deterministic algorithms. Check fixed state bytes and phase norm error <1e-10. Required differences 0.0; no objective training on parity failure. This is finite tested parity, not a proof for all platforms, training/dropout modes or batch sizes.

## Stage A: supervised capacity before objective training

Same Gate 10 math and signed-logic train/eval splits. Three arms: original ungated Kuramoto, selective Kuramoto, original NexusExpanseByteCore causal attention (4 heads, Monarch FFN, no skips). L=2, dim=64, FFN multiplier 2, dropout zero, independent last-valid hidden linear 4-class and 2-class task heads; identical head initialization (seed 1011); paired encoder seeds 42,43,44. No future or label input to encoder. 15 epochs, 20 batches per epoch, six math and six science per batch, deterministic independent domain shuffles seed+epoch; equal mean of two supervised cross-entropies. AdamW .003/.01, clip norm 1.0; 300 steps per arm. Evaluate train and exact original 40 math / 40 science eval rows after training. Report predictions, labels, per-arm and per-seed accuracy and elapsed optimizer time.

Predeclared escalation: if any capacity arm in any seed scores <95% on either *training* task, run all three arms/all seeds on fixed 10x math/science training coverage (1200 each; 2400 total, original train plus prefixes 2/3/4, 90 synthetic groups each with four balanced counterfactuals). Original eval unchanged, check train/eval contexts and futures disjoint; no expansion based on eval scores. 15 epochs, 200 batches per epoch, 3000 steps, same optimizer/architecture. If triggered the expanded regime is decisive for all capacity criteria, otherwise original regime is decisive. Both results retained; no mixing regimes by winner or seed. If expansion is incomplete Gate 11 cannot be classified. Attention control is diagnostic, not an additional PASS condition. Matched training steps/rows within each regime, not matched parameter counts or FLOPs.

## Stage B: objective comparison

Only after all required capacity receipts exist. Selective core next-byte CE vs selective InfoNCE-JEPA, same initial encoder per seed 42/43/44. Exact unchanged Gate 10 360 train / 120 eval manifest. 10 epochs, 30 batches of 12 (4 per domain), 300 steps, identical deterministic schedules; AdamW .003/.01, clip 1, CPU single thread. CE covers adjacent context+future bytes with padding mask. InfoNCE predicts last-valid EMA future latent from context only, DirectMonarch predictor 64-128-64, row-normalized cosine scores / temperature .1, diagonal positive and other 11 in-batch futures negative (including other domains); no VICReg. No target gradients, EMA cosine .90 to .99, target eval geometry. Different objectives can have different FLOPs and access different target representations; record optimizer time. Primary retrieval identical Gate 10 train-only ridge (lambda 1 including intercept) and same-domain stable 16-candidate sets on all 120 eval queries; report Top1 and MRR and domain ranks. Recompute histogram control (Top1 .20); direct and native predictor retrieval diagnostics not substitutes. Math mod-4 and causal chain use Gate 10 train-only ridge on frozen context features; report noncollapse descriptively. Recheck trained streaming latent parity on three fixed eval spans. No objective choice after results.

Speed: per seed compare trained InfoNCE encoder at dim64 and initialized same architecture at dim128 against actual repository CausalKVCache (two layers, four heads, dim matched, max_seq_len=256, all FFN gates open, no dropout, no online memory/fast weights). Fixed 256 raw random bytes from seed, no window rollover, both output next-byte probabilities; verify cache logits match batch with atol 2e-5/rtol 1e-5. One warmup then five alternating-order timings, median ms/byte, reset/setup outside timing. Record state tensor and cache tensor bytes; speed claims limited to this CPU implementation/geometry, not optimized SDPA/FlashAttention or long-window throughput. Dim128 is structural timing, not trained-model accuracy.

## Predeclared decision

All comparisons per seed, not only means. PASS iff parity passes, selective supervised beats ungated strictly on both heldout tasks in every seed and achieves math >=45% and science >=70% in every seed of the decisive regime, InfoNCE retrieval Top1 strictly exceeds CE and .20 histogram in all three seeds, and speedup >=2.0x at both dims in every seed. MIXED iff parity, strict supervised improvements and strict histogram improvements all hold but any absolute accuracy, CE or speed bar misses. FAIL if exactness, strict supervised improvement on either task, or histogram superiority fails. Infrastructure error/missing receipt is UNVERIFIED, not a scientific FAIL; a completed nonzero parity result is an engineering failure and blocks training. No scientific generalization, retention or autonomy claims from this reused DEV set.

## Execution and custody

Linux CPU, Python/PyTorch/NumPy, no scheduled work; Gate 10 helper imports resource (native Windows not qualified). Clean frozen commit, external work directory on filesystem supporting exclusive hard links. Record exact source SHA/tree, manifest hash, versions/platform, one thread. Use a single writer. Per epoch exclusive immutable intent before training, checkpoint and SHA256 parent-chain metadata after; restore optimizer and RNG only from complete verified epochs, do not replay ambiguous consumed epochs. Final state checkpoint and fresh-process cold state hash, receipts exclusive; merge requires all seed/phase/arm receipts, exact identity, expected steps, checkpoint hashes and matched objective initialization. Final receipts are not independent review. Checkpoints stay outside source; preserve them in authorized custody, publish merged evidence only after completion. No trial resets or post-result tuning.

Run full pytest and prefix reproducer before training; then parity:

    python -m pytest -q
    python benchmarks/prefix_causality_dev.py
    git fsck --full --strict
    git diff --check
    python benchmarks/gate11_selective_kuramoto_dev.py --directory /outside/gate11 --parity

For each seed 42,43,44 run:

    python benchmarks/gate11_selective_kuramoto_dev.py --directory /outside/gate11 --seed 42 --phase capacity

Use --epoch-limit 2 for bounded resumable slices if needed. After all capacity receipts, if train-only trigger holds run --phase expanded for every seed; otherwise skip expansion. Then --phase objective for every seed. Each command processes phase arms in fresh processes; --arm permits a single arm. Merge from frozen clean source:

    python benchmarks/gate11_selective_kuramoto_dev.py --directory /outside/gate11 --merge /outside/gate11/evidence.json

Repeat full pytest, 100-seed prefix reproducer, git fsck and git diff --check on exact final candidate; publish actual outputs and merged evidence in a separate evidence-only commit. Keep main unchanged.
