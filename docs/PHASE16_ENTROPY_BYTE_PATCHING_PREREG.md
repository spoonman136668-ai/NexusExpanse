# Phase 16 causal entropy-guided byte patching DEV preregistration

Freeze this protocol, the implementation, the benchmark and the tests in the first Phase 16 commit after `d1936aaa760c644915653ddc1dcefedf0ce32aa6`. Score once from the clean commit into a fresh external path. Record the source, tree and protocol digest before scoring. Preserve interrupted or invalid attempts. This comparison uses Linux CPU FP32, one PyTorch thread, local data and seeds 3001–3005.

## Causal contract

Fit an add-one byte bigram table from the training spans and freeze it. An unseen preceding byte has a uniform next-byte row. At index `t`, surprise is the negative base-2 log of the probability of `x[t]` under the row for `x[t-1]`. The first byte always starts a patch. Each subsequent byte starts a patch when surprise reaches the threshold or the preceding patch has reached its size cap. A high-surprise byte is the first byte of its new patch.

The local encoder uses the current byte and the two preceding bytes, with zero padding at the beginning. Pool only byte embeddings inside a completed patch. A causal Transformer computes patch latent `z[j]` from completed patches through `j`. When scoring the target after byte `t`, the decoder sees the local state at `t` and the latent of the patch immediately before the current one, with zero context in the first patch. A patch whose last byte is `t` can first influence the logit at `t+1`. A surprise boundary discovered at `t` can reveal the preceding patch ending at `t-1`.

For a fixed prefix through `t`, all boundary decisions through `t` depend on the fixed bigram table and that prefix. The visible completed patch byte sets, ordered latent attention inputs, and local encoder inputs through `t` are therefore fixed. Exact byte and patch block dimensions are determined by block index. Causal masks exclude future patch slots inside each block, so future bytes and later patch counts cannot change the active prefix reduction geometry. Test both shortening the sequence and changing every suffix byte in 100 independent seeded cases. Compare boundaries, known completed ends, completed and visible latents, hidden states and logits. Require exact `0.0` for float differences and zero boundary disagreements. Stream decoding caches completed blocks, recomputes the current patch block when a patch closes, and uses the same fixed kernels. Require exact `0.0` for streaming logits and hidden states. There is no FP32 tolerance.

## Data and controls

Reuse the Phase 13 corpus builder for each seed, using the Python and Markdown sources frozen at `105e6e754abe940de5c66ca41a75c3bf825aa1cb` and generated arithmetic. Each domain supplies 96 FIT and 24 CHECK 64-byte windows. Code and prose use disjoint FIT/CHECK source files. Math uses disjoint operand ranges. Verify no shared exact 32-byte windows across all FIT and CHECK files. Keep the reserved VAL selections unused. The patcher fits only the three FIT files and uses threshold 5.0 bits and cap eight. The public default threshold is 3.5 bits.

Both cores train from scratch at width 32, two deep layers, four heads, dense FFN multiplier 2, untied byte readout and dropout zero. Share the initial parameters for every corresponding weight and bias. Use the flat `NexusExpanseByteCore` with causal SDPA and its FFN skip disabled. Use one context-512 block per update, cycle math, code, and prose, and choose the block with Gate 17's seeded bijection. Both arms receive the same 48 AdamW updates, valid target bytes and gradient clipping. Use rate .003, weight decay .01 and gradient norm cap 1. Time the entire update loop after warming one forward call, and alternate arm order by seed. Report all valid CHECK targets per domain as BPB and pool by valid target-byte count.

Warm both arms at each context and domain. At 512 and 1,024 bytes, time three forward calls on the first CHECK span per domain in alternating arm order. Report raw wall times, bytes per second, and the speedup using the sum of per-domain median times. Flat uses the actual timed context length. Count every opened patch for the primary compression ratio, including an open tail. Also record completed-patch steps and executed padded attention steps. The exact path pads the final active patch block to 16 slots, and byte decoding uses 32-row blocks. These overheads are included in timings. Compare streaming outputs at context 512 on all three domains for every seed. In a fresh process, restore only the saved tensor state into the same configuration and require identical state and output digests.

## Frozen verdicts

- **ENGINEERING PASS** requires 100/100 exact Phase 16 prefix cases, exact stream and cold restore on all three domains for 5/5 trained seeds, pooled multi-domain compression of at least 2.0× for each seed at both contexts, and zero donor forwards. It also requires bare pytest with zero failures, skips or warnings, the existing prefix benchmark at 0/100 failures, and clean Git integrity and whitespace checks. A miss makes the engineering verdict **FAIL**.
- **SCIENTIFIC PASS (DEV)** requires both quality and speed bars. Quality requires mean pooled patched BPB no more than 0.05 above flat and the same margin in at least 4/5 seeds. Speed requires median forward speedup of at least 1.25× at both contexts and median training throughput at least equal to flat. **MIXED (DEV)** means one bar passed. **FAIL (DEV)** means neither passed. A failed scored engineering check makes the scientific result invalid. Keep every seed and domain, including regressions.
- This comparison concerns the frozen small CPU workload. Independent scientific qualification and GPU throughput require separate evidence.

```bash
/workspace/.venv/bin/python benchmarks/phase16_entropy_patching.py --out /workspace/work/nexus-phase16-scored.json
/workspace/.venv/bin/pytest -q
/workspace/.venv/bin/python benchmarks/prefix_causality_dev.py
git fsck --full --strict
git diff --check origin/main..HEAD
```
