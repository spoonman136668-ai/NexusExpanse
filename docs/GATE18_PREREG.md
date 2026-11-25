# Gate 18 preregistration

Use the clean Phase 2 baseline `38cd40a27d7e3b1ac48d5d69766944de02e4a7fb`. Gate 15 showed a seed-dependent math failure. Readout normalization and warmup are a joint hypothesis. This experiment cannot isolate their individual effects or establish the cause of the earlier failure.

The opt-in v2 uses LayerNorm on the flattened DeltaNet matrix readout before the existing output projection, then multiplies the projected residual by a trainable per-channel vector initialized to .5. Keep the original hybrid constructor behavior by default. Use the existing exact causal stream and fixed state. Before training, require all five prefix and streaming differences to be exactly zero, across 50 seeds at each dimension 64 and 128.

Use the exact Gate 15 expanded split and matched 15 × 200 optimizer steps, AdamW .003/.01, gradient clip 1.0, common task head initialization and seeds 42–46. The candidate starts at base LR divided by 300, reaches .003 at step 300, then decays by cosine to zero at step 3000. Reuse verified Gate 12 pure-attention receipts for seeds 42–44. Fit that exact constant-rate control protocol for seeds 45 and 46.

Measure live recurrent/KV state tensors and median ms per byte with five timings after warmup in fresh processes. Use seeded 256-byte inputs, trained width 64 and initialized width 128, no rollover, and the same FP64 softmax and pure-attention KV implementation as Gate 15. Record RSS and exact trained candidate stream parity. Rotate arm order by seed.

PASS requires exact parity, both held-out scores at least .95 in every seed, 5.0x speed and 40% state reduction in every timing cell. An engineering mismatch or two or more failed capacity seeds means FAIL. Otherwise, four qualifying seeds or a missed speed/cache bar means MIXED. Only PASS unlocks the opted-in Gate 17 CLI route.

Freeze code, unit tests and this document before execution. Use `--parity` then separate `--epoch-limit 3` slices for each arm and seed with immutable resumed epochs. Merge in the frozen checkout using `--control-directory` to the Gate 12 checkpoint store. Commit the raw output to `evidence/gate18-hybrid-v2-dev-001.json` after execution.
