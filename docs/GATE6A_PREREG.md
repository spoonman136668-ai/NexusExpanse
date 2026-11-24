# Gate 6A — Holographic Vector-Symbolic Relational Memory (Frozen DEV Protocol)

Evaluate `HolographicRelationalMemory` (`nexus_expanse/holographic_memory.py`, fixed `D=4096` unitary HRR superposition vector, Delta-Rule `eta=0.35`, readout `temperature=12.0`, `resonance_threshold=0.22`) on the exact frozen Gate 4A/5A noisy and surface-shifted compositional benchmark (`seeds 42, 43, 44`, 10% observation noise, 192 unseen two-hop compositions per seed, `40,000`-byte serialized state ceiling, exact fresh-process cold restore).

No hardcoded byte-offset grammar parsers or punctuation lists are supplied to the candidate; framing/shifted punctuation is filtered natively via holographic resonance.

Compare against the exact Gate 4A/5A controls on identical exposure order and repaired `376353d` backbone: `frozen`, `sparse_support`, `lookup`, `local_chain` (Gate 3), `symbolic_chain`, and Gate 5A `LatentAssociation` (`0%` unseen accuracy).

Pre-declared disposition criteria:
- **PASS (Full Gate 4A Bar):** Unseen-composition BPB beats the strongest of all five controls by `>= 0.05` AND worst per-association drift `<= 0.02 BPB` on all 3 seeds AND serialized state `<= 40,000 B` AND exact cold-process restore.
- **MIXED (Tokenless Breakthrough vs. Raw Controls):** Unseen-composition first-target accuracy `>= 75%` AND unseen BPB beats raw tokenless `sparse_support` memory (`6.66–7.07 BPB`) by `>= 2.0 BPB` on all 3 seeds AND serialized state `<= 40,000 B` AND exact cold-process restore, while failing either the canonical-grammar oracle margin or the `<= 0.02 BPB` drift ceiling.
- **FAIL:** Otherwise.

No post-result threshold tuning or code modification between preregistration commit and evidence commit.
