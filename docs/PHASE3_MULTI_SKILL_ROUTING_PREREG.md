# Phase 3A DEV preregistration

**Hypothesis.** Discounted evidence from observed raw bytes can select among four frozen skills, preserve performance on the Python base domain, and recover after unlabeled regime switches. Sparse routing may retain a predictive advantage while reducing receiver calls. The two hypotheses can have different outcomes.

**Frozen source and environment.** Use the Phase 2D base `105e6e754abe940de5c66ca41a75c3bf825aa1cb` and a clean checkout of the commit that adds this protocol, router, tests, and harness. Run seeds 1501–1505 once for the scored experiment on Linux CPU using `/workspace/.venv/bin/python`, PyTorch 2.15.0+cpu and NumPy 2.3.5. Use FP32 inference with one Torch thread and deterministic algorithms. The experiment uses local Git objects and creates all donor packs locally. GPU and independent acceptance are separate.

## Fit and held-out streams

- Independently initialize four 256-byte Dense donors per seed in a producer process. Use the existing 32-wide, two-layer checkpoint transfer geometry and optimizer. Fit each donor for 192 AdamW steps, batch size eight, on 96 64-byte windows. Each fit step samples windows with replacement from that domain. Use random byte-to-row bijections; export only tensors and the map. Exit the producer before creating receivers. Import all four static packs with `load_skill(method="residual8")` inside the donor-forward guard, and freeze all receivers.
- Python FIT comes from `core.py` and `unified.py`. Python CHECK comes from `verifier_agent.py` and `deltanet.py`. Markdown FIT comes from `README.md` and `ARCHITECTURE_V1.md`. Markdown CHECK comes from `GATE25_PREREG.md` and `CHECKPOINT_TRANSFER_PROTOCOL.md`. Read exact bytes from the frozen base tree. Shuffle aligned 64-byte windows with seed + 30000.
- Rule A and Rule B share 32 two-byte cues and the delimiter byte 125. The same cue maps to different answer bytes under the two frozen seed-specific permutation tables. A record is cue, delimiter, answer. Draw 16 distinct keys per window from disjoint FIT and CHECK generator draws using the same fixture RNG. No rule or domain labels are exposed to a router. Reject exact 32-byte overlap when selecting windows and check the concatenated switch streams against FIT again to catch boundary overlap.
- Single-domain CHECK has eight independent 64-byte streams per domain. Switch CHECK has eight independent 64-byte streams per transition, with 32 bytes before and after the boundary. Transitions are Rule A to Rule B, Python to Markdown, and Markdown to Rule A. Reserve separate held-out windows for every domain role. Persist digests of FIT and CHECK bytes. CHECK is used once for scoring, with no hyperparameter fitting or selection.

## Forecasts and matched controls

Every row at position *t* forecasts byte *t* from the prefix up to and excluding *t*. Byte 0 receives a uniform 256-way forecast and no receiver call. Every later receiver forecast uses up to 64 preceding bytes. All arms have the same frozen receivers, prefix budget, evaluation bytes, and unconditional starting distribution.

| Arm | Rule |
| --- | --- |
| `oracle_manual` | Evaluate only the receiver indicated by the ground-truth domain of the target byte. This is the privileged routing reference. |
| `bayes_bma` | Accumulate each skill's preceding log likelihood as `score = 0.85 * score + log p(previous byte)`. Normalize the scores with softmax and average probabilities. |
| `bayes_argmax` | Use the same scores. Choose the highest score, with stable base-first order for ties. |
| `surprise_sparse` | Hold the active skill between probes. Probe at the next forecast if the last byte's surprise exceeds 3.5 bits or the position is divisible by 16. Discount scores each byte. At a probe, compare the preceding observed byte across experts using the penultimate output in the same bounded prefix pass, then select the highest score for the next forecast. The first unconditional likelihood is equal for all experts. Count every stream/expert invocation, with no dense precomputation or off-path replay. |
| `uniform_mixture` | Average the same four expert probabilities with constant 1/4 weights. |
| `base_only` | Evaluate only the frozen Python receiver. |

On single-domain streams, report micro-averaged top-1 accuracy at rule answer positions, Python and Markdown bits per byte (BPB), and BPB across every scored byte in all four domains. Routing accuracy is the top-weight skill or hard selection across all four domains after dropping the first eight positions. Uniform-mixture routing ties use the base index as a reporting convention. On each switch transition, report post-switch accuracy and BPB plus the first byte offset of an eight-byte window with at least 80% routing accuracy across the eight matched streams. An unrecovered transition has null latency. The per-seed gate uses the worst latency across all three transitions.

Report actual stream/skill calls divided by all forecast positions, including the zero-call prior, and wall milliseconds per byte for each arm and suite. Measure donor training and receiver import separately. Save the frozen bank after scoring and restore it in a fresh CPU process; compare all state and configuration digests, all three routers' forecasts and routing digests on both suites, and the sparse invocation rate. Stop on a source, corpus, donor-forward, or serialization contract failure. Preserve failed or null outcomes. Altering the protocol after a scored run requires a separate preregistration and evidence record.

## Frozen verdict rules

- **ENGINEERING PASS** requires at least 191 pytest tests passing with zero failures or skips, 0/100 failures in `prefix_causality_dev.py`, exact 0.0 prefix causality on 50 seeds for all three routers, zero donor forwards and exact fresh-process restore on every scored seed.
- **SCIENTIFIC PASS** requires one dense Bayes arm to strictly beat both controls on single-domain rule accuracy and overall BPB in all five seeds, achieve at least 90% post-warmup single-domain routing accuracy in all five, and recover within eight bytes for all transitions in at least four seeds. It also requires sparse routing to use at most 2.4 skill calls per byte on both suites and strictly beat both controls on the same single-domain accuracy and BPB metrics in all five seeds.
- **SCIENTIFIC MIXED** applies when one dense Bayes arm strictly beats both controls on both predictive metrics in at least four seeds and a PASS requirement is unmet.
- **SCIENTIFIC FAIL** applies when neither dense Bayes arm meets that four-seed control comparison.

Run the scored harness once from the preregistration commit and record the unrounded per-seed, per-arm results in `evidence/phase3-multi-skill-routing-dev-001.json` in the following commit. Qualify the final integration commit separately.
