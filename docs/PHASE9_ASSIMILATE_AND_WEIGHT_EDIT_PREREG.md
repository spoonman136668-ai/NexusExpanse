# Phase 9 offline donor assimilation and surgical editing preregistration

The frozen base is `deb473f70bdbdfc6d25192a3140c06d75d5a6d6e`. Run one scored DEV confirmation on the clean first Phase 9 commit using Linux CPU and `/workspace/.venv/bin/python`. Use seeds 2201–2205. Preserve the unrounded output and publish it in the second Phase 9 commit. The qualification command is bare `/workspace/.venv/bin/pytest -q` with PYTHONPATH unset. Keep protected main at `bb21fbe53c0299d432e866eb4d016c25b81ee6eb`.

## Input and fixed budgets

Use authentic repository Python and Markdown bytes from commit `105e6e754abe940de5c66ca41a75c3bf825aa1cb`. Each seed uses 96 FIT, 24 VAL and 24 CHECK windows of 64 bytes per domain. Use the Phase 3A FIT and CHECK paths for base Python and Markdown and the Phase 7 VAL paths for those domains. Reject any shared 32-byte substring between selected windows, including across domains. Record file paths and per-seed split digests. CHECK enters the calculation only after both edit arms select their candidate from VAL and both student configurations are frozen.

Train the 32-wide, two-layer, four-head dense donor for 256 AdamW steps on the same combined FIT corpus. Batch size is 8. Import its static checkpoint using the existing residual8 conversion to make the clean native skill for editing. The donor then runs only in the local exporter. Export 16 FIT windows per shard at top-k 32. For each observed nonempty byte prefix, encode only that prefix, project next-token mass onto token-first bytes, then renormalize the top 32 byte probabilities. This is the same boundary projection used by the existing compact offline KL cache. Each native arm starts from the same seed and geometry and gets the same raw FIT batches, AdamW steps, clipping and training budget. Both train for 192 steps with batch size 8, learning rate 0.004 and weight decay 0.01. Use the mixed CE and static KL objective with weight 0.9 for assimilation and ground-truth CE with weight 0 for scratch. Report held-out next-byte BPB across both domains and by domain. The guarded student path sees only static tensors and has zero donor forwards.

The exporter accepts a local DenseTokenDonorLM weights-only package with either a singleton byte-to-row bijection or canonical BPE spellings and merge ranks. The HF path requires local tokenizer.json and local safetensors. It excludes special-token mass and rejects prefixes whose tokenizer transformations cannot reproduce their bytes, allowing the synthetic initial SentencePiece space. The helper is checked with local decoder fixtures and a loader double. Running a real HF checkpoint and CUDA qualification require separate locally provisioned environments.

## Fault and controls

Python is the fault domain and Markdown is the retention domain. Use SVD on the paired differences of sequence-mean attention contexts before the output projection. Set the intervention direction to the layer 1 FIT contrast with its sign chosen to give a nonnegative dot product with the Python FIT mean. Inject a rank-one addition of scale 6.0 into layers.1.attn.out_proj using the normalized space-byte readout direction centered across the 256 byte rows. This simulates a local bias toward repeated spaces. The repair receives VAL bytes and the faulted skill. It uses neither the FIT direction nor the clean weights. The known intervention and its aligned FIT domain contrast make this a controlled localization test.

Compare SVD localization with one seeded isotropic random unit vector. Both arms search only alpha 0.25, 0.50, 0.75 and 1.00 at this projection. Validate every candidate for strictly lower Python VAL BPB, absolute Markdown VAL drift at most 0.02 BPB relative to the faulted skill, zero prefix difference over 16 probes and exact cold tensor and forecast parity. Select the accepted candidate with lowest Python VAL BPB and resolve ties by earliest alpha. Score the unchanged faulted arm when no candidate qualifies. Report the clean reference and the faulted, SVD and random forecasts on CHECK. Recovery is the corrected Python BPB improvement divided by positive clean-to-faulted Python BPB damage. Nonpositive damage cannot satisfy the recovery gate. Held Markdown drift is absolute change from the faulted arm. Also report signed differences from the clean Markdown BPB so injection damage stays visible. Check zero calls to Tensor.backward, torch.autograd.backward and torch.autograd.grad during the guarded inference-only edit experiment.

## Frozen verdicts

**ENGINEERING PASS** requires at least 586 bare pytest passes, zero failures, skips or warnings, 0/100 failures on the legacy prefix suite, exact 0.0 prefix difference across 50 seeded edited and assimilated skill checks, zero guarded donor calls and edit autograd calls, and 5/5 seeds with exact cold restores.

**SCIENTIFIC PASS** requires SVD edits accepted in 5/5 seeds, SVD strictly better than matched random and faulted on held Python BPB in 5/5, at least half the injected Python damage recovered in 5/5, held Markdown drift at most 0.02 BPB in 5/5, and assimilation strictly better than CE-only scratch on held BPB with zero donor calls in 5/5.

**SCIENTIFIC MIXED** requires SVD strictly better than matched random and faulted in at least 4/5 seeds with retention within bounds in all five and some other PASS requirement unmet. **SCIENTIFIC FAIL** applies if SVD is better than both controls in fewer than 4/5 or if any seed violates the held retention bound. Ties count as no win. The verdict covers this Linux CPU DEV proof-of-concept. Independent scientific acceptance and deployment require separate authority and evidence.

## Reproduction

```bash
/workspace/.venv/bin/python benchmarks/phase9_weight_edit_and_assimilate.py --out /workspace/work/nexus-phase9-scored.json
env -u PYTHONPATH /workspace/.venv/bin/pytest -q
/workspace/.venv/bin/python benchmarks/prefix_causality_dev.py
git fsck --full --strict
git diff --check deb473f..HEAD
```
