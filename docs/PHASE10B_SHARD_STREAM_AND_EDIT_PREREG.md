# Phase 10B shard streaming and one-sided weight edit preregistration

Use Linux CPU, FP32, one Torch thread and seeds 2401–2405. Freeze code and this protocol on the clean first commit above Phase 9 `7b1f81a07ddbd0f37682a8132f483d29298a0408` before scoring. Preserve the unrounded receipt in the second commit. Keep protected main at `bb21fbe53c0299d432e866eb4d016c25b81ee6eb`. All fixtures are local. Use no external downloads or paid APIs.

Use Phase 9's authenticated Python and Markdown from `105e6e754abe940de5c66ca41a75c3bf825aa1cb`. Use those same path sets, 64-byte windows, 96 FIT, 24 VAL and 24 CHECK windows per domain, and the same split generator with the fresh seeds. Reject overlap of any 32-byte substring across retained windows. Record split and artifact hashes.

## Suite A

Reuse Phase 9's training, residual8 import, layer 1 attention output projection fault, FIT-derived injection direction, scale 6.0, and computational budgets. Compare the VAL SVD and seed-matched random directions with the faulted baseline. Both search alpha 0.25, 0.50, 0.75 and 1.00 on VAL. The verified edit requires lower Python BPB, exact causal and cold checks, and signed Markdown drift `edited - faulted <= 0.02` BPB. Negative drift is improvement. Select the accepted candidate with lowest Python VAL BPB, using earliest alpha for ties. Fall back to the faulted skill if every candidate is rejected.

On CHECK, report clean and faulted Python BPB, SVD and random Python BPB, signed Markdown drift for both arms, and accepted alpha. Recovery is the faulted-to-edited Python BPB gain as a percentage of positive clean-to-faulted damage. Nonpositive damage cannot clear the recovery gate. Record Tensor.backward, torch.autograd.backward and torch.autograd.grad attempts and exact cold checks for both selected arms or their fallback. Input fixtures, extraction budgets, and the fault specification match Phase 9. This experiment isolates the gate change and does not establish that an arbitrary fault can be localized.

## Suite B

The named Phase 10 techniques are implemented in this DEV branch from the operator's specification. The earlier winning artifact and its exact configuration were unavailable in the named base, project refs, and indexed context. The receipt qualifies this implementation.

Train a two-layer, four-head 64-wide donor for 320 AdamW steps with 8 FIT windows per batch, learning rate 0.004, weight decay 0.01, and gradient clipping at 1.0. Fit 64 BPE merges on FIT only for 320 rows, including the 256 singleton byte rows. The separate producer publishes three explicit mock tensor shards and tokenizer.json. The receiver supports standard local safetensors indexes using the optional local package and never imports donor contracts, configurations with remote code, or Transformers. The benchmark uses weights-only mock .pt shards. Tests also exercise the safetensors reader interface against a local double. A live safetensors model load remains an independent qualification.

Learn a 32-dimensional input spectral subspace from donor singleton embeddings and align it by orthogonal Procrustes to the seeded native embedding. Project attention and input/output tables into that basis. Mine MLP keys and values by the product of their projected norms, keeping the native FFN inner width. Fit each reduced MLP affine as a DirectMonarch block approximation with a rank-four spectral residual. Compute a static association operator by averaging each layer's OV and mined value/key products. Apply the projected token readout to normalized singleton byte embeddings enriched by that operator and group token mass by its first raw byte. The optional static Engram table contributes additive byte logits. No donor model is instantiated or executed in this conversion. Each one-shard visit reduces needed tensors and releases original tensors before opening another shard.

Compare the coherent stack, a shuffled control, and seeded native scratch. The shuffled control consumes the same saved donor tensors and tokenizer and independently permutes rows and columns of the reduced embedding, head, attentional and MLP projections. Scratch uses the same initialized native geometry and trainable rank-four spectral correction capacity with a zero residual. All use the same 64 native byte cross-entropy steps, batches of 8, AdamW learning rate 0.004, weight decay 0.01 and clipping at 1.0. The two donor-derived arms blend static and neural probabilities at alpha 0.25 for the first update and linearly decay to 0.0 on the 64th. Scratch starts and stays at alpha 0. All use identical FIT batch schedules for matched adaptation. Report combined and by-domain CHECK BPB at zero updates and after 64 updates, exact cold checks at both stages, times and shard custody. The measured BPB is for native raw-byte forecasts and does not require donor tokenization at evaluation.

## Frozen verdicts

**ENGINEERING PASS** requires at least 606 bare pytest passes with zero failures, skips and warnings, legacy prefix failures 0/100, sharded exact prefix checks 50/50, peak one loaded shard per import, zero guarded donor calls, zero autograd calls during weight editing and 5/5 exact cold restores in each suite.

**Suite A SCIENTIFIC PASS** requires SVD accepted in 5/5, SVD strictly better than both faulted and random on held Python BPB in 5/5, at least half positive damage recovered in 5/5, and signed held Markdown degradation at most 0.02 BPB in 5/5. **MIXED** requires acceptance and wins over both controls in at least 4/5 with retention within bounds in all five and a missed PASS criterion. **FAIL** applies otherwise.

**Suite B SCIENTIFIC PASS** requires a zero-update BPB advantage of at least 0.50 against shuffled in 5/5 and lower post-adaptation BPB than both shuffled and scratch in 5/5. **MIXED** requires one of those two complete criteria. **FAIL** applies if neither complete criterion clears. Ties do not count as wins for strict comparisons.

Independent acceptance, live safetensors qualification and deployment are separate from this DEV confirmation.

```bash
/workspace/.venv/bin/python benchmarks/phase10b_shard_stream_and_edit.py --out /workspace/work/nexus-phase10b-scored.json
env -u PYTHONPATH /workspace/.venv/bin/pytest -q
/workspace/.venv/bin/python benchmarks/prefix_causality_dev.py
git fsck --full --strict
git diff --check 7b1f81a..HEAD
```
