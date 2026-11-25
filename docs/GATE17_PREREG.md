# Gate 17 preregistration and scaling CLI

Use the frozen Gate 15 SHA-bound raw evidence to select the neural backbone. Select the hybrid only on Gate 15 PASS. Otherwise use causal attention plus DirectMonarch. Fit the selected raw-byte backbone with next-byte cross-entropy. The training and fresh-process domain evaluation use the requested PyTorch device and precision. The qualification path transfers those exact neural weights into the existing CPU FP32 `NexusExpanseUnified` runtime for prefix and streaming checks.

## Data and checkpoint custody

Provide separate raw files for math, code and prose in train and evaluation. Bind every source to exact byte count and SHA-256 and reject full-file equality across splits. Supply disjoint evaluation data for a new corpus. For each domain, use contiguous input/target windows and a seeded bijection over window indices before a repeat, with no cross-domain boundaries. Mask the partial tail on both training and evaluation. Give each domain the same number of windows per optimizer step and score the first configured number of evaluation blocks per domain.

Use AdamW .003/.01 and gradient norm cap 1.0. Use FP32 weights and optimizer. BF16 runs use PyTorch autocast. CUDA BF16 requires native support. Select CUDA with `auto` when available. Record the resolved platform, device, dtype, framework versions, geometry, seed, Gate 15 evidence digest and clean source SHA/tree. Preserve CPU RNG and the selected CUDA device RNG alongside the model and optimizer. Validate all prior epoch hashes before resuming. Each epoch is step-bounded and committed immutably. An ambiguous consumed epoch blocks replay. Work directories must support exclusive file creation, locking and hard links, as on local NTFS or a typical Linux filesystem.

Record valid target bytes, CPU and wall time, training bytes/sec, peak RSS, and CUDA allocated/reserved peaks when applicable. A fresh process restores the final checkpoint and reports domain BPB and throughput. For the integrated runtime, fit compact bigrams and bounded episodic memory on the first 512 training bytes per domain. Verify exact prefix logits and hidden states, stream logits, and integrated forecast probabilities on the first up-to-32 prose evaluation bytes. The integrated runtime evaluation uses FP32 CPU and the native cache.

## Linux CPU smoke

Freeze this document, code and tests before starting. Use seed 42, dim 32, 32-byte context, two epochs, two steps per epoch, one sequence per domain per step and two evaluation windows per domain. Use fixed Phase 1 Git blobs for code and prose and deterministic arithmetic spans for math. In two separate commands, run:

```text
python benchmarks/gate17_gpu_scaling_harness.py --smoke --device cpu --dtype fp32 --directory ../gate17-smoke --epoch-limit 1
```

Repeat with `--until-complete` to resume the first checkpoint in a fresh one-epoch child and exercise the foreground operator route. Then use the original command arguments plus `--smoke-evidence ../gate17-smoke/result.json`. Qualification requires two distinct training processes, a validated checkpoint chain with an exact model digest and optimizer step counter at the second epoch start, a third process for exact restore and per-domain evaluation, and all four unified parity differences of exactly `0.0`. Commit the raw output as `evidence/gate17-harness-smoke-dev-001.json`. This smoke establishes Linux CPU engineering execution for the stated profile. Windows, consumer CUDA, BF16 throughput and independent learning advantage are separate qualification targets.

## Windows consumer-GPU example

From the checkout root with a CUDA-enabled PyTorch environment, run this command in PowerShell. The foreground invocation runs one checkpointed epoch at a time. Rerun the same command with the same source, inputs and parameters to resume a cleanly stopped run.

```powershell
python benchmarks/gate17_gpu_scaling_harness.py --device cuda --dtype bf16 --directory ..\nexus-gpu-run --train-math .\data\math-train.bin --train-code .\data\code-train.bin --train-prose .\data\prose-train.bin --eval-math .\data\math-eval.bin --eval-code .\data\code-eval.bin --eval-prose .\data\prose-eval.bin --dim 64 --context 256 --batch-per-domain 4 --steps-per-epoch 64 --epochs 64 --epoch-limit 1 --until-complete
```

This example covers one traversal of up to about 4 MiB per training domain. Set epochs from the actual byte counts for larger corpora. Use `--dtype fp32` for devices without native BF16. The run directory receives the immutable epoch chain, final checkpoint and fresh-evaluation receipt.
