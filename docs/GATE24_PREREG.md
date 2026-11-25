# Gate 24 RC1 CLI smoke protocol

Freeze the CLI, docs, runner and tests before launch. Use a clean committed checkout and fresh work directory. Verify a Linux CPU FP32 developer workflow on the six small authored raw byte files in the harness. Windows CUDA and long-run throughput remain open.

Run train in two independent one-epoch invocations with width 32, context 16, one step per epoch, one block per domain. The first produces only the first checkpoint. The second resumes with the same model digest and optimizer step counter. Require the exact fresh-evaluation receipt and export a checked CPU unified snapshot using the fitted Gate 17 compact and episodic priors.

Run verify on that snapshot in a fresh subprocess. Require 0.0 for prefix logits, prefix hidden states, streaming logits, unified streaming probabilities and round-trip continuation probabilities. Require exact adaptive and neural state. Run stream from the verified snapshot in a fresh subprocess, feeding two hex byte chunks via stdin. Require five next-byte candidates, a short bounded continuation, observed-only arbitration updates, and a checked saved snapshot. Run solve in a fresh subprocess with the bounded AST library, source `010203`, target `040302`, and depth two. Require exact zero residual in at most six calls. Report primitive calibration separately.

Run `python benchmarks/gate24_cli_smoke.py --directory OUTSIDE --output OUTSIDE/result.json`. Preserve command output, training and checkpoint custody, snapshot hashes and all smoke measures under `evidence/gate24-v1-rc1-cli-smoke.json` only after successful execution.
