# Phase 21 local five-category scoreboard preregistration

Freeze the implementation and this protocol at a clean direct child of `61d238f84334c9716c2d1c4a56388f83cf1f3e20`. Run seeds 3501–3505 in order with one PyTorch CPU thread and deterministic algorithms. Write an intent with the exact source commit, tree, and protocol digest before scoring. Keep all observed category scores, including zeros.

## What is measured

The study uses built-in synthetic fixtures and local objective verifiers. It yields a five-category development score. It does not use benchmark questions, an LLM judge, network access, or official leaderboard scoring. The LiveBench-shaped JSONL adapter recognizes single-turn rows with an applicable local spec or reference. Its coverage limits are part of the interface. The upstream row layout and grader dispatch were checked at LiveBench commit `1d1006a0d10ece52c2e706f8a208a659b5bbb96f` during implementation. No upstream code or datasets are executed by the offline suite.

- Math uses operands 1–12 and 1–8 respectively with `+`, `-`, `*`, and `%`. Commutative tasks use only ascending operand order, including ties. Grade the exact result with the Phase 20 arithmetic verifier.
- Coding uses the constants 1–24. Odd constants specify addition and even constants multiplication in a one-return `def f(x)` task. Check exact outputs on six inputs `-9, -3, 0, 2, 7, 13` through the Phase 20 interpreter.
- Reasoning assigns A and B distinct consecutive slots `k` and `k+1` with A before B, for `k=1..24`. Grade the full assignment and both constraints.
- Data analysis asks for a one-row table with `id,value` headers and cells `k,k+1`, for `k=1..24`. Odd constants specify CSV and even constants JSON. Score the exact table in the requested format.
- Instruction following requests a one-line, two-word response starting `go ` and ending with the supplied constant `k=1..24`. Grade all explicit checks.

Within each category, shuffle the generated pool with `random.Random(seed)` in the above category order. Reserve the first six prompts for CHECK, the next two for calibration, and the rest for FIT. Require all exact prompt strings to be disjoint across partitions. The partitions share templates. The built-in CLI uses the seed-3501 CHECK set, and the benchmark uses the corresponding held fixtures for each seed. Every held canonical response must pass its verifier. Do not pass held records to training.

## Training and comparison

Use the Phase 20 compact training geometry at each seed. The 64-wide, two-layer native skill backbone has context 64 and trains on CPU FP32 at constant LR 0.003 for one epoch of 192 AdamW updates with batch four. Give each FIT math record one copy and each FIT record of the other categories five copies. Shuffle the resulting raw prompt-and-response records with `Random(seed+100000)`, concatenate, and pad with newline bytes to one above a context multiple. Each run must train on 49,152 target bytes. Use the separate calibration bytes for four export-check blocks. Evaluate the exact checked INT8 export.

Score one nested candidate population per task. `K=1` is candidate zero, greedy. `K=16` adds 15 nucleus draws at temperature 0.7 and `top_p=0.9`. Search uses the independent frozen cursors and always-wake mode, without an SSD Engram in this experiment. For zero-based task index `i`, the search seed is `1000 × training_seed + 64 × i`, and samples add their candidate index to that seed. Candidate limits are 8 math, 24 coding, 22 reasoning, 48 data analysis, and 16 instruction bytes. Arithmetic and reasoning stop at one newline. The remaining categories stop at a blank-line delimiter. Use the highest causal logprob among verified proposals and report the highest-logprob miss when none verify.

For each category, score exact verified tasks divided by its six held tasks. The primary metric is the arithmetic mean of all five category scores, with equal weights. Record every selected response in hex, exact logprob, task counts, decoded bytes, and candidate and wall time. Replay the full nested comparison with the same task rows read from a local JSONL file. Timings may differ. All other scoreboard fields must match exactly.

## Decision rules

**Engineering.** The five verifier paths must pass valid, malformed, partial, and adversarial fixture checks with zero side effects. The builtin and custom JSONL CLI paths must replay exactly apart from timing. Require exact INT8 export, matched training budgets, complete five-category coverage for every scored seed, exact JSONL replay, and zero donor forwards. On the first seed's first held task run 100 prefix probes with four shared random continuation bytes, two five-byte suffixes, and a truncated control. Exact `0.0` difference is required. The regression suite also checks the combined Engram/INT8 prefix and search path. Require final bare pytest with zero failures, skips or warnings, legacy prefix causality at `0/100`, Git integrity, and a clean diff check.

**Scientific DEV.** PASS requires `K=16` to strictly beat `K=1` on macro accuracy in 5/5 seeds. MIXED requires one to four strict wins, and zero strict wins yields FAIL. Failed scored engineering makes the comparison INVALID. Nested proposals preserve earlier solutions by construction. Report per-category solves even when only one category drives the change. Official benchmark performance and broader skill transfer remain unmeasured.
