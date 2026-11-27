# Owner-executed Colab receipts

This log transcribes the owner's externally executed Colab run details. Artifact sizes, SHA-256 digests and metrics are recorded as supplied. The release candidate's local Linux CPU regression is recorded separately. [Release review](PR_V1_MAIN_REVIEW.md#release-candidate-qualification)

## Tesla T4 CUDA FP32 training with SDPA

The owner ran `train --use-sdpa` on `2ca22b298cf8d61dad68aefea731b56ff83b4053` using a 4 MB corpus, width 128 and context 512. The run completed 16/16 epochs in approximately 33 seconds and processed 6,289,402 target bytes.

`cli.py verify` reported **PASS** for the 1,215,342-byte snapshot. The reported snapshot SHA-256 is `9c9a07ea130f01828868f9425757247c97bbb6e2808a611ca166485c324408cd`.

| Verification check | Reported maximum difference |
| --- | ---: |
| Prefix logits | 0.0 |
| Prefix hidden states | 0.0 |
| Streaming logits | 0.0 |
| Unified stream probabilities | 0.0 |
| Snapshot round-trip probabilities | 0.0 |

## Google Drive FUSE training

The owner ran training on `65547bfff507f217516d3040303a3b89da7c8151` and completed 8/8 epochs with writes directly to Google Drive FUSE. `cli.py verify` reported **PASS**. The reported SHA-256 is `27da3e5fca04bd5578af84ed1b2bb25ce75ffa9e426a4b04dfaf25fd9b54c43f`.

## Qwen2.5-0.5B zero-execution assimilation

The owner imported safetensors from `Qwen/Qwen2.5-0.5B` on `2ca22b298cf8d61dad68aefea731b56ff83b4053`. The reported donor size was 988 MB with width 896 and 151,651 vocabulary tokens.

| Receipt field | Reported value |
| --- | --- |
| Standalone skill size | 438,096 bytes |
| Skill SHA-256 | `8357c75cda76b3a862f625849b3eaf2e870579695bfe37973373ff81bdd0a70f` |
| `peak_loaded_shards` | 1 |
| `donor_forward_calls` | 0 |
| Parity difference | 0.0 |

## DeepSeek-V4.1-Flash-1017 streamed zero-execution assimilation

The owner used `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision `08c8657f02291e11b6f6b63e0e16014f8e748518`. The reported 763B-parameter checkpoint spans 48 shards and approximately 510 GB. The runs reported `peak_disk_shards = 1`, `peak_loaded_shards = 1` and `donor_forward_calls = 0`. These are peak counts for the selected shards. [Pinned donor revision](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/08c8657f02291e11b6f6b63e0e16014f8e748518)

### 32-wide, two-layer skill

The owner ran `145c00226fbcfd285efac04fad5ff7b322d7c2fe` and reported a 438,096-byte skill with SHA-256 `003f4d3bd69cc4513ee474af94152660d550fa508f4f49677c54597b1e0e635f`. The supplied receipt gives BPB changes.

| Domain | Zero-update Δ BPB | After 64 updates Δ BPB |
| --- | ---: | ---: |
| Markdown | -1.0441 | -0.2195 |
| Python | -1.3503 | -0.0179 |

### 128-wide, four-layer skill

The owner ran `12c5b87141aaae4c09888afc9948180ed98ce2e7`. Lower bits per byte (BPB) is better. The table preserves the reported changes versus scratch. Subtracting the displayed BPB values can differ from the reported change by 0.0001.

| Stage | Domain | Assimilated BPB | Scratch BPB | Reported Δ BPB |
| --- | --- | ---: | ---: | ---: |
| Zero updates | Markdown | 7.1016 | 8.3950 | -1.2933 |
| Zero updates | Python | 6.7206 | 8.3879 | -1.6674 |
| After 64 updates | Markdown | 4.5065 | 5.1148 | -0.6083 |
| After 64 updates | Python | 5.8137 | 6.5914 | -0.7778 |

### 256-wide, six-layer skill

The owner reported an 8,259,317-byte skill with SHA-256 `caae641862b01b436c54c9b3e9e69970f0af2dfc04595746bf162ebc85ed12e7`. The import used 8 of 48 shards, with `peak_disk_shards = 1`, `peak_loaded_shards = 1` and `donor_forward_calls = 0`. The prefix and snapshot round-trip differences were both 0.0.

The 64-step adaptation used `RESEARCH.md`. The table records the supplied BPB values and changes versus scratch.

| Stage | Domain | Assimilated BPB | Scratch BPB | Reported Δ BPB |
| --- | --- | ---: | ---: | ---: |
| Zero updates | Markdown | 7.0603 | 8.2330 | -1.1728 |
| Zero updates | Python | 6.6980 | 8.3027 | -1.6047 |
| After 64 updates | Markdown | 4.6386 | 5.1550 | -0.5165 |
| After 64 updates | Python | 5.9888 | 6.4030 | -0.4142 |

## Phase 13 DeepSeek warm-start and CPU generation on Colab

On a Colab Tesla T4 runtime at `bf39629efb44397c24d5d7a10a9b6fd853c0f3c5`, the owner used a synthetic 4 MB corpus and a 128-wide, four-layer DeepSeek warm-start. CPU generation took 38.13–56.76 ms/byte with `donor_forward_calls: 0`. The reported `snapshot_sha256` was `d89919248527090dc4dd3d7a9672c0cc2a696722b1105890a62e18eda862ecb4`.

## Phase 14 real-text training, verification and T4 CUDA generation

At `d9b79f014edc96c7d4558debcef72283383da62a`, the owner trained on a corpus of 821,018 bytes from local repository prose and code plus math, with `global_overlap_32b: 0`. The matched warm-start and scratch runs used width 128, four layers, context 512, `--use-sdpa`, 16 epochs of 64 steps, and 3,138,649 target bytes each.

| Run | Snapshot bytes | `neural_state_sha256` | `snapshot_sha256` |
| --- | ---: | --- | --- |
| Warm-start with `--init-skill deepseek_v41_128x4.pt` | 2,843,357 | `3d974e39ba6cee54c736bfdfa8461f811a9a321c9a74f6dd2b495265297321d0` | `e2bd9b30bc84f9568fb989ed1e6d95bbd007be1b50a72658415540efa0a6879c` |
| Scratch with `--skill-backbone` | 2,843,357 | `225c62487c9b373f4d0f362f75b8cae9000929296f4ee7864e31e6d0ac3c3bae` | `ceb9219ba8c0a7240b0d2c68404f3c9079ae1b5ef75978514887298f75a04bc2` |

Both runs reported `cold_equal: true`, `exact_state: true`, `donor_forward_calls: 0` and `0.0` for all four parity checks and the snapshot round-trip. For 200 greedy bytes on a matched story prompt from the warm-start snapshot, CPU decode took 47.87 ms/byte with 112.38 ms prefill. T4 CUDA decode took 4.33 ms/byte with 303.79 ms prefill. The owner reported an **11.1× decode speedup** and identical `generated_hex`. Across all CUDA prompts the reported range was 4.16–5.07 ms/byte, with zero donor forward calls.

Owner runs streaming `roneneldan/TinyStories` at 12 MB and `Salesforce/wikitext:wikitext-103-raw-v1` at 3 MB stalled in the Phase 14 corpus splitter. Phase 14B covers the fix with an offline stream fixture. Live Hub streaming on Colab awaits the owner's rerun.

## Open qualification

Independent scientific qualification, multi-GB GPU pretraining and live DeepSeek Engram extraction from the approximately 101.5 GB Engram shards remain open. The DeepSeek BPB values above are owner-run measurements at the stated revisions. The preregistered Phase 12 comparison used a local BPE donor and a FIT-only static byte table. [Phase 12 evidence](../evidence/phase12-scaled-zero-exec-dev-001.json)
