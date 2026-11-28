# Owner-executed Colab and Windows receipts

This log transcribes the owner's externally executed Colab and local Windows run details. Artifact sizes, SHA-256 digests and metrics are recorded as supplied. The release candidate's local Linux CPU regression is recorded separately. [Release review](PR_V1_MAIN_REVIEW.md#release-candidate-qualification)

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

Owner runs streaming `roneneldan/TinyStories` at 12 MB and `Salesforce/wikitext:wikitext-103-raw-v1` at 3 MB stalled in the Phase 14 corpus splitter. The owner later reran TinyStories on the Phase 14B release. The wikitext rerun remains open.

## Phase 14B TinyStories Hub streaming and CUDA decode

On the `v1.1.0-rc1` checkout at `e124207417be94c3ea242a8c5042067860ac23f5`, the owner streamed TinyStories from the Hub and trained a 128-wide, four-layer model for 16 epochs of 64 steps on Colab. CUDA decode took 0.71–4.61 ms/byte. The run reported `donor_forward_calls: 0` and snapshot SHA-256 `3be436c6e81047e1c2dd3a434906648638ecf237635e9f14a573b63064f912cd`.

## Phase 15 cosine training and always-awake generation

At `6d23046834673c338298348e21c21e92f809629b`, the owner trained a 128-wide, four-layer TinyStories model for 4,096 steps with cosine learning-rate scheduling on Colab. The run reported `donor_forward_calls: 0`. Its snapshot, `v12_ts_128x4_4096steps.pt`, had SHA-256 `152c21d8cc969f6855f6b2415d3262897d1c55d71c0c262257f229e97f672c2e`.

On that snapshot, the owner reproduced a repeated-space loop under the default `BytePatchReader` gate. Setting the restored reader's `minimum_probability` to `2.0` before generation kept the neural path active and cleared the loop on all four supplied prompts, which covered prose greedy and nucleus decoding and Python greedy decoding. This manual setting is the behavior exposed by `generate --always-wake` in Phase 15B.

## Windows RTX 5070 Ti Laptop GPU training and generation

The owner ran `v1.2.0` at `d1936aaa760c644915653ddc1dcefedf0ce32aa6` in Windows 11 PowerShell on an NVIDIA GeForce RTX 5070 Ti Laptop GPU with PyTorch `2.11.0+cu128`. The 45 MB corpus comprised TinyStories, local repository Python, and synthetic math from `scripts/prepare_real_corpus.py`, with zero shared 32-byte train/eval windows.

The FP32 CUDA run used width 512, eight layers, context 512, and 12 epochs of 256 steps at `--batch-per-domain 4`. It used `--lr 0.001 --lr-schedule cosine --skill-backbone --use-sdpa --dtype fp32`. It processed 18,872,770 target bytes at about 25.7–28.5 seconds per epoch and about 5.4 minutes total. The run reported `cold_equal: true`, `exact_state: true`, `0.0` across all four parity checks and snapshot round-trip, and `donor_forward_calls: 0`.

`cli.py verify` reported **PASS** on the 70,015,619-byte snapshot. The neural-state SHA-256 was `d334ff74af5720737943223d5c0a0dfd5f22d1d9f8ca4a593708534101c20b20`, and the snapshot SHA-256 was `44ec5dfed7030e36cfe7791fbc3d5cbbfae4bcb3d7a259bece9a24f72b3e0a27`.

| Held-out domain | BPB | Validation B/s |
| --- | ---: | ---: |
| Prose | 1.880117 | 137,039 |
| Code | 10.170066 | 138,945 |
| Math | 10.541899 | 32,555 |

The owner attributed the code result to overfitting on about 60 KB of local Python. The receipt reported `peak_cuda_allocated_bytes: 144,418,304`. Live `generate --device cuda --always-wake` took 4.11–4.28 ms/byte with zero donor forward calls.

The owner also caught a BF16 autocast dtype mismatch in the gated FFN on this hardware. The v1.2.1 fix casts the FFN update to the destination dtype before indexed assignment. Later owner-executed Windows BF16 runs are listed below.

## Windows 11 RTX 5070 Ti BF16 and INT8 receipts

The owner executed these runs on Windows 11 with an RTX 5070 Ti and PyTorch `2.11.0+cu128`, with BF16 support reported as `True`. The snapshot identifiers below reproduce the supplied prefixes.

| Run | Snapshot | Verification | Held-out BPB |
| --- | --- | --- | --- |
| Monarch BF16 + INT8, 512×8, `v1.2.1` | `ce59bf1a…`, 10,570,865 bytes, 6.62× smaller than 70.02 MB Dense FP32 | PASS, `0.0` parity | Pooled 4.8607, prose 3.5749, code 5.5215, math 5.4858 |
| Pure-prose TinyStories BF16 + INT8, 512×8, Phase 17 | `d616c75f…`, 18,226,989 bytes, 3.84× INT8 compression | PASS, `0.0` parity | 1.8276 |
| Expansive FineWeb-Edu BF16 + INT8, 512×8, `v1.3.0` | `e32fd6a9…`, 18,225,581 bytes, 3.84× INT8 compression | PASS, `0.0` parity | 1.4363 |

Monarch CUDA decode took 7.99 ms/byte. The expansive FineWeb-Edu run used a 150 MB source corpus, completed 64/64 epochs, and processed 268,434,685 target bytes. On the uninstructed prose snapshot, `eval-livebench` at `1f02106` with `--builtin-dev-suite`, `K=16`, and CUDA scored 0.0% in 158.7 s and reported `donor_forward_calls: 0`. The owner traced unrelated generation continuations to the snapshot's 40 KB `ConsolidatingMemory` and its FineWeb-Edu training tail.

Phase 22B adds a session-local bypass for that memory and corrects the NTFS Engram identity check. Offline regressions cover both changes. These receipts predate those fixes, and the corresponding live Windows retest remains open.

## Open qualification

Independent scientific qualification, multi-GB GPU pretraining, full-scale 256×6 and 512×8 training on L4/A100 GPUs, and live DeepSeek Engram extraction from the approximately 101.5 GB Engram shards remain open. The DeepSeek BPB values above are owner-run measurements at the stated revisions. The preregistered Phase 12 comparison used a local BPE donor and a FIT-only static byte table. [Phase 12 evidence](../evidence/phase12-scaled-zero-exec-dev-001.json)
