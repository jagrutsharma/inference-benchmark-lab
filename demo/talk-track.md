# 5-Minute Demo Talk-Track

## 0. Frame (30s)
Personal project: I'm a backend / distributed-data engineer moving one layer
deeper into inference serving. Built a progressive-optimization lab on a DGX Spark.
Each optimization is measured and explained — not just "number went up."

## 1. The box & the one fact that drives everything (45s)
DGX Spark: GB10, 128 GB unified memory, ~273 GB/s bandwidth.
KEY: it's memory-BANDWIDTH-bound, not compute-bound.
Decode = stream all weights across the bus per token; tiny compute.
So decode speed is set by BYTES MOVED, not FLOPs. Everything follows from this.

## 2. Optimization 1 — FP8 quantization (90s)   [SHOW: fp8-vs-bf16.png]
Open the table in Google Doc -> A4 (** FP8 Quantization)
Quantized weights bf16 -> FP8. Model 5.75 -> 3.2 GB (44% smaller).
Result: 1.67x decode throughput (139 -> 232 tok/s), TPOT 27.75 -> 16.57 ms.
WHY: half the bytes per weight on a bandwidth-bound box.
WHY NOT 2x: KV cache stayed bf16, and fixed overhead doesn't shrink.
Honest tell: TPOT and throughput both moved 1.67x — same win, two measurements.
Quality: near-lossless on spot-check.

## 3. Optimization 2 — prefix caching (90s)   [SHOW: prefix-caching-ttft.png]
Open the table in Google Doc -> A5 (** Prefill Caching)
Shared-prefix workload (think RAG / shared system prompts).
Result: median TTFT 424 -> 73 ms = 5.8x faster.
WHY: reuse the shared prefix's KV cache — skip re-prefilling ~94% of the prompt.
KEY CONTRAST: this hits PREFILL (TTFT). FP8 hit DECODE (TPOT). Mirror images.
Honest: 5.8x not 17x because suffix prefill + fixed overhead can't be cached.
p99 stays high = the cold-miss requests (first per prefix).

## 4. The insight that ties them (30s)
Two optimizations, two different phases:
- FP8 makes each token CHEAPER (decode / bandwidth).
- Prefix caching SKIPS redundant work (prefill / reuse).
Different mechanisms, different metrics — that's WHY they stack.

## 5. What's next (30s)
- Chunked prefill: single-node, time-multiplexed fix for prefill/decode interference.
- Disaggregation (Dynamo): the multi-GPU, space-partitioned version of the same idea.
- Cross-platform: same pipeline on AMD MI300X (NVIDIA vs AMD benchmark).
Repo is public, fully reproducible — scripts + raw results + charts.

## If asked "did you understand it or just run it?"
- KV cache = 36 KB/token here (2 * 36 layers * 2 KV heads * 128 dim * 2 bytes); GQA gives 8x reduction.
- Roofline: 5.75 GB / 273 GB/s ~= 21 ms/token floor; measured 27.75 (KV + overhead on top).
- Capacity ceiling 333 seqs but run 4 — bandwidth binds long before capacity.
