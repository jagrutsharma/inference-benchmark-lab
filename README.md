# Inference Benchmark Lab

Progressive LLM inference optimization on an NVIDIA DGX Spark (GB10 Grace-Blackwell,
128 GB unified LPDDR5x, ~273 GB/s memory bandwidth, sm_121). Each stage is measured
(TTFT, TPOT, throughput, p50/p95/p99, memory) so every optimization is backed by
numbers and an explanation of *why* it moves them.

## Headline result (Stage 1 + 3)

FP8 dynamic quantization of Qwen2.5-3B-Instruct, measured before/after on the same box,
same serving config, same benchmark. Only weight precision changed.

| Metric | bf16 baseline | FP8 dynamic | Change |
|---|---|---|---|
| Output throughput | 139.7 tok/s | 232.3 tok/s | **1.66x** |
| TPOT (median) | 27.75 ms | 16.57 ms | **1.67x faster** |
| TTFT (median) | 257 ms | 177 ms | 1.45x faster |
| Model size (on disk) | 5.75 GB | 3.17 GB | 45% smaller |
| Weights (in memory) | 5.79 GB | 3.23 GB | 44% smaller |
| Quality (spot-check) | coherent | coherent | near-lossless |

Benchmark: 200 prompts, 1024 input / 256 output tokens, concurrency 4, greedy
(temperature 0), `--ignore-eos` for fixed output length.

## Why the gain is a bandwidth story

The Spark is memory-bandwidth-bound, not compute-bound. During decode, generating one
token requires streaming the entire weight set across the ~273 GB/s memory bus, while
the actual arithmetic per token is small. So decode time is dominated by bytes moved,
not FLOPs.

FP8 halves the bytes per weight (2 bytes to 1), so on a bandwidth-bound box the decode
speedup tracks the byte reduction. The result (1.67x) lands just short of the theoretical
2x because:

- The KV cache stayed bf16 (`kv_cache_dtype=auto`), so KV-read bytes per token did not shrink.
- Fixed per-step overhead (kernel launches, scheduling, sampling) does not scale with precision.

TPOT (decode) improved 1.67x but TTFT (prefill) only 1.45x. Prefill processes all 1024
input tokens at once and is more compute-bound, so a bandwidth optimization helps it less.
The gap between the two speedups is the signature of decode being bandwidth-bound and
prefill being partly compute-bound.

## Reproduce

Serve and benchmark the bf16 baseline:
./scripts/serve.sh
./scripts/bench-baseline.sh

Quantize to FP8, then serve and benchmark:
python scripts/quantize-fp8.py        # in a venv with: pip install llmcompressor
./scripts/serve-fp8.sh
./scripts/bench-fp8.sh

Raw benchmark results (with the exact commands and metadata) are in `results/`.

## Environment

- NVIDIA DGX Spark, GB10 Grace-Blackwell, 128 GB unified memory, DGX OS (ARM64).
- vLLM served via the CUDA-13 nightly container (`vllm/vllm-openai:cu130-nightly`);
  the stable image does not support GB10 (sm_121).
- Quantization via `llm-compressor` (FP8_DYNAMIC: per-channel static weight scales,
  per-token dynamic activation scales, no calibration data required).

## Roadmap

- [x] Stage 1 — Serve: vLLM + Docker + OpenAI-compatible API (baseline, measured)
- [ ] Stage 2 — Stream: WebSockets / gRPC streaming
- [x] Stage 3 — Quantize: FP8 dynamic, measured (throughput / TPOT / memory)
- [ ] Stage 3 (cont.) — prefix caching, chunked prefill (measure each)
- [ ] Concurrency sweep — p50 vs p99 knee across concurrency 1..16
- [ ] Stage 4 — Profile: Nsight Systems / Compute
- [ ] Stage 5 — Speculative decoding (EAGLE)
- [ ] Stage 6 — Distributed: disaggregated prefill/decode (NVIDIA Dynamo)
- [ ] Stage 7 — Observability: Prometheus + Grafana
- [ ] Cross-platform — same pipeline on AMD MI300X (ROCm vLLM + Quark)
