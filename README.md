# Inference Benchmark Lab

Progressive LLM inference optimization on an NVIDIA DGX Spark (GB10 Grace-Blackwell,
128 GB unified LPDDR5x, ~273 GB/s memory bandwidth, sm_121). Each stage is measured
(TTFT, TPOT, throughput, p50/p95/p99, memory) so every optimization is backed by
numbers and an explanation of *why* it moves them. Each stage folder below is
self-contained: its own scripts, raw results, chart, and full writeup.

## Stages

| Stage | Optimization | Headline result | Details |
|---|---|---|---|
| 1 + 3 | FP8 dynamic quantization | **1.66x** throughput, **1.67x** faster TPOT | [01-fp8-quantization/](01-fp8-quantization/) |
| 3 (cont.) | Prefix caching | **5.8x** faster median TTFT, TPOT unchanged | [02-prefix-caching/](02-prefix-caching/) |
| 3 (cont.) | Chunked prefill | OFF baseline only — ON runs pending | [03-chunked-prefill/](03-chunked-prefill/) |
| Cross-platform | Same FP8 pipeline on AMD MI300X | 1.04x/1.08x — bandwidth-rich box barely moves | [cross-platform-mi300x/](cross-platform-mi300x/) |

FP8 and prefix caching are mirror-image optimizations: FP8 makes every decode step
cheaper (attacks TPOT), prefix caching skips redundant prefill work (attacks TTFT).
Different phase, different mechanism — which is why they stack instead of competing
for the same budget. See each stage's README for the full "why," the roofline/KV-cache
math, and exact reproduce steps.

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
- [x] Stage 3 (cont.) — prefix caching, measured (TTFT 5.8x, TPOT unchanged)
- [ ] Stage 3 (cont.) — chunked prefill: OFF baseline captured, ON (2048/512) + chart pending
- [ ] Concurrency sweep — p50 vs p99 knee across concurrency 1..16
- [ ] Stage 4 — Profile: Nsight Systems / Compute
- [ ] Stage 5 — Speculative decoding (EAGLE)
- [ ] Stage 6 — Distributed: disaggregated prefill/decode (NVIDIA Dynamo)
- [ ] Stage 7 — Observability: Prometheus + Grafana
- [x] Cross-platform — same pipeline on AMD MI300X (ROCm vLLM), see [cross-platform-mi300x/](cross-platform-mi300x/)
