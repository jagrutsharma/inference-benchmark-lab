<!--
Images to attach when posting:
1. ../01-fp8-quantization/charts/fp8-vs-bf16.png — FP8 vs bf16 headline chart
2. ../02-prefix-caching/charts/prefix-caching-ttft.png — prefix caching TTFT p50/p95/p99, cache on vs off
-->

Two different optimizations, two different bottlenecks, same GPU. Here's what I found running both on an NVIDIA DGX Spark.

**FP8 quantization.** Same model (Qwen2.5-3B-Instruct), same benchmark, only weight precision changed: bf16 to FP8. Output throughput jumped 1.66x, decode (TPOT) got 1.67x faster.

Why: the Spark is memory-bandwidth-bound, not compute-bound. Decoding one token means streaming the entire weight set across a ~273 GB/s bus — the arithmetic itself is cheap. FP8 halves the bytes per weight, and on a bandwidth-bound box the speedup tracks the byte reduction almost directly. Checked this wasn't just a good guess: the roofline floor (model size / bandwidth) predicts ~21ms/token; measured bf16 TPOT is 27.75ms — only 1.31x that floor. Almost the whole budget really is bytes on the bus.

**Prefix caching.** Different mechanism entirely, different phase. A cache hit skips re-prefilling a shared prompt prefix and only processes the unique suffix. Result: TTFT (time to first token) dropped 5.8x, while TPOT stayed flat — because this attacks prefill, not decode.

The 5.8x number needed a sanity check of its own. 94% of the prompt was cached, so the naive guess is more like 17x. It isn't, because TTFT is really three additive costs: shared-prefix prefill + unique-suffix prefill + fixed per-request overhead. Caching only zeroes out the first term — the other two don't care whether anything is cached, so they set a floor the speedup can't get past. Do the arithmetic on the actual measured pieces and it reconciles with the 5.8x almost exactly.

That's the pattern worth remembering: FP8 makes every decode step cheaper and barely touches prefill; prefix caching removes prefill work and barely touches decode. Different phase, different mechanism — which is exactly why they stack instead of competing for the same budget.

Full writeup, real commands, raw JSON: https://github.com/jagrutsharma/inference-benchmark-lab/blob/main/README.md

#LLMInference #vLLM #GPU #Quantization #Benchmarking
