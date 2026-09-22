#!/usr/bin/env bash
# bf16 baseline — Qwen2.5-3B-Instruct on DGX Spark (GB10), vLLM cu130-nightly
# Result: 139.7 tok/s output, TPOT 27.75ms, TTFT 257ms p50 / 314ms p99 @ concurrency 4
docker exec -it vllm vllm bench serve \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset-name random \
  --num-prompts 200 \
  --num-warmups 3 \
  --random-input-len 1024 \
  --random-output-len 256 \
  --max-concurrency 4 \
  --temperature 0.0 \
  --ignore-eos \
  --metric-percentiles 50,95,99 \
  --save-result --save-detailed \
  --result-dir /root/results \
  --result-filename baseline-bf16.json \
  --metadata precision=bf16 model=Qwen2.5-3B gpu=GB10
