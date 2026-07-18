#!/usr/bin/env bash
# FP8 benchmark — identical to bench-baseline.sh except model name, tokenizer, output file.
# --tokenizer points at the local path because the served-name isn't a real HF repo.
# Result: 232.3 tok/s output, TPOT 16.57ms, TTFT 177ms p50 @ concurrency 4 (1.67x over bf16).
docker exec -it vllm-fp8 vllm bench serve \
  --model Qwen2.5-3B-FP8 \
  --tokenizer /models/Qwen2.5-3B-Instruct-FP8-Dynamic \
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
  --result-filename fp8-dynamic.json \
  --metadata precision=fp8-dynamic model=Qwen2.5-3B gpu=GB10
