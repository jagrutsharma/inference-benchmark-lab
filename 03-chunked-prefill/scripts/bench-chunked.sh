#!/usr/bin/env bash
# A7 — chunked-prefill benchmark. One command per arm; the served config differs, the
# benchmark itself is identical across all three so the only variable is the server knob.
# Usage: ./bench-chunked.sh {off|on2048|on512}
#
# Why the RANDOM dataset (not prefix_repetition like A6): random prompts share no prefix,
# so prefix caching can never fire. That isolates chunked prefill as the ONLY thing acting
# on prefill — a clean single-variable experiment.
#
# Why 4096-in / concurrency 16: long inputs make each prefill a big, blocking unit of work,
# and high concurrency guarantees other requests are mid-decode when that prefill is scheduled.
# That is the prefill/decode collision we want to measure. Headline metric = ITL/TPOT p99
# (the decode-latency tail); honest cost = TTFT.
set -euo pipefail

MODE="${1:?usage: bench-chunked.sh off|on2048|on512}"

# docker exec needs a TTY (-t) for the pretty progress bar, but -t fails when there's no
# terminal (e.g. run from automation). Detect and only add -it when stdin is a real TTY.
TTY=""; [ -t 0 ] && TTY="-it"

docker exec $TTY vllm-cp vllm bench serve \
  --model Qwen2.5-3B-FP8 \
  --tokenizer /models/Qwen2.5-3B-Instruct-FP8-Dynamic \
  --dataset-name random \
  --num-prompts 200 \
  --num-warmups 3 \
  --random-input-len 4096 \
  --random-output-len 256 \
  --max-concurrency 16 \
  --temperature 0.0 \
  --ignore-eos \
  --metric-percentiles 50,95,99 \
  --save-result --save-detailed \
  --result-dir /root/results \
  --result-filename chunked-prefill-$MODE.json \
  --metadata chunked_prefill=$MODE precision=fp8-dynamic model=Qwen2.5-3B gpu=GB10
