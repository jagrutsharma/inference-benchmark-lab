#!/usr/bin/env bash
# A6 — prefix caching benchmark. Same workload both times; only the server (and thus
# whether prefix caching is active) differs, so caching is the only variable.
# Usage: ./bench-prefix.sh {on|off}
#   on  -> execs into vllm-fp8 (01-fp8-quantization's server, caching on by default)
#   off -> execs into vllm-fp8-nocache (started by serve-prefix-off.sh, --no-enable-prefix-caching)
#
# Dataset prefix_repetition: 5 distinct 2048-token prefixes, each with a unique 128-token
# suffix, 200 prompts total (~40/prefix) so the shared prefix gets heavily reused.
set -euo pipefail

MODE="${1:?usage: bench-prefix.sh on|off}"
case "$MODE" in
  on)  CONTAINER=vllm-fp8 ;;
  off) CONTAINER=vllm-fp8-nocache ;;
  *) echo "unknown mode: $MODE (expected on|off)" >&2; exit 1 ;;
esac

docker exec -it "$CONTAINER" vllm bench serve \
  --model Qwen2.5-3B-FP8 \
  --tokenizer /models/Qwen2.5-3B-Instruct-FP8-Dynamic \
  --dataset-name prefix_repetition \
  --num-prompts 200 \
  --num-warmups 3 \
  --prefix-repetition-prefix-len 2048 \
  --prefix-repetition-suffix-len 128 \
  --prefix-repetition-num-prefixes 5 \
  --prefix-repetition-output-len 128 \
  --max-concurrency 4 \
  --temperature 0.0 \
  --ignore-eos \
  --metric-percentiles 50,95,99 \
  --save-result --save-detailed \
  --result-dir /root/results \
  --result-filename prefix-cache-$MODE.json \
  --metadata prefix_caching=$MODE precision=fp8-dynamic
