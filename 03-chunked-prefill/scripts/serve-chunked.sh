#!/usr/bin/env bash
# A7 — serve FP8 Qwen2.5-3B with chunked prefill toggled, for the chunked-prefill A/B.
# Usage: ./serve-chunked.sh {off|on2048|on512}
#
# All three arms share the SAME serving config except the chunked-prefill knob, so the
# knob is the only variable. max-num-seqs is raised to 16 (vs 4 elsewhere in this repo)
# on purpose: we need MANY sequences decoding at once so that when a long prefill lands
# it collides with in-flight decodes — that collision is exactly what chunked prefill
# smooths, and it barely shows up at low concurrency.
#
#   off     --no-enable-chunked-prefill   -> a prefill runs as one atomic scheduler step
#                                            (vLLM auto-raises the token budget to >= max-model-len
#                                            so a 4096-tok prefill is never split). Every in-flight
#                                            decode freezes for that whole step -> fat ITL tail.
#   on2048  chunked prefill, 2048-token budget per step -> a 4096-tok prefill = ~2 chunks,
#                                            decode steps interleave between chunks.
#   on512   chunked prefill, 512-token budget -> ~8 smaller chunks -> smoother ITL tail,
#                                            but each prefill takes more steps to finish (TTFT cost).
set -euo pipefail

MODE="${1:?usage: serve-chunked.sh off|on2048|on512}"
case "$MODE" in
  off)    CP=(--no-enable-chunked-prefill) ;;
  on2048) CP=(--enable-chunked-prefill --max-num-batched-tokens 2048) ;;
  on512)  CP=(--enable-chunked-prefill --max-num-batched-tokens 512) ;;
  *) echo "unknown mode: $MODE (expected off|on2048|on512)" >&2; exit 1 ;;
esac

docker run -d --name vllm-cp --ipc=host --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/code/inference-benchmark-lab/models:/models \
  vllm/vllm-openai:cu130-nightly \
  /models/Qwen2.5-3B-Instruct-FP8-Dynamic \
    --served-model-name Qwen2.5-3B-FP8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 16 \
    "${CP[@]}"
