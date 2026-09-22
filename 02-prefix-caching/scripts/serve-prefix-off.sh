#!/usr/bin/env bash
# Prefix caching OFF — second server, same FP8 model, on host port 8001 so it doesn't
# conflict with the vllm-fp8 container from 01-fp8-quantization (which stays up on 8000
# and serves the "caching ON" case, since prefix caching is on by default in vLLM).
docker run -d --name vllm-fp8-nocache --ipc=host --gpus all -p 8001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/code/inference-benchmark-lab/models:/models \
  vllm/vllm-openai:cu130-nightly \
  /models/Qwen2.5-3B-Instruct-FP8-Dynamic \
    --served-model-name Qwen2.5-3B-FP8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching
