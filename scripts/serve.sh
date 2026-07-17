#!/usr/bin/env bash
# Serve Qwen2.5-3B-Instruct on DGX Spark (GB10) — vLLM CUDA-13 nightly (stable image lacks sm_121)
docker run -d --name vllm --ipc=host --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu130-nightly \
  Qwen/Qwen2.5-3B-Instruct \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 4
