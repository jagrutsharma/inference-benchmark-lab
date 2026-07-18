#!/usr/bin/env bash
# Serve FP8-Dynamic Qwen2.5-3B on DGX Spark (GB10) — vLLM CUDA-13 nightly.
# Model is a local path, so mount the repo's models/ dir into the container.
# Docker flags (left of image) vs vLLM flags (right of image) — see notes.
docker run -d --name vllm-fp8 --ipc=host --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/code/inference-benchmark-lab/models:/models \
  vllm/vllm-openai:cu130-nightly \
  /models/Qwen2.5-3B-Instruct-FP8-Dynamic \
    --served-model-name Qwen2.5-3B-FP8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 4
