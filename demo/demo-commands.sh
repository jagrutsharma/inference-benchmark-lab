#!/usr/bin/env bash
# ── LIVE DEMO CHEAT SHEET ── run these one at a time (do NOT execute the whole file)

# 0. Confirm server is up
docker ps
curl -sS http://localhost:8000/v1/models | jq -r '.data[0].id'

# 1. Show a single completion (proves it serves)
curl -sS http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-3B-FP8","messages":[{"role":"user","content":"Explain what a KV cache is in two sentences."}]}' | jq -r '.choices[0].message.content'

# 2. Show streaming (tokens arrive one per step)
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-3B-FP8","messages":[{"role":"user","content":"Count to ten slowly."}],"stream":true}'

# 3. Show the live metrics endpoint (Stage 7 preview)
curl -sS http://localhost:8000/metrics | grep -E "vllm:(generation_tokens|prompt_tokens|num_requests)" 

# ── CHARTS to show (open these images) ──
# ../01-fp8-quantization/charts/fp8-vs-bf16.png       -> FP8: 1.67x decode, 44% smaller
# ../02-prefix-caching/charts/prefix-caching-ttft.png -> prefix caching: 5.8x median TTFT
