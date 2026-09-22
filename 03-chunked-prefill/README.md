# Stage 3 (cont.) — Chunked Prefill

**Status: incomplete.** The OFF baseline (Step 3) is run and captured below. The ON runs
(budget 2048, budget 512) and the chart/analysis (Steps 4-5) haven't been done yet —
tracked as a remaining Track A experiment.

**Goal:** make prefill/decode *interference* visible and then tune it away.

**Mechanism.** vLLM's scheduler builds one batch per step, under a token budget
(`max_num_batched_tokens`). A step mixes cheap decode tokens (1 per running sequence)
with an expensive prefill (the whole new prompt at once).
- **OFF:** a prefill must run in ONE step, so every in-flight decode freezes for the full
  prefill duration → a fat **ITL / TPOT p99 tail** (one user's long prompt hiccups everyone else).
- **ON:** the prefill is split into budget-sized chunks and interleaved with decode → flat
  ITL tail, at a small **TTFT** cost (that prefill's first token now arrives a bit later).

**Where it sits in the lab (third mechanism, third metric):**
- FP8 → each decode token *cheaper* (TPOT / throughput).
- Prefix caching → *skip* redundant prefill (TTFT median).
- **Chunked prefill → remove prefill/decode interference (ITL / TPOT p99 tail).**
It's the single-node, time-multiplexed version of disaggregation (Stage 6, Dynamo).

**Caveat (vLLM V1):** chunked prefill is ON by default (A6 server log:
`enable_chunked_prefill=True`, effective budget 2048). So all prior stages already had it on.
A7 turns it OFF to expose the interference, then dials chunk size to show the tradeoff.

## Contents

- [Step 1: Flags](#step-1-flags)
- [Step 2: The workload](#step-2-the-workload)
- [Step 3: Run with Chunked Prefill OFF (baseline)](#step-3-run-with-chunked-prefill-off-baseline)
- [Step 4: Run with Chunked Prefill ON (budget 2048, then 512)](#step-4-run-with-chunked-prefill-on-budget-2048-then-512) — pending
- [Step 5: Analysis & Chart](#step-5-analysis-and-chart) — pending
- [Gotchas](#gotchas)
- [Reference](#reference)

---

## Step 1: Flags

The chunked-prefill knobs (verified via `vllm serve --help=SchedulerConfig`):

- `--enable-chunked-prefill` / `--no-enable-chunked-prefill` — split prefills across steps or not.
  With it OFF, vLLM forces `max_num_batched_tokens >= max_model_len` so a prefill always fits one step.
- `--max-num-batched-tokens N` — the per-step token budget = the **chunk size**. Smaller N →
  more, smaller chunks → smoother decode ITL but slower prefill (higher TTFT). This is the dial.

Three arms (only this knob changes; everything else identical):

| Arm    | Flags                                                     | Prefill behavior                    |
|--------|----------------------------------------------------------|-------------------------------------|
| off    | `--no-enable-chunked-prefill`                            | 4096-tok prefill = 1 blocking step  |
| on2048 | `--enable-chunked-prefill --max-num-batched-tokens 2048` | 4096-tok prefill ≈ 2 chunks         |
| on512  | `--enable-chunked-prefill --max-num-batched-tokens 512`  | 4096-tok prefill ≈ 8 chunks         |

Scripts: `scripts/serve-chunked.sh {off|on2048|on512}` and `scripts/bench-chunked.sh {off|on2048|on512}`.

---

## Step 2: The workload

- **Dataset `random`** (not `prefix_repetition` like A6): random prompts share no prefix, so
  prefix caching can never fire → chunked prefill is the ONLY variable acting on prefill.
- **4096-in / 256-out**: long inputs make each prefill a big, blocking unit of work.
- **concurrency 16 + `--max-num-seqs 16`**: guarantees many sequences are mid-decode when a
  long prefill lands — that collision is the thing we measure. (Raised from the repo's usual 4;
  at low concurrency the interference barely shows.)
- 200 prompts, 3 warmups, greedy (`temperature 0`), `--ignore-eos` (force exactly 256 out tokens
  so every run does identical work).

**Headline metric:** ITL p99 and TPOT p99 (decode-latency tail). **Honest cost:** TTFT.

**Prediction to check against reality:**
- OFF: ITL p99 ≫ ITL median (a big tail); TTFT relatively low.
- ON: ITL p99 collapses toward the median; TTFT rises. 512 < 2048 on ITL tail, but 512 > 2048 on TTFT.

---

## Step 3: Run with Chunked Prefill OFF (baseline)

### 3a. Confirm the box is idle
```
docker ps            # nothing on :8000
nvidia-smi           # GPU memory near-empty
```

### 3b. Serve with chunked prefill disabled
```
cd 03-chunked-prefill
./scripts/serve-chunked.sh off
```

### 3c. Verify V1 actually honored the disable (before benchmarking)
```
docker logs vllm-cp 2>&1 | grep -iE "enable_chunked_prefill|max_num_batched_tokens"
```
Expect `enable_chunked_prefill=False` and an effective `max_num_batched_tokens >= 8192`. Actual:
```
non-default args: {..., 'gpu_memory_utilization': 0.85, 'max_num_seqs': 16, 'enable_chunked_prefill': False}
```
(`max_num_batched_tokens` isn't printed in `non-default args` when disabled — vLLM auto-raises
it to `max_model_len` (8192) internally, confirmed by the 4096-token prefill running as one
atomic step in the benchmark below, with no chunk-boundary stalls in the trace.)

### 3d. Run the benchmark
```
./scripts/bench-chunked.sh off
```
```
============ Serving Benchmark Result ============
Successful requests:                     200
Failed requests:                         0
Maximum request concurrency:             16
Benchmark duration (s):                  85.47
Total input tokens:                      819200
Total generated tokens:                  51200
Request throughput (req/s):              2.34
Output token throughput (tok/s):         599.03
Peak output token throughput (tok/s):    640.00
Peak concurrent requests:                32.00
Total token throughput (tok/s):          10183.55
---------------Time to First Token----------------
Mean TTFT (ms):                          163.75
Median TTFT (ms):                        156.32
P95 TTFT (ms):                           259.32
P99 TTFT (ms):                           321.64
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          25.35
Median TPOT (ms):                        25.56
P95 TPOT (ms):                           25.67
P99 TPOT (ms):                           25.71
---------------Inter-token Latency----------------
Mean ITL (ms):                           25.35
Median ITL (ms):                         25.30
P95 ITL (ms):                            27.24
P99 ITL (ms):                            37.43
==================================================
```

| Metric | Median | P95 | P99 | tail (p99/median) |
|---|---|---|---|---|
| ITL | 25.30 | 27.24 | 37.43 | **1.48x** |
| TPOT | 25.56 | 25.67 | 25.71 | ~flat |
| TTFT | 156.32 | 259.32 | 321.64 | - |

Matches the OFF prediction from Step 2: a real ITL p99 tail (1.48x median) with TPOT
median essentially flat — the interference shows up in the worst case, not the average.

### 3e. Copy the result out, then stop the server
```
docker cp vllm-cp:/root/results/chunked-prefill-off.json results/
docker rm -f vllm-cp
```

---

## Step 4: Run with Chunked Prefill ON (budget 2048, then 512)

Repeat Step 3's serve → verify → bench → copy → remove, changing only the arg.

### 4a. Budget 2048
```
./scripts/serve-chunked.sh on2048
docker logs vllm-cp 2>&1 | grep -iE "enable_chunked_prefill|max_num_batched_tokens"   # expect True / 2048
./scripts/bench-chunked.sh on2048
docker cp vllm-cp:/root/results/chunked-prefill-on2048.json results/
docker rm -f vllm-cp
```
⟨PASTE on2048 LOG + RESULT BLOCK HERE⟩

### 4b. Budget 512
```
./scripts/serve-chunked.sh on512
docker logs vllm-cp 2>&1 | grep -iE "enable_chunked_prefill|max_num_batched_tokens"   # expect True / 512
./scripts/bench-chunked.sh on512
docker cp vllm-cp:/root/results/chunked-prefill-on512.json results/
docker rm -f vllm-cp
```
⟨PASTE on512 LOG + RESULT BLOCK HERE⟩

---

## Step 5: Analysis and Chart

### 5a. Results table (fill from the three JSONs)

| Dimension                         | Better | OFF | ON 2048 | ON 512 |
|-----------------------------------|:------:|:---:|:-------:|:------:|
| Median ITL (ms)                   | lower  |     |         |        |
| **P99 ITL (ms)** (headline)       | lower  |     |         |        |
| Median TPOT (ms)                  | lower  |     |         |        |
| P99 TPOT (ms)                     | lower  |     |         |        |
| Median TTFT (ms) (cost)           | lower  |     |         |        |
| P99 TTFT (ms) (cost)              | lower  |     |         |        |
| Output throughput (tok/s)         | higher |     |         |        |

### 5b. What to explain (the "why", A6-style)

1. **The fingerprint:** does ITL p99 collapse while TPOT *median* barely moves? That's the
   signature of a decode-interference fix — the pain lived in the tail, not the average.
2. **The cost model for the ITL tail:**
   - OFF worst-case ITL ≈ time to prefill one 4096-tok step (all decoders wait that long).
   - ON worst-case ITL ≈ time for one chunk (budget-sized) + the interleaved decode.
   - Predict OFF-tail from the chunk math and compare to measured — same decomposition style as A6.
3. **The dial (2048 vs 512):** smaller chunks → lower ITL tail but higher TTFT. Quantify the
   trade: how many ms of TTFT did we pay per ms of ITL-p99 saved?
4. **Mirror to A6/FP8:** prefix caching moved TTFT median; chunked prefill moves ITL tail.
   Different phase, different metric — that's why the optimizations stack.
5. **Throughput:** expect it roughly flat (chunked prefill reshuffles *when* work happens, it
   doesn't remove work like caching or make it cheaper like FP8). Note any small delta and why.

### 5c. Chart + commit
```
python3 scripts/make-chart-chunked.py \
  results/chunked-prefill-off.json \
  results/chunked-prefill-on2048.json \
  results/chunked-prefill-on512.json \
  charts/chunked-prefill-itl.png

git add -A && git commit -m "A7: chunked prefill — ITL-tail before/after + chunk-size dial"
git push
```

## Gotchas

- **`max_num_batched_tokens` doesn't show up in `non-default args` when chunked prefill
  is disabled** — only `enable_chunked_prefill: False` does. vLLM auto-raises the budget
  to `max_model_len` internally; the confirmation is indirect (the 4096-token prefill runs
  as one atomic step with no chunk-boundary stalls), not a printed value.
- **Concurrency is raised to 16 here, not the repo's usual 4** — at low concurrency the
  interference this stage measures barely shows up, because there's rarely a long prefill
  landing while many sequences are mid-decode. This is a deliberate workload change, not
  an inconsistency with Stages 01/02.
- **Dataset is `random`, not `prefix_repetition`** — deliberately, so prefix caching can
  never fire and chunked prefill is isolated as the only variable acting on prefill.
- **This is the one stage still missing its ON arms.** Don't read the OFF-only numbers
  above as "chunked prefill doesn't help" — no comparison has been run yet.

## Reference

- Raw results so far: `results/chunked-prefill-off.json` (ON 2048/512 not yet run).
- Serve/bench scripts: `scripts/serve-chunked.sh {off|on2048|on512}`, `scripts/bench-chunked.sh {off|on2048|on512}`.
- Chart script: not yet written (`scripts/make-chart-chunked.py`, planned).
