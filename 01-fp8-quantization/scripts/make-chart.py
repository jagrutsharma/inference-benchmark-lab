#!/usr/bin/env python3
"""
A5 — before/after chart for the FP8 vs bf16 inference benchmark.

Reads the two `vllm bench serve` result JSONs and renders a 2x2 comparison:
output throughput, TPOT (decode), TTFT (prefill), and model memory.

Usage:
    python make-chart.py [baseline.json] [fp8.json] [out.png]
Defaults to results/baseline-bf16.json, results/fp8-dynamic.json, charts/fp8-vs-bf16.png
"""
import json
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- inputs ----
baseline_path = sys.argv[1] if len(sys.argv) > 1 else "results/baseline-bf16.json"
fp8_path      = sys.argv[2] if len(sys.argv) > 2 else "results/fp8-dynamic.json"
out_path      = sys.argv[3] if len(sys.argv) > 3 else "charts/fp8-vs-bf16.png"

bf16 = json.load(open(baseline_path))
fp8  = json.load(open(fp8_path))

# ---- memory numbers are NOT in the benchmark JSON; they came from the serve boot logs.
# ---- VERIFY these against your logs before trusting the memory panel.
MEM_BF16_GIB = 5.79   # "Model loading took 5.79 GiB" (bf16 boot log)
MEM_FP8_GIB  = 3.23   # "Model loading took 3.23 GiB" (fp8 boot log)

# ---- colors ----
C_BF16 = "#8899A6"   # muted slate — the "before"
C_FP8  = "#2E7D6B"   # teal — the "after"

# ---- figure ----
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("FP8 Dynamic vs bf16 — Qwen2.5-3B-Instruct on DGX Spark (GB10)",
             fontsize=15, fontweight="bold", y=0.98)
subtitle = (f"{bf16['num_prompts']} prompts · 1024 in / 256 out · concurrency "
            f"{bf16['max_concurrency']} · greedy · same model, only weight precision changed")
fig.text(0.5, 0.935, subtitle, ha="center", fontsize=9.5, color="#555")

labels = ["bf16", "FP8"]
colors = [C_BF16, C_FP8]

def bar_panel(ax, vals, title, unit, higher_is_better, fmt="{:.1f}"):
    bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="white", linewidth=1.5)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=28)
    ax.set_ylabel(unit, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.25)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, fmt.format(v),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    # speedup / delta annotation
    if higher_is_better:
        factor = vals[1] / vals[0]
        note = f"{factor:.2f}x higher"
    else:
        factor = vals[0] / vals[1]
        pct = (vals[0] - vals[1]) / vals[0] * 100
        note = f"{factor:.2f}x faster  (-{pct:.0f}%)"
    ax.text(0.5, 1.06, note, transform=ax.transAxes, ha="center",
            fontsize=10.5, color=C_FP8, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)

# Panel 1: output throughput (higher better)
bar_panel(axes[0,0],
          [bf16["output_throughput"], fp8["output_throughput"]],
          "Output throughput", "tokens / sec", higher_is_better=True)

# Panel 2: TPOT median (lower better) — the decode / roofline metric
bar_panel(axes[0,1],
          [bf16["median_tpot_ms"], fp8["median_tpot_ms"]],
          "TPOT — time per output token (decode)", "ms / token", higher_is_better=False,
          fmt="{:.2f}")

# Panel 3: TTFT median (lower better) — prefill
bar_panel(axes[1,0],
          [bf16["median_ttft_ms"], fp8["median_ttft_ms"]],
          "TTFT — time to first token (prefill)", "ms", higher_is_better=False)

# Panel 4: model memory (lower better)
bar_panel(axes[1,1],
          [MEM_BF16_GIB, MEM_FP8_GIB],
          "Model weights in memory", "GiB", higher_is_better=False,
          fmt="{:.2f}")

fig.text(0.5, 0.01,
         "Bandwidth-bound box: FP8 halves bytes-per-weight, so decode (TPOT) gains most; "
         "prefill (TTFT) is partly compute-bound and gains less.",
         ha="center", fontsize=8.5, color="#777", style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.92])
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved chart to {out_path}")
