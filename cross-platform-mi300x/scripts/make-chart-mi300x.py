#!/usr/bin/env python3
"""
Before/after chart for the FP8 vs bf16 benchmark on AMD MI300X.
Mirrors the main capstone's scripts/make-chart.py style/layout for a direct visual
comparison between the two platforms.

Reads the two `vllm bench serve` result JSONs and renders a 2x2 comparison:
output throughput, TPOT (decode), TTFT (prefill), and benchmark duration.

Usage:
    python make-chart-mi300x.py [bf16.json] [fp8.json] [out.png]
Defaults to results/mi300x-bf16.json, results/mi300x-fp8.json, graphs/mi300x-fp8-vs-bf16.png
"""
import json
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- inputs ----
baseline_path = sys.argv[1] if len(sys.argv) > 1 else "results/mi300x-bf16.json"
fp8_path      = sys.argv[2] if len(sys.argv) > 2 else "results/mi300x-fp8.json"
out_path      = sys.argv[3] if len(sys.argv) > 3 else "graphs/mi300x-fp8-vs-bf16.png"

bf16 = json.load(open(baseline_path))
fp8  = json.load(open(fp8_path))

# ---- colors (same palette as the Spark chart, for direct visual comparability) ----
C_BF16 = "#8899A6"   # muted slate — the "before"
C_FP8  = "#2E7D6B"   # teal — the "after"

# ---- figure ----
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("FP8 Dynamic vs bf16 — Qwen2.5-3B-Instruct on AMD MI300X",
             fontsize=15, fontweight="bold", y=0.98)
subtitle = (f"{bf16['num_prompts']} prompts · 1024 in / 256 out · concurrency "
            f"{bf16['max_concurrency']} · greedy · same model, only weight precision changed")
fig.text(0.5, 0.935, subtitle, ha="center", fontsize=9.5, color="#555")

labels = ["bf16", "FP8"]
colors = [C_BF16, C_FP8]

def bar_panel(ax, vals, title, unit, direction, fmt="{:.1f}"):
    """direction: 'higher_better', 'lower_better_faster', or 'lower_better_slower'
    (the last one is for cases like MI300X TTFT where FP8 got worse, not better)."""
    bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="white", linewidth=1.5)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=28)
    ax.set_ylabel(unit, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.25)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, fmt.format(v),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    if direction == "higher_better":
        factor = vals[1] / vals[0]
        pct = (vals[1] - vals[0]) / vals[0] * 100
        note = f"{factor:.2f}x ({'+' if pct >= 0 else ''}{pct:.0f}%)"
        color = C_FP8
    elif direction == "lower_better_faster":
        factor = vals[0] / vals[1]
        pct = (vals[0] - vals[1]) / vals[0] * 100
        note = f"{factor:.2f}x faster  (+{pct:.0f}%)"
        color = C_FP8
    else:  # lower_better_slower -- FP8 made it worse
        factor = vals[1] / vals[0]
        pct = (vals[1] - vals[0]) / vals[0] * 100
        note = f"{factor:.2f}x slower  (+{pct:.0f}%)"
        color = "#B3413E"  # red — flag the regression
    ax.text(0.5, 1.06, note, transform=ax.transAxes, ha="center",
            fontsize=10.5, color=color, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)

# Panel 1: output throughput (higher better) -- barely moves
bar_panel(axes[0,0],
          [bf16["output_throughput"], fp8["output_throughput"]],
          "Output throughput", "tokens / sec", direction="higher_better")

# Panel 2: TPOT median (lower better) -- decode, small gain
bar_panel(axes[0,1],
          [bf16["median_tpot_ms"], fp8["median_tpot_ms"]],
          "TPOT — time per output token (decode)", "ms / token",
          direction="lower_better_faster", fmt="{:.2f}")

# Panel 3: TTFT median -- prefill, got WORSE with FP8
bar_panel(axes[1,0],
          [bf16["median_ttft_ms"], fp8["median_ttft_ms"]],
          "TTFT — time to first token (prefill)", "ms",
          direction="lower_better_slower", fmt="{:.1f}")

# Panel 4: benchmark duration (lower better)
bar_panel(axes[1,1],
          [bf16["duration"], fp8["duration"]],
          "Benchmark duration (200 requests)", "sec",
          direction="lower_better_faster")

fig.text(0.5, 0.01,
         "Bandwidth-rich box (~5.3 TB/s): decode was never bandwidth-starved, so FP8's fewer "
         "bytes buy almost nothing — and prefill's dequant overhead actively costs TTFT.",
         ha="center", fontsize=8.5, color="#777", style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.92])
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved chart to {out_path}")
