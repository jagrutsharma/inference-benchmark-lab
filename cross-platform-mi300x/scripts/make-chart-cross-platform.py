#!/usr/bin/env python3
"""
Combined cross-platform chart: same FP8-vs-bf16 change, both platforms side by side.
The single visual that makes the doc's thesis obvious at a glance -- one platform's
bars jump a lot under FP8, the other's barely move.

Usage:
    python make-chart-cross-platform.py [out.png]
Defaults to graphs/cross-platform-fp8-comparison.png. Numbers are hardcoded from the
already-verified results (both platforms' raw JSON), since this reads from two
different repos' result files (Spark's live in ../results/, not here).
"""
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

out_path = sys.argv[1] if len(sys.argv) > 1 else "graphs/cross-platform-fp8-comparison.png"

# Verified against results/*.json on both sides (see README tables).
platforms = ["DGX Spark\n(~273 GB/s)", "AMD MI300X\n(~5.3 TB/s)"]
throughput_bf16 = [139.7, 1080.8]
throughput_fp8  = [232.3, 1126.2]
tpot_bf16 = [27.75, 3.61]
tpot_fp8  = [16.57, 3.34]

C_BF16 = "#8899A6"
C_FP8  = "#2E7D6B"

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle("Same FP8 Quantization, Two Platforms — Qwen2.5-3B-Instruct",
             fontsize=15, fontweight="bold", y=1.0)
fig.text(0.5, 0.93,
         "Identical model, identical benchmark (200 prompts, 1024 in / 256 out, concurrency 4) — "
         "only the GPU and precision change",
         ha="center", fontsize=9.5, color="#555")

x = np.arange(len(platforms))
width = 0.32

def grouped_panel(ax, bf16_vals, fp8_vals, title, unit, higher_is_better, fmt="{:.1f}"):
    b1 = ax.bar(x - width/2, bf16_vals, width, label="bf16", color=C_BF16, edgecolor="white")
    b2 = ax.bar(x + width/2, fp8_vals, width, label="FP8", color=C_FP8, edgecolor="white")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel(unit, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(platforms, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h, fmt.format(h),
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    # annotate FP8 gain factor per platform, above each pair -- always framed as "Nx faster/higher",
    # never as a sub-1.0 ratio, to match how every other number in this doc is expressed.
    # Fixed points-offset (not a multiple of bar height) so it clears the value label even when
    # the bar is tiny relative to the shared y-axis scale.
    for i, (b, f) in enumerate(zip(bf16_vals, fp8_vals)):
        factor = (f / b) if higher_is_better else (b / f)
        label = f"{factor:.2f}x" if higher_is_better else f"{factor:.2f}x faster"
        ax.annotate(label, xy=(x[i], max(b, f)), xytext=(0, 22), textcoords="offset points",
                    ha="center", fontsize=10.5, fontweight="bold", color=C_FP8)
    ax.set_ylim(0, max(max(bf16_vals), max(fp8_vals)) * 1.3)

grouped_panel(axes[0], throughput_bf16, throughput_fp8,
              "Output throughput (higher = better)", "tokens / sec", higher_is_better=True)
grouped_panel(axes[1], tpot_bf16, tpot_fp8,
              "Median TPOT (lower = better)", "ms / token", higher_is_better=False, fmt="{:.2f}")

fig.legend(handles=[mpatches.Patch(color=C_BF16, label="bf16"), mpatches.Patch(color=C_FP8, label="FP8")],
           loc="upper center", bbox_to_anchor=(0.5, 0.87), ncol=2, frameon=False, fontsize=10.5)

fig.text(0.5, 0.01,
         "FP8 gives the bandwidth-starved Spark a 1.66x/1.67x jump; the bandwidth-rich MI300X "
         "barely moves (1.04x/1.08x) -- same optimization, opposite payoff.",
         ha="center", fontsize=9, color="#777", style="italic")

plt.tight_layout(rect=[0, 0.05, 1, 0.90])
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved chart to {out_path}")
