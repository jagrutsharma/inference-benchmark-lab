#!/usr/bin/env python3
"""
A6 — prefix caching before/after chart.

Prefix caching is a PREFILL optimization, so the story is TTFT (not TPOT/throughput).
Left panel: TTFT at p50/p95/p99, cache off vs on -- shows the median collapse AND the
tall p99 "on" bar (the cold-miss requests). Right panel: TPOT, to show decode is untouched.

Usage: python make-chart-prefix.py [off.json] [on.json] [out.png]
"""
import json, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

off_path = sys.argv[1] if len(sys.argv) > 1 else "results/prefix-cache-off.json"
on_path  = sys.argv[2] if len(sys.argv) > 2 else "results/prefix-cache-on.json"
out_path = sys.argv[3] if len(sys.argv) > 3 else "charts/prefix-caching-ttft.png"

off = json.load(open(off_path))
on  = json.load(open(on_path))

C_OFF = "#8899A6"   # slate — cache off
C_ON  = "#2E7D6B"   # teal  — cache on

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2),
                               gridspec_kw={"width_ratios": [2.3, 1]})
fig.suptitle("Prefix Caching — Qwen2.5-3B FP8 on DGX Spark (GB10)",
             fontsize=14, fontweight="bold", y=0.99)
fig.text(0.5, 0.925,
         "200 prompts · 5 shared prefixes (2048 tok) + 128-tok unique suffix · concurrency 4",
         ha="center", fontsize=9, color="#555")

# ---- Left: TTFT percentiles ----
pcts = ["p50", "p95", "p99"]
off_v = [off["p50_ttft_ms"], off["p95_ttft_ms"], off["p99_ttft_ms"]]
on_v  = [on["p50_ttft_ms"],  on["p95_ttft_ms"],  on["p99_ttft_ms"]]
x = np.arange(len(pcts)); w = 0.38

b1 = ax1.bar(x - w/2, off_v, w, label="cache OFF", color=C_OFF, edgecolor="white")
b2 = ax1.bar(x + w/2, on_v,  w, label="cache ON",  color=C_ON,  edgecolor="white")
ax1.set_title("Time to first token (prefill)", fontsize=12, fontweight="bold", pad=10)
ax1.set_ylabel("TTFT (ms)"); ax1.set_xticks(x); ax1.set_xticklabels(pcts, fontsize=11)
ax1.set_ylim(0, max(off_v) * 1.18)
for bars in (b1, b2):
    for b in bars:
        ax1.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.0f}",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax1.legend(frameon=False, fontsize=10, loc="upper left")
ax1.spines[["top","right"]].set_visible(False)
# median speedup callout
sp = off["p50_ttft_ms"]/on["p50_ttft_ms"]
ax1.text(0.145, 0.72, f"median TTFT\n{sp:.1f}x faster", transform=ax1.transAxes,
         ha="center", fontsize=11, fontweight="bold", color=C_ON, linespacing=1.3)
# annotate the p99-on = cold misses
ax1.annotate("p99 'on' = cold-miss\nrequests (1 per prefix)",
             xy=(2+w/2, on_v[2]), xytext=(1.55, on_v[2]+150),
             fontsize=8, color="#666", ha="center",
             arrowprops=dict(arrowstyle="->", color="#999", lw=1))

# ---- Right: TPOT (the non-effect) ----
tp = [off["median_tpot_ms"], on["median_tpot_ms"]]
bars = ax2.bar(["OFF","ON"], tp, color=[C_OFF, C_ON], width=0.55, edgecolor="white")
ax2.set_title("TPOT (decode)\n— unchanged", fontsize=11, fontweight="bold", pad=10)
ax2.set_ylabel("ms / token"); ax2.set_ylim(0, max(tp)*1.3)
for b in bars:
    ax2.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.2f}",
             ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.spines[["top","right"]].set_visible(False)

fig.text(0.5, 0.005,
         "Prefix caching reuses the shared prefix's KV cache -> skips ~94% of prefill on hits. "
         "It attacks prefill (TTFT), not decode (TPOT). Speedup capped by irreducible per-request overhead.",
         ha="center", fontsize=8, color="#777", style="italic")

plt.tight_layout(rect=[0, 0.04, 1, 0.90])
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved chart to {out_path}")

