"""Bar chart visualisation of IQA ablation, reading report/iqa_table.csv.

Produces two figures:
  report/iqa_bar.png       — overall mean per method, for each metric
  report/iqa_bar_08_09.png — per-image scores for the 2 submission images

Usage:
    python plot_iqa.py
"""
from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = Path(r"C:\Users\user\Desktop\NYCU\影像處理\Term Project")
OUT_DIR = PROJ / "report"
CSV = OUT_DIR / "iqa_table.csv"

METHODS = ["input", "darkir", "miscf", "fftf", "chain", "pipeA_v1", "pipeA", "final"]
METHOD_LABELS = {
    "input":    "Input",
    "darkir":   "DarkIR",
    "miscf":    "MISCFilter",
    "fftf":     "FFTformer",
    "chain":    "Chain (DK→FFTf)",
    "pipeA_v1": "Pipe A v1",
    "pipeA":    "Pipe A v2",
    "final":    "Plan I+ v2 (08,09)",
}
METRICS = [
    ("niqe",    "NIQE",    True,  3),
    ("musiq",   "MUSIQ",   False, 2),
    ("maniqa",  "MANIQA",  False, 4),
    ("nrqm",    "NRQM",    False, 3),
]
HIGHLIGHT = {"pipeA", "final"}  # v2 + Plan I+ v2 in red

def safe_float(x):
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except (ValueError, TypeError):
        return None

rows = []
with open(CSV, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append(r)
print(f"[info] {len(rows)} rows from {CSV}")


def mean_col(col):
    vals = [safe_float(r.get(col, "")) for r in rows]
    vals = [v for v in vals if v is not None]
    return float("nan") if not vals else sum(vals) / len(vals)


# Figure 1: 5 subplots (one per metric), each a bar chart of method means
fig, axes = plt.subplots(1, 4, figsize=(20, 5.0))
for ax, (mkey, mlabel, lower_better, _) in zip(axes, METRICS):
    means = []
    for tag in METHODS:
        means.append(mean_col(f"{mkey}_{tag}"))
    colors = ["#4C72B0" if t not in HIGHLIGHT else "#C44E52" for t in METHODS]
    xs = np.arange(len(METHODS))
    bars = ax.bar(xs, means, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABELS[t] for t in METHODS], rotation=30, ha="right", fontsize=9)
    arrow = "↓ lower better" if lower_better else "↑ higher better"
    ax.set_title(f"{mlabel} ({arrow})", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    if lower_better:
        best_idx = int(np.nanargmin(means))
    else:
        best_idx = int(np.nanargmax(means))
    ax.annotate("best", xy=(best_idx, means[best_idx]),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8, color="#C44E52", fontweight="bold")
    yl = ax.get_ylim()
    ax.set_ylim(yl[0], yl[1] * 1.12)
fig.suptitle("IQA Ablation — mean over 15 images (Plan I+ uses 08, 09)", fontsize=13, y=1.02)
fig.tight_layout()
out1 = OUT_DIR / "figures" / "fig_4_1_iqa_bar.png"
out1.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out1, dpi=150, bbox_inches="tight")
print(f"[ok] wrote {out1}")
plt.close(fig)


# Figure 2: per-image scores for 08 and 09 only (the 2 submission images)
TARGET_STEMS = ["08_KFC_Rider_Rainy_Night_Delivery", "09_White_Truck_Zoom_Blur_Rain"]
target_rows = [r for r in rows if r["stem"] in TARGET_STEMS]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ri, r in enumerate(target_rows):
    short = r["stem"].split("_", 1)[0] + " " + r["stem"].split("_", 1)[1].split("_")[0]
    for ci, (mkey, mlabel, lower_better, _) in enumerate(METRICS):
        ax = axes[ri][ci]
        vals = [safe_float(r.get(f"{mkey}_{t}", "")) for t in METHODS]
        colors = ["#4C72B0" if t not in HIGHLIGHT else "#C44E52" for t in METHODS]
        xs = np.arange(len(METHODS))
        ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([METHOD_LABELS[t] for t in METHODS], rotation=30, ha="right", fontsize=8)
        arrow = "↓" if lower_better else "↑"
        ax.set_title(f"{r['stem'][:2]} — {mlabel} {arrow}", fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
fig.suptitle("IQA per submission image (08, 09)", fontsize=13, y=1.01)
fig.tight_layout()
out2 = OUT_DIR / "figures" / "fig_4_2_iqa_bar_08_09.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
print(f"[ok] wrote {out2}")
plt.close(fig)
