"""Pipe A pipeline diagram for PPT Slide 4.

Vertical flowchart, 3 colour zones:
  orange = preprocessing
  blue   = deep model
  grey   = post-processing
Right-side annotations explain the 'why' of each step.
Output: report/pipe_a_diagram.png  (1200x1500, ready for PowerPoint)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

OUT = Path(r"C:\Users\user\Desktop\NYCU\影像處理\Term Project\report\figures") / "fig_3_1_pipeline.png"

# Find a CJK font (Microsoft JhengHei / Microsoft YaHei) for Chinese labels
CJK_FONTS = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "PMingLiU", "MS Gothic"]
available = {f.name for f in font_manager.fontManager.ttflist}
font_name = next((f for f in CJK_FONTS if f in available), "DejaVu Sans")
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False
print(f"[info] using font: {font_name}")


COLOR_PRE   = "#F4B860"  # orange — preprocessing
COLOR_MODEL = "#4C72B0"  # blue   — model
COLOR_POST  = "#9E9E9E"  # grey   — post

STEPS = [
    # (label, sublabel, color, why)
    ("Input",                  "夜間原圖 (raw night-blur)",    "white",     None),
    ("Gamma 校正",             "γ = 0.75 (per-image)",         COLOR_PRE,   "拉亮暗區（可條件性套用）"),
    ("CLAHE 局部對比",         "clip = 5.0, grid = 8×8 (LAB-L)", COLOR_PRE,   "核心成分：紋理/文字邊緣浮現"),
    ("Saturation 增強",        "scale × 1.2 (HSV-S)",          COLOR_PRE,   "視覺色彩增強"),
    ("Resize",                 "max-side = 1024",              COLOR_POST,  "對齊 FFTformer 訓練解析度"),
    ("FFTformer",              "RealBlur-J 預訓練",             COLOR_MODEL, "FFT-domain attention，主力去模糊"),
    ("Resize back",            "回原始解析度 (Lanczos)",        COLOR_POST,  "回到 6K~8K native"),
    ("Unsharp mask",           "r = 2, pct = 200, thr = 2",    COLOR_POST,  "核心成分：補回高頻細節"),
    ("Output",                 "去模糊結果",                    "white",     None),
]

fig, ax = plt.subplots(figsize=(13, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, len(STEPS) * 1.25 + 1.8)
ax.axis("off")

# Title
ax.text(5.0, len(STEPS) * 1.25 + 1.4,
        "Pipe A v2 Pipeline",
        ha="center", va="top", fontsize=30, fontweight="bold")
ax.text(5.0, len(STEPS) * 1.25 + 0.65,
        "通用前處理 + FFTformer + 後處理（CLAHE 5 + unsharp 200，全 4 指標領先）",
        ha="center", va="top", fontsize=18, color="#555")

# Draw boxes from top to bottom
n = len(STEPS)
positions = []
for i, (label, sub, color, why) in enumerate(STEPS):
    y_center = (n - 1 - i) * 1.25 + 0.6
    positions.append(y_center)

    is_endpoint = color == "white"
    width = 4.6 if not is_endpoint else 3.8
    height = 1.0
    x_left = 2.3 if not is_endpoint else 2.7

    box = FancyBboxPatch(
        (x_left, y_center - height / 2),
        width, height,
        boxstyle="round,pad=0.05,rounding_size=0.14",
        linewidth=2.0,
        edgecolor="black" if not is_endpoint else "#333",
        facecolor=color if not is_endpoint else "#F0F0F0",
        zorder=2,
    )
    ax.add_patch(box)

    text_color = "white" if color in (COLOR_MODEL,) else "black"
    fontweight = "bold" if not is_endpoint else "normal"
    ax.text(x_left + width / 2, y_center + 0.18,
            label, ha="center", va="center",
            fontsize=24 if not is_endpoint else 20,
            fontweight=fontweight,
            color=text_color)
    ax.text(x_left + width / 2, y_center - 0.24,
            sub, ha="center", va="center",
            fontsize=18, color=text_color,
            style="italic" if not is_endpoint else "normal")

    # Why annotation on the right
    if why:
        ax.text(x_left + width + 0.3, y_center,
                "← " + why,
                ha="left", va="center", fontsize=17, color="#222")

# Arrows between boxes
for i in range(n - 1):
    y_from = positions[i] - 0.52
    y_to = positions[i + 1] + 0.52
    arrow = FancyArrowPatch(
        (4.6, y_from), (4.6, y_to),
        arrowstyle="-|>", mutation_scale=26,
        color="#333", linewidth=2.0, zorder=1,
    )
    ax.add_patch(arrow)

# Zone labels on the left
def zone_label(y_top, y_bot, text, color):
    ax.plot([1.8, 1.8], [y_bot, y_top], color=color, linewidth=7, zorder=0)
    ax.text(1.5, (y_top + y_bot) / 2, text,
            ha="right", va="center",
            rotation=90, fontsize=20, fontweight="bold",
            color=color)

# preprocessing zone: steps 1..3 (gamma, clahe, sat)
zone_label(positions[1] + 0.5, positions[3] - 0.5, "前處理 Pre", COLOR_PRE)
# model zone: step 5 (FFTformer)
zone_label(positions[5] + 0.5, positions[5] - 0.5, "模型 Model", COLOR_MODEL)
# post zone: steps 6 + 7 (resize back + unsharp)
zone_label(positions[6] + 0.5, positions[7] - 0.5, "後處理 Post", COLOR_POST)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
print(f"[ok] wrote {OUT}")
plt.close()
