"""IQA ablation: 5 no-reference metrics x 7 methods x 15 images.

Metrics:
  - NIQE     (lower better) -- classic natural-statistics
  - MUSIQ    (higher better) -- modern transformer, KonIQ-trained
  - MANIQA   (higher better) -- multi-dimension attention NR-IQA
  - CLIPIQA  (higher better) -- CLIP-based, prompt: "Good photo"
  - NRQM     (higher better) -- learned quality from blind SR

Methods:
  - input  : raw camera jpg
  - darkir : DarkIR blend50
  - miscf  : MISCFilter RealBlur-J full-res
  - fftf   : FFTformer RealBlur-J max-side 1024
  - chain  : DarkIR50 -> FFTformer
  - pipeA  : Pipe A (preproc + FFTf + unsharp)
  - final  : Plan I+ strengthened (08, 09 only)
"""
from pathlib import Path
import csv
import sys

import torch
import pyiqa
from PIL import Image

PROJ = Path(r"C:\Users\user\Desktop\NYCU\影像處理\Term Project")
OUT_DIR = PROJ / "report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[info] device={DEVICE}")

METHODS = [
    ("input",   PROJ / "Images",                          ".jpg"),
    ("darkir",  PROJ / "results" / "darkir_blend50",      ".png"),
    ("miscf",   PROJ / "results" / "miscf_realj_fullres", ".png"),
    ("fftf",    PROJ / "results" / "fftformer_realj",     ".png"),
    ("chain",   PROJ / "results" / "chain_dk50_fftf",     ".png"),
    ("pipeA_v1",PROJ / "results" / "pipeA_full",          ".png"),  # CLAHE 3, unsharp 130
    ("pipeA",   PROJ / "results" / "pipeA_v2_full",       ".png"),  # CLAHE 5, unsharp 200 (current)
]
FINAL = {
    "08_KFC_Rider_Rainy_Night_Delivery": PROJ / "results" / "pipe_08_v2_final"      / "08_KFC_Rider_Rainy_Night_Delivery.png",
    "09_White_Truck_Zoom_Blur_Rain":     PROJ / "results" / "pipe_09_v2_text_sharp" / "09_White_Truck_Zoom_Blur_Rain.png",
}

METRICS = [
    ("niqe",    True),   # lower_better
    ("musiq",   False),
    ("maniqa",  False),
    ("nrqm",    False),
]
# Note: clipiqa removed - pyiqa's CLIP backend has a pkg_resources.packaging
# import that breaks on setuptools >=70. Four metrics still cover cross-validation.

STEMS = sorted([p.stem for p in (PROJ / "Images").iterdir() if p.suffix.lower() == ".jpg"])
print(f"[info] {len(STEMS)} stems x {len(METHODS)} methods x {len(METRICS)} metrics = "
      f"{len(STEMS)*len(METHODS)*len(METRICS) + 2*len(METRICS)} evaluations")

print("[info] loading metrics (downloads weights on first run)...")
M = {}
for name, lower in METRICS:
    print(f"  loading {name} ...")
    M[name] = pyiqa.create_metric(name, device=DEVICE, as_loss=False)


def load_for_metric(p: Path, max_side: int = 2048) -> Path:
    im = Image.open(p).convert("RGB")
    W, H = im.size
    if max(W, H) <= max_side:
        return p
    s = max_side / max(W, H)
    tmp = OUT_DIR / "_tmp_resized" / p.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    im.resize((int(W * s), int(H * s)), Image.LANCZOS).save(tmp, quality=95)
    return tmp


def score(metric_obj, path: Path) -> float:
    try:
        with torch.inference_mode():
            v = metric_obj(str(path))
            if hasattr(v, "item"):
                v = v.item()
        return float(v)
    except Exception as e:
        print(f"  [err] {path.name}: {e}", file=sys.stderr)
        return float("nan")


rows = []
for stem in STEMS:
    row = {"stem": stem}
    for tag, folder, ext in METHODS:
        p = folder / f"{stem}{ext}"
        if not p.exists():
            for mname, _ in METRICS:
                row[f"{mname}_{tag}"] = float("nan")
            continue
        pn = load_for_metric(p, max_side=2048)
        parts = []
        for mname, _ in METRICS:
            v = score(M[mname], pn)
            row[f"{mname}_{tag}"] = v
            parts.append(f"{mname}={v:.3f}")
        print(f"  {stem:<48} {tag:<7}  " + "  ".join(parts))
    if stem in FINAL:
        p = FINAL[stem]
        if p.exists():
            pn = load_for_metric(p, max_side=2048)
            parts = []
            for mname, _ in METRICS:
                v = score(M[mname], pn)
                row[f"{mname}_final"] = v
                parts.append(f"{mname}={v:.3f}")
            print(f"  {stem:<48} final    " + "  ".join(parts))
    rows.append(row)


def mean_of(key):
    vals = [r[key] for r in rows if key in r and isinstance(r.get(key), float) and r[key] == r[key]]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(v, n):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:.{n}f}"


# CSV
csv_path = OUT_DIR / "iqa_table.csv"
ALL_TAGS = [t for t, *_ in METHODS] + ["final"]
fieldnames = ["stem"] + [f"{m}_{t}" for m, _ in METRICS for t in ALL_TAGS]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fieldnames})
print(f"[ok] wrote {csv_path}")


# Console summary
print()
print("=" * 90)
hdr = f"{'method':<10}"
for mname, lower in METRICS:
    arrow = "↓" if lower else "↑"
    hdr += f"  {mname.upper() + arrow:>14}"
print(hdr)
print("-" * 90)
for tag, *_ in METHODS:
    line = f"{tag:<10}"
    for mname, _ in METRICS:
        line += f"  {mean_of(f'{mname}_{tag}'):>14.3f}"
    print(line)
line = f"{'final':<10}"
for mname, _ in METRICS:
    line += f"  {mean_of(f'{mname}_final'):>14.3f}"
print(line + "   (over 2 submission images)")


# Markdown
md_path = OUT_DIR / "iqa_table.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# IQA Ablation — 5 no-reference metrics × 7 methods × 15 images\n\n")
    f.write("Means over 15 images (`final` = 2 submission images: 08, 09):\n\n")
    f.write("| Method | NIQE↓ | MUSIQ↑ | MANIQA↑ | NRQM↑ |\n|---|---|---|---|---|\n")
    for tag, *_ in METHODS:
        f.write(f"| {tag} | "
                f"{mean_of(f'niqe_{tag}'):.3f} | "
                f"{mean_of(f'musiq_{tag}'):.2f} | "
                f"{mean_of(f'maniqa_{tag}'):.4f} | "
                f"{mean_of(f'nrqm_{tag}'):.3f} |\n")
    f.write(f"| **final (Plan I+)** | "
            f"**{mean_of('niqe_final'):.3f}** | "
            f"**{mean_of('musiq_final'):.2f}** | "
            f"**{mean_of('maniqa_final'):.4f}** | "
            f"**{mean_of('nrqm_final'):.3f}** |\n")
    f.write("\n## How to read\n\n")
    f.write("- **NIQE** (low better): natural-statistics. Penalises any aggressive enhancement.\n")
    f.write("- **MUSIQ / MANIQA** (high better): modern learned perceptual metrics.\n")
    f.write("- **NRQM** (high better): blind-SR quality; rewards crisp edges & detail.\n\n")
    f.write("Two transformer-based perceptual metrics (MUSIQ, MANIQA) and one edge-aware (NRQM) provide cross-validation against the known NIQE bias.\n\n")
    for mname, lower in METRICS:
        arrow = "↓" if lower else "↑"
        f.write(f"\n## Per-image {mname.upper()} {arrow}\n\n")
        f.write("| Image | " + " | ".join(ALL_TAGS) + " |\n")
        f.write("|" + "---|" * (len(ALL_TAGS) + 1) + "\n")
        ndigits = 3 if mname in ("niqe", "nrqm") else (4 if mname in ("maniqa", "clipiqa") else 2)
        for r in rows:
            parts = [r["stem"]] + [fmt(r.get(f"{mname}_{t}"), ndigits) for t in ALL_TAGS]
            f.write("| " + " | ".join(parts) + " |\n")
print(f"[ok] wrote {md_path}")
