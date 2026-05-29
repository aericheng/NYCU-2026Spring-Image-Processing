"""IQA on Tier 1 improvement variants.

Targets:
  - 09 γ=1.0 + masked text sharpen (single image — score & compare to pipe_09_text_sharp)
  - Unsharp sweep: unsharp_sweep_{100,160,200}_full (15 images each)
  - CLAHE sweep: clahe_sweep_{2,4,5}_full (15 images each, ready after run_clahe_sweep.py)
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

# Reference Pipe A means (from earlier run, 15-image average)
PIPE_A_MUSIQ = 30.96
PIPE_A_NIQE  = 6.348
PIPE_A_MANIQA= 0.1920
PIPE_A_NRQM  = 4.057

# Reference 09 specific scores
SUBMISSION_09_PIPE_A_TEXT_SHARP = {"niqe": 5.924, "musiq": 28.71, "maniqa": 0.1339, "nrqm": 4.129}

METRICS = [("niqe", True), ("musiq", False), ("maniqa", False), ("nrqm", False)]
M = {}
for n, _ in METRICS:
    print(f"  loading {n} ...")
    M[n] = pyiqa.create_metric(n, device=DEVICE, as_loss=False)


def load_for_metric(p: Path, max_side: int = 2048) -> Path:
    im = Image.open(p).convert("RGB")
    W, H = im.size
    if max(W, H) <= max_side:
        return p
    s = max_side / max(W, H)
    tmp = OUT_DIR / "_tmp_tier1" / p.name
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


# ----- (1) 09 γ=1.0 + text sharpen, single image -----
print("\n=== 09 γ=1.0 + masked text sharpen ===")
p = PROJ / "results" / "pipe_09_no_gamma_text_sharp" / "09_White_Truck_Zoom_Blur_Rain.png"
if p.exists():
    pn = load_for_metric(p)
    out = {}
    for n, _ in METRICS:
        out[n] = score(M[n], pn)
    print(f"  no_gamma+text_sharp:  niqe={out['niqe']:.3f}  musiq={out['musiq']:.2f}  "
          f"maniqa={out['maniqa']:.4f}  nrqm={out['nrqm']:.3f}")
    print(f"  pipe_09_text_sharp:   niqe={SUBMISSION_09_PIPE_A_TEXT_SHARP['niqe']:.3f}  "
          f"musiq={SUBMISSION_09_PIPE_A_TEXT_SHARP['musiq']:.2f}  "
          f"maniqa={SUBMISSION_09_PIPE_A_TEXT_SHARP['maniqa']:.4f}  "
          f"nrqm={SUBMISSION_09_PIPE_A_TEXT_SHARP['nrqm']:.3f}")
    delta_musiq = out['musiq'] - SUBMISSION_09_PIPE_A_TEXT_SHARP['musiq']
    print(f"  Δ MUSIQ: {delta_musiq:+.2f}  ({'BETTER' if delta_musiq > 0 else 'WORSE'})")
else:
    print(f"  missing: {p}")


# ----- (2) Unsharp sweep means over 15 images -----
print("\n=== Unsharp percent sweep (15-image means) ===")
print(f"{'variant':<25} {'NIQE↓':>8} {'MUSIQ↑':>8} {'MANIQA↑':>9} {'NRQM↑':>8}  Δ MUSIQ vs Pipe A")
STEMS = sorted([p.stem for p in (PROJ / "Images").iterdir() if p.suffix.lower() == ".jpg"])

usweep_rows = []
for pct in (100, 130, 160, 200):
    folder = PROJ / "results" / (f"unsharp_sweep_{pct}_full" if pct != 130 else "pipeA_full")
    if not folder.exists():
        print(f"  {pct:<3}: folder missing")
        continue
    sums = {n: 0.0 for n, _ in METRICS}
    cnt = 0
    for stem in STEMS:
        p = folder / f"{stem}.png"
        if not p.exists():
            continue
        pn = load_for_metric(p)
        for n, _ in METRICS:
            sums[n] += score(M[n], pn)
        cnt += 1
    if cnt == 0:
        continue
    means = {n: sums[n] / cnt for n in sums}
    delta = means['musiq'] - PIPE_A_MUSIQ
    label = f"pipeA (130, baseline)" if pct == 130 else f"unsharp {pct}"
    print(f"  {label:<25} {means['niqe']:>8.3f} {means['musiq']:>8.2f} "
          f"{means['maniqa']:>9.4f} {means['nrqm']:>8.3f}  {delta:+.2f}")
    usweep_rows.append({"pct": pct, **means, "delta": delta})


# ----- (3) CLAHE sweep means over 15 images -----
print("\n=== CLAHE clip sweep (15-image means) ===")
print(f"{'variant':<25} {'NIQE↓':>8} {'MUSIQ↑':>8} {'MANIQA↑':>9} {'NRQM↑':>8}  Δ MUSIQ vs Pipe A")

clahe_rows = []
for clip in (2.0, 3.0, 4.0, 5.0):
    folder = PROJ / "results" / (f"clahe_sweep_{int(clip)}_full" if clip != 3.0 else "pipeA_full")
    if not folder.exists():
        print(f"  clip {clip}: folder missing (run run_clahe_sweep.py first)")
        continue
    sums = {n: 0.0 for n, _ in METRICS}
    cnt = 0
    for stem in STEMS:
        p = folder / f"{stem}.png"
        if not p.exists():
            continue
        pn = load_for_metric(p)
        for n, _ in METRICS:
            sums[n] += score(M[n], pn)
        cnt += 1
    if cnt == 0:
        continue
    means = {n: sums[n] / cnt for n in sums}
    delta = means['musiq'] - PIPE_A_MUSIQ
    label = f"clahe {clip:.0f} (baseline)" if clip == 3.0 else f"clahe {clip:.0f}"
    print(f"  {label:<25} {means['niqe']:>8.3f} {means['musiq']:>8.2f} "
          f"{means['maniqa']:>9.4f} {means['nrqm']:>8.3f}  {delta:+.2f}")
    clahe_rows.append({"clip": clip, **means, "delta": delta})


# Write CSV
csv_path = OUT_DIR / "iqa_tier1.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["variant", "niqe", "musiq", "maniqa", "nrqm", "delta_musiq"])
    for r in usweep_rows:
        label = "pipeA (unsharp 130, baseline)" if r["pct"] == 130 else f"unsharp {r['pct']}"
        w.writerow([label, r["niqe"], r["musiq"], r["maniqa"], r["nrqm"], r["delta"]])
    for r in clahe_rows:
        label = "pipeA (clahe 3, baseline)" if r["clip"] == 3.0 else f"clahe {r['clip']:.0f}"
        w.writerow([label, r["niqe"], r["musiq"], r["maniqa"], r["nrqm"], r["delta"]])
print(f"\n[ok] wrote {csv_path}")
