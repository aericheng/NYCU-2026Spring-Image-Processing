"""IQA on ablation variants. Reads results/abl_*_full/, computes 4 metrics,
writes report/iqa_ablation.csv + appends to iqa_table.md ablation section.
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

VARIANTS = [
    ("abl_no_gamma",   "− γ (no gamma)"),
    ("abl_no_clahe",   "− CLAHE"),
    ("abl_no_sat",     "− Saturation"),
    ("abl_no_unsharp", "− Unsharp"),
]
METRICS = [
    ("niqe",   True),
    ("musiq",  False),
    ("maniqa", False),
    ("nrqm",   False),
]

print(f"[info] device={DEVICE}")
M = {}
for name, _ in METRICS:
    print(f"  loading {name} ...")
    M[name] = pyiqa.create_metric(name, device=DEVICE, as_loss=False)

STEMS = sorted([p.stem for p in (PROJ / "Images").iterdir() if p.suffix.lower() == ".jpg"])


def load_for_metric(p: Path, max_side: int = 2048) -> Path:
    im = Image.open(p).convert("RGB")
    W, H = im.size
    if max(W, H) <= max_side:
        return p
    s = max_side / max(W, H)
    tmp = OUT_DIR / "_tmp_resized_abl" / p.name
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
    for vkey, _ in VARIANTS:
        folder = PROJ / "results" / f"{vkey}_full"
        p = folder / f"{stem}.png"
        if not p.exists():
            for mname, _ in METRICS:
                row[f"{mname}_{vkey}"] = float("nan")
            continue
        pn = load_for_metric(p, 2048)
        parts = []
        for mname, _ in METRICS:
            v = score(M[mname], pn)
            row[f"{mname}_{vkey}"] = v
            parts.append(f"{mname}={v:.3f}")
        print(f"  {stem:<48} {vkey:<16}  " + "  ".join(parts))
    rows.append(row)


def mean_of(key):
    vals = [r[key] for r in rows if key in r and isinstance(r.get(key), float) and r[key] == r[key]]
    return sum(vals) / len(vals) if vals else float("nan")


csv_path = OUT_DIR / "iqa_ablation.csv"
fieldnames = ["stem"] + [f"{m}_{v}" for m, _ in METRICS for v, _ in VARIANTS]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fieldnames})
print(f"[ok] wrote {csv_path}")


# Print summary table
PIPE_A_REFS = {"niqe": 6.348, "musiq": 30.96, "maniqa": 0.1920, "nrqm": 4.057}
print()
print("=" * 100)
print(f"{'Variant':<25}  {'NIQE↓':>10} {'MUSIQ↑':>10} {'MANIQA↑':>10} {'NRQM↑':>10}  vs Pipe A MUSIQ")
print("-" * 100)
print(f"{'Pipe A (full reference)':<25}  "
      f"{PIPE_A_REFS['niqe']:>10.3f} {PIPE_A_REFS['musiq']:>10.2f} "
      f"{PIPE_A_REFS['maniqa']:>10.4f} {PIPE_A_REFS['nrqm']:>10.3f}     —")
for vkey, vlabel in VARIANTS:
    m_musiq = mean_of(f"musiq_{vkey}")
    delta = m_musiq - PIPE_A_REFS["musiq"]
    print(f"{vlabel:<25}  "
          f"{mean_of(f'niqe_{vkey}'):>10.3f} {m_musiq:>10.2f} "
          f"{mean_of(f'maniqa_{vkey}'):>10.4f} {mean_of(f'nrqm_{vkey}'):>10.3f}    {delta:+.2f}")

# Append to iqa_table.md
md_path = OUT_DIR / "iqa_table.md"
ablation_md = "\n\n---\n\n## Ingredient Ablation (each removes one component from Pipe A)\n\n"
ablation_md += "Means over 15 images:\n\n"
ablation_md += "| Variant | NIQE↓ | MUSIQ↑ | MANIQA↑ | NRQM↑ | Δ MUSIQ |\n"
ablation_md += "|---|---|---|---|---|---|\n"
ablation_md += f"| **Pipe A (full)** | 6.348 | **30.96** | 0.1920 | 4.057 | — |\n"
for vkey, vlabel in VARIANTS:
    m_niqe   = mean_of(f"niqe_{vkey}")
    m_musiq  = mean_of(f"musiq_{vkey}")
    m_maniqa = mean_of(f"maniqa_{vkey}")
    m_nrqm   = mean_of(f"nrqm_{vkey}")
    delta = m_musiq - PIPE_A_REFS["musiq"]
    ablation_md += (f"| {vlabel} | {m_niqe:.3f} | {m_musiq:.2f} | {m_maniqa:.4f} | "
                    f"{m_nrqm:.3f} | {delta:+.2f} |\n")
ablation_md += "\n**解讀**：每一個變體相對於 Pipe A 完整版的 MUSIQ 差距，反映該成分的貢獻——數值越負，代表該成分越重要。\n"

with open(md_path, "a", encoding="utf-8") as f:
    f.write(ablation_md)
print(f"\n[ok] appended ablation section to {md_path}")
