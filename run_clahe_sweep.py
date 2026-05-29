"""CLAHE clip sweep: keep γ=0.75, sat=1.2, unsharp=130 and vary CLAHE clip.

Variants:
  clahe_sweep_2  : CLAHE clip 2.0
  clahe_sweep_4  : CLAHE clip 4.0
  clahe_sweep_5  : CLAHE clip 5.0

For comparison, Pipe A uses clip 3.0 (MUSIQ 30.96).
"""
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageFilter

PROJ = Path(r"C:\Users\user\Desktop\NYCU\影像處理\Term Project")
IMAGES = PROJ / "Images"
RESULTS = PROJ / "results"
PYEXE = Path(r"C:\ProgramData\miniconda3\envs\deblur\python.exe")

VARIANTS = [
    ("clahe_sweep_2", 2.0),
    ("clahe_sweep_4", 4.0),
    ("clahe_sweep_5", 5.0),
]


def run(cmd, cwd=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, cwd=cwd, check=False)
    if r.returncode != 0:
        sys.exit(f"[fatal] command failed: rc={r.returncode}")


def apply_unsharp(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(in_dir.iterdir()):
        if f.suffix.lower() != ".png":
            continue
        im = Image.open(f).convert("RGB")
        im2 = im.filter(ImageFilter.UnsharpMask(radius=2, percent=130, threshold=2))
        im2.save(out_dir / (f.stem + ".png"))


for name, clip in VARIANTS:
    pre_dir  = RESULTS / f"{name}_pre"
    fft_dir  = RESULTS / f"{name}_fft"
    full_dir = RESULTS / f"{name}_full"
    print(f"\n=== {name} :: preprocess (gamma=0.75, clahe={clip}, sat=1.2) ===")
    run([str(PYEXE), str(PROJ / "preprocess.py"),
         "-i", str(IMAGES), "-o", str(pre_dir),
         "--gamma", "0.75", "--clahe-clip", str(clip), "--saturation", "1.2"])

    print(f"\n=== {name} :: FFTformer (max-side 1024) ===")
    run([str(PYEXE), "run_fftformer.py",
         "-i", str(pre_dir), "-o", str(fft_dir),
         "-w", "pretrain_model/Realblur/net_g_Realblur_J.pth",
         "--max-side", "1024"],
        cwd=str(PROJ / "FFTformer"))

    print(f"\n=== {name} :: unsharp 130 ===")
    apply_unsharp(fft_dir, full_dir)

print("\n[done] CLAHE sweep complete")
