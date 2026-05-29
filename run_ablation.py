"""Run Pipe A ablation: drop one ingredient at a time, full pipeline.

Variants:
  abl_no_gamma   : gamma=1.0 (skip),  CLAHE 3.0, sat 1.2, FFTf, unsharp 130
  abl_no_clahe   : gamma=0.75,        CLAHE 0.0 (skip), sat 1.2, FFTf, unsharp 130
  abl_no_sat     : gamma=0.75,        CLAHE 3.0, sat 1.0 (skip), FFTf, unsharp 130
  abl_no_unsharp : gamma=0.75,        CLAHE 3.0, sat 1.2, FFTf, NO unsharp

Output folders (all under results/):
  abl_no_gamma_pre  / abl_no_gamma_full
  abl_no_clahe_pre  / abl_no_clahe_full
  abl_no_sat_pre    / abl_no_sat_full
  abl_no_unsharp_pre/ abl_no_unsharp_full

Total: 4 FFTformer passes x 15 images. Expect ~25-35 min on RTX 5070 Ti.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageFilter

PROJ = Path(r"C:\Users\user\Desktop\NYCU\影像處理\Term Project")
IMAGES = PROJ / "Images"
RESULTS = PROJ / "results"
PYEXE = Path(r"C:\ProgramData\miniconda3\envs\deblur\python.exe")
FFTW = PROJ / "FFTformer" / "pretrain_model" / "Realblur" / "net_g_Realblur_J.pth"

VARIANTS = [
    # (name, gamma, clahe_clip, saturation, do_unsharp)
    ("abl_no_gamma",   1.00, 3.0, 1.2, True),
    ("abl_no_clahe",   0.75, 0.0, 1.2, True),
    ("abl_no_sat",     0.75, 3.0, 1.0, True),
    ("abl_no_unsharp", 0.75, 3.0, 1.2, False),
]


def run(cmd, cwd=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, cwd=cwd, check=False)
    if r.returncode != 0:
        sys.exit(f"[fatal] command failed: rc={r.returncode}")


def apply_unsharp(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(in_dir.iterdir()):
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        im = Image.open(f).convert("RGB")
        im2 = im.filter(ImageFilter.UnsharpMask(radius=2, percent=130, threshold=2))
        im2.save(out_dir / (f.stem + ".png"))
        print(f"  unsharp: {f.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated subset of variant names")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    for name, gamma, clip, sat, do_unsharp in VARIANTS:
        if only and name not in only:
            print(f"[skip] {name}")
            continue
        pre_dir  = RESULTS / f"{name}_pre"
        fft_dir  = RESULTS / f"{name}_fft"
        full_dir = RESULTS / f"{name}_full"

        # Step 1: preprocess
        print(f"\n=== {name} :: preprocess (gamma={gamma}, clahe={clip}, sat={sat}) ===")
        run([str(PYEXE), str(PROJ / "preprocess.py"),
             "-i", str(IMAGES), "-o", str(pre_dir),
             "--gamma", str(gamma),
             "--clahe-clip", str(clip),
             "--saturation", str(sat)])

        # Step 2: FFTformer at max-side 1024
        print(f"\n=== {name} :: FFTformer (max-side 1024) ===")
        run([str(PYEXE), "run_fftformer.py",
             "-i", str(pre_dir), "-o", str(fft_dir),
             "-w", "pretrain_model/Realblur/net_g_Realblur_J.pth",
             "--max-side", "1024"],
            cwd=str(PROJ / "FFTformer"))

        # Step 3: unsharp (or copy)
        if do_unsharp:
            print(f"\n=== {name} :: unsharp 130 ===")
            apply_unsharp(fft_dir, full_dir)
        else:
            print(f"\n=== {name} :: skip unsharp, copy FFTf output ===")
            full_dir.mkdir(parents=True, exist_ok=True)
            for f in sorted(fft_dir.iterdir()):
                if f.suffix.lower() == ".png":
                    Image.open(f).save(full_dir / f.name)

    print("\n[done] all ablation variants finished")


if __name__ == "__main__":
    main()
