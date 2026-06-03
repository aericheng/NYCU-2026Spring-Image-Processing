"""Pipe A v2 = γ=0.75 + CLAHE 5.0 + sat 1.2 + FFTformer (max-side 1024) + unsharp 200.

Plus Plan I+ v2 submission variants:
  09 v2: γ=1.0 + CLAHE 5.0 + sat 1.2 + FFTf + unsharp 200 + masked text sharpen
  08 v2: γ=0.70 + CLAHE 5.0 + sat 1.25 + FFTf + unsharp 200

Outputs:
  results/pipeA_v2_full/             — Pipe A v2 baseline on all 15 images
  results/pipe_09_v2_text_sharp/     — final 09 submission (v2)
  results/pipe_08_v2_final/          — final 08 submission (v2)
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

PROJ = Path(r"C:\Users\user\Desktop\NYCU\影像處理\Term Project")
IMAGES = PROJ / "Images"
RESULTS = PROJ / "results"
PYEXE = Path(r"C:\ProgramData\miniconda3\envs\deblur\python.exe")


def run(cmd, cwd=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, cwd=cwd, check=False)
    if r.returncode != 0:
        sys.exit(f"[fatal] command failed: rc={r.returncode}")


def apply_unsharp(in_dir: Path, out_dir: Path, percent: int = 200, radius: float = 2.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(in_dir.iterdir()):
        if f.suffix.lower() != ".png":
            continue
        im = Image.open(f).convert("RGB")
        im2 = im.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=2))
        im2.save(out_dir / (f.stem + ".png"))


def soft_saturation_mask(rgb_u8, sat_thr=70, feather=15):
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    m = np.clip((s - sat_thr) / max(1, 255 - sat_thr), 0, 1)
    k = feather
    return cv2.GaussianBlur(m, (2 * k + 1, 2 * k + 1), k / 2)


def apply_masked_text_sharpen(in_path: Path, out_path: Path,
                              pct: int = 220, radius: float = 3.0,
                              sat_thr: int = 70, feather: int = 15):
    pil = Image.open(in_path).convert("RGB")
    rgb = np.array(pil)
    sharp_pil = pil.filter(ImageFilter.UnsharpMask(radius=radius, percent=pct, threshold=2))
    sharp = np.array(sharp_pil)
    mask = soft_saturation_mask(rgb, sat_thr=sat_thr, feather=feather)[..., None]
    out_arr = sharp.astype(np.float32) * mask + rgb.astype(np.float32) * (1 - mask)
    out_arr = np.clip(out_arr, 0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out_arr).save(out_path)


# ============================================================
# Pipeline 1: Pipe A v2 baseline (15 images)
# ============================================================
print("\n" + "=" * 70)
print("Pipeline 1: Pipe A v2 baseline (γ=0.75, CLAHE=5, sat=1.2, unsharp=200)")
print("=" * 70)
pre_dir  = RESULTS / "pipeA_v2_pre"
fft_dir  = RESULTS / "pipeA_v2_fft"
full_dir = RESULTS / "pipeA_v2_full"

run([str(PYEXE), str(PROJ / "scripts" / "preprocess.py"),
     "-i", str(IMAGES), "-o", str(pre_dir),
     "--gamma", "0.75", "--clahe-clip", "5.0", "--saturation", "1.2"])

run([str(PYEXE), "run_fftformer.py",
     "-i", str(pre_dir), "-o", str(fft_dir),
     "-w", "pretrain_model/Realblur/net_g_Realblur_J.pth",
     "--max-side", "1024"],
    cwd=str(PROJ / "FFTformer"))

apply_unsharp(fft_dir, full_dir, percent=200)
print(f"[ok] Pipe A v2 full pipeline -> {full_dir}")


# ============================================================
# Pipeline 2: 09 v2 (γ=1.0 instead of 0.75)
# ============================================================
print("\n" + "=" * 70)
print("Pipeline 2: 09 v2 (γ=1.0, CLAHE=5, sat=1.2, unsharp=200, masked text sharpen)")
print("=" * 70)
# Use single-image temp folders
TMP_09 = PROJ / "_tmp_09_v2"
tmp_inp = TMP_09 / "input"
tmp_inp.mkdir(parents=True, exist_ok=True)
src = IMAGES / "09_White_Truck_Zoom_Blur_Rain.jpg"
import shutil
shutil.copy2(src, tmp_inp / src.name)

tmp_pre = TMP_09 / "pre"
tmp_fft = TMP_09 / "fft"
tmp_full = TMP_09 / "full"

run([str(PYEXE), str(PROJ / "scripts" / "preprocess.py"),
     "-i", str(tmp_inp), "-o", str(tmp_pre),
     "--gamma", "1.0", "--clahe-clip", "5.0", "--saturation", "1.2"])

run([str(PYEXE), "run_fftformer.py",
     "-i", str(tmp_pre), "-o", str(tmp_fft),
     "-w", "pretrain_model/Realblur/net_g_Realblur_J.pth",
     "--max-side", "1024"],
    cwd=str(PROJ / "FFTformer"))

apply_unsharp(tmp_fft, tmp_full, percent=200)

# Now apply masked text sharpen
out_09 = RESULTS / "pipe_09_v2_text_sharp" / "09_White_Truck_Zoom_Blur_Rain.png"
apply_masked_text_sharpen(tmp_full / "09_White_Truck_Zoom_Blur_Rain.png", out_09)
print(f"[ok] 09 v2 submission -> {out_09}")


# ============================================================
# Pipeline 3: 08 v2 (γ=0.70, CLAHE=5, sat=1.25)
# ============================================================
print("\n" + "=" * 70)
print("Pipeline 3: 08 v2 (γ=0.70, CLAHE=5, sat=1.25, unsharp=200)")
print("=" * 70)
TMP_08 = PROJ / "_tmp_08_v2"
tmp_inp = TMP_08 / "input"
tmp_inp.mkdir(parents=True, exist_ok=True)
src = IMAGES / "08_KFC_Rider_Rainy_Night_Delivery.jpg"
shutil.copy2(src, tmp_inp / src.name)

tmp_pre = TMP_08 / "pre"
tmp_fft = TMP_08 / "fft"

run([str(PYEXE), str(PROJ / "scripts" / "preprocess.py"),
     "-i", str(tmp_inp), "-o", str(tmp_pre),
     "--gamma", "0.70", "--clahe-clip", "5.0", "--saturation", "1.25"])

run([str(PYEXE), "run_fftformer.py",
     "-i", str(tmp_pre), "-o", str(tmp_fft),
     "-w", "pretrain_model/Realblur/net_g_Realblur_J.pth",
     "--max-side", "1024"],
    cwd=str(PROJ / "FFTformer"))

out_dir_08 = RESULTS / "pipe_08_v2_final"
apply_unsharp(tmp_fft, out_dir_08, percent=200)
print(f"[ok] 08 v2 submission -> {out_dir_08}")

# Cleanup temp
print("\n[cleanup] removing temp folders")
shutil.rmtree(TMP_09, ignore_errors=True)
shutil.rmtree(TMP_08, ignore_errors=True)
print("\n[done] Pipe A v2 + Plan I+ v2 complete")
