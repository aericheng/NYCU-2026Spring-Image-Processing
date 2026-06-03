"""v3 per-image pipeline: Pipe A preprocess + per-image OPTIMAL resolution + gentler sharpen.

Sweep finding: optimal deblur resolution is inversely related to blur-kernel size.
  07 (mild panning blur) -> t3072 tiling  (recovers true plate/paint detail)
  08 (mild panning blur) -> t3072 tiling  (recovers Colonel engraving, QR, small text)
  09 (severe zoom blur)  -> up1024         (downscale brings huge kernel into trained range)

For each image we emit BOTH:
  old_up1024  : preprocess -> max-side 1024 -> sharpen          (current submission path)
  new_bestres : preprocess -> optimal resolution -> sharpen     (this proposal)
so the resolution change can be A/B'd at native resolution WITH preprocessing held identical.

Output: results/v3/<stem>/{old_up1024,new_bestres}.png  (native resolution)
"""
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

PROJ = Path(r"C:\Users\user\Desktop\NYCU\影像處理\Term Project")
IMAGES = PROJ / "Images"
V3 = PROJ / "results" / "v3"
PYEXE = Path(r"C:\ProgramData\miniconda3\envs\deblur\python.exe")
FFT_DIR = PROJ / "FFTformer"
WEIGHTS = "pretrain_model/Realblur/net_g_Realblur_J.pth"

# per-image config: gamma, clahe, sat, bestres-mode, unsharp percent, masked-text?
CFG = {
    "07_Yellow_Taxi_Neon_Rain_Street": dict(
        gamma=0.90, clahe=2.5, sat=1.15, longside=3072, unsharp=130, masked=False),
    "08_KFC_Rider_Rainy_Night_Delivery": dict(
        gamma=0.70, clahe=4.5, sat=1.25, longside=3072, unsharp=150, masked=False),
    "09_White_Truck_Zoom_Blur_Rain": dict(
        gamma=1.00, clahe=5.0, sat=1.20, longside=0, unsharp=180, masked=True),  # longside 0 => up1024 only
}
RAW = {k: k + ".jpg" for k in CFG}


def run(cmd, cwd=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, cwd=cwd, check=False)
    if r.returncode != 0:
        sys.exit(f"[fatal] rc={r.returncode}")


def preprocess(src_img: Path, out_dir: Path, gamma, clahe, sat):
    tin = out_dir / "_in"
    tin.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, tin / src_img.name)
    run([str(PYEXE), str(PROJ / "scripts" / "preprocess.py"), "-i", str(tin), "-o", str(out_dir),
         "--gamma", str(gamma), "--clahe-clip", str(clahe), "--saturation", str(sat)])
    # preprocess writes <stem>.png
    return out_dir / (src_img.stem + ".png")


def unsharp(in_path: Path, out_path: Path, percent: int):
    im = Image.open(in_path).convert("RGB")
    im2 = im.filter(ImageFilter.UnsharpMask(radius=2, percent=percent, threshold=2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im2.save(out_path)


def soft_sat_mask(rgb_u8, sat_thr=70, feather=15):
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    m = np.clip((s - sat_thr) / max(1, 255 - sat_thr), 0, 1)
    return cv2.GaussianBlur(m, (2 * feather + 1, 2 * feather + 1), feather / 2)


def masked_text_sharpen(in_path: Path, out_path: Path, pct=220, radius=3.0):
    pil = Image.open(in_path).convert("RGB")
    rgb = np.array(pil)
    sharp = np.array(pil.filter(ImageFilter.UnsharpMask(radius=radius, percent=pct, threshold=2)))
    mask = soft_sat_mask(rgb)[..., None]
    out = np.clip(sharp.astype(np.float32) * mask + rgb.astype(np.float32) * (1 - mask), 0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_path)


def deblur_up1024(pre_png: Path, work: Path) -> Path:
    indir = work / "in"; indir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pre_png, indir / pre_png.name)
    outdir = work / "out"
    run([str(PYEXE), "run_fftformer.py", "-i", str(indir), "-o", str(outdir),
         "-w", WEIGHTS, "--max-side", "1024"], cwd=str(FFT_DIR))
    return outdir / pre_png.name


def deblur_tiled(pre_png: Path, work: Path, longside: int, native_size) -> Path:
    # resize preprocessed image to longside, tile, then upscale back to native
    im = Image.open(pre_png).convert("RGB")
    w, h = im.size
    s = longside / max(w, h)
    if s < 1.0:
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    indir = work / "in"; indir.mkdir(parents=True, exist_ok=True)
    im.save(indir / pre_png.name)
    outdir = work / "out"
    run([str(PYEXE), "run_fftformer.py", "-i", str(indir), "-o", str(outdir),
         "-w", WEIGHTS, "--tile", "768", "--overlap", "96"], cwd=str(FFT_DIR))
    res = Image.open(outdir / pre_png.name).convert("RGB")
    if res.size != native_size:
        res = res.resize(native_size, Image.BICUBIC)
    up = work / "upscaled.png"
    res.save(up)
    return up


def finalize(deblurred: Path, out_path: Path, unsharp_pct: int, masked: bool):
    if masked:
        tmp = out_path.parent / "_u.png"
        unsharp(deblurred, tmp, unsharp_pct)
        masked_text_sharpen(tmp, out_path)
        tmp.unlink(missing_ok=True)
    else:
        unsharp(deblurred, out_path, unsharp_pct)


def main():
    for stem, c in CFG.items():
        print("\n" + "=" * 70 + f"\n{stem}\n" + "=" * 70)
        src = IMAGES / RAW[stem]
        native_size = Image.open(src).size
        base = V3 / stem
        # preprocess once
        pre = preprocess(src, base / "pre", c["gamma"], c["clahe"], c["sat"])

        # old up1024
        d_old = deblur_up1024(pre, base / "work_old")
        finalize(d_old, base / "old_up1024.png", c["unsharp"], c["masked"])
        print(f"[ok] {base / 'old_up1024.png'}")

        # new bestres
        if c["longside"] > 0:
            d_new = deblur_tiled(pre, base / "work_new", c["longside"], native_size)
            finalize(d_new, base / "new_bestres.png", c["unsharp"], c["masked"])
            print(f"[ok] {base / 'new_bestres.png'}")
        else:
            # 09: best res IS up1024; copy
            shutil.copy2(base / "old_up1024.png", base / "new_bestres.png")
            print(f"[ok] {base / 'new_bestres.png'} (== up1024, zoom blur)")

    print("\n[done] v3 per-image outputs under", V3)


if __name__ == "__main__":
    main()
