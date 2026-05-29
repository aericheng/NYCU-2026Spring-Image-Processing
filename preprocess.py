"""Pre-processing for night-blur images before deblurring.

Options (in order of application):
  1. gamma correction (<1 lifts shadows, >1 darkens)
  2. CLAHE on the L channel of LAB (strong local contrast)
  3. saturation multiplier in HSV
  4. unsharp mask pre-sharpen

Usage:
    python preprocess.py -i Images -o results/pre_clahe_gamma \
        --gamma 0.85 --clahe-clip 3.0 --clahe-grid 8 --saturation 1.15
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter


def apply_gamma(img_u8: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-3:
        return img_u8
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img_u8, table)


def apply_clahe(img_u8_rgb: np.ndarray, clip: float, grid: int) -> np.ndarray:
    if clip <= 0:
        return img_u8_rgb
    lab = cv2.cvtColor(img_u8_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def apply_saturation(img_u8_rgb: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-3:
        return img_u8_rgb
    hsv = cv2.cvtColor(img_u8_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def apply_unsharp(img_pil: Image.Image, radius: float, percent: int, thr: int) -> Image.Image:
    if percent <= 0:
        return img_pil
    return img_pil.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=thr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inp", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--clahe-clip", type=float, default=0.0,
                    help="0 disables CLAHE; typical useful range 1.5-4.0")
    ap.add_argument("--clahe-grid", type=int, default=8)
    ap.add_argument("--saturation", type=float, default=1.0)
    ap.add_argument("--unsharp-pct", type=int, default=0,
                    help="0 disables; >0 applies pre-unsharp")
    ap.add_argument("--unsharp-radius", type=float, default=1.5)
    ap.add_argument("--unsharp-thr", type=int, default=2)
    args = ap.parse_args()

    inp = Path(args.inp); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted([f for f in inp.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"gamma={args.gamma}, clahe={args.clahe_clip}, sat={args.saturation}, unsharp={args.unsharp_pct}")
    for i, f in enumerate(files, 1):
        pil = Image.open(f).convert("RGB")
        arr = np.array(pil)
        arr = apply_gamma(arr, args.gamma)
        arr = apply_clahe(arr, args.clahe_clip, args.clahe_grid)
        arr = apply_saturation(arr, args.saturation)
        pil2 = Image.fromarray(arr)
        pil2 = apply_unsharp(pil2, args.unsharp_radius, args.unsharp_pct, args.unsharp_thr)
        pil2.save(out / (f.stem + ".png"))
        print(f"[{i}/{len(files)}] {f.name}")
    print("done")


if __name__ == "__main__":
    main()
