"""Saturation-masked extra sharpening.

Reads an already-processed image (e.g. Pipe A output) and applies an
additional unsharp mask only where pixels are highly saturated (e.g. the
red Biffa lettering). Other regions are left unchanged so we don't
re-amplify night-time noise.

Usage:
    python masked_sharpen.py -i results/pipeA_full -o results/pipeA_text_sharp \
        --pct 200 --radius 3 --sat-thr 80
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter


def soft_saturation_mask(rgb_u8: np.ndarray, sat_thr: int, feather: int) -> np.ndarray:
    """Return [0,1] mask emphasising highly saturated pixels."""
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    # raw mask: 0 at sat=sat_thr, 1 at sat=255
    m = np.clip((s - sat_thr) / max(1, 255 - sat_thr), 0, 1)
    # dilate + blur a bit so edges around text get included
    k = max(1, feather)
    blurred = cv2.GaussianBlur(m, (2 * k + 1, 2 * k + 1), k / 2)
    return blurred  # (H,W) float32 in [0,1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inp", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--pct", type=int, default=200,
                    help="Unsharp percent (strength)")
    ap.add_argument("--radius", type=float, default=3.0,
                    help="Unsharp radius in pixels")
    ap.add_argument("--threshold", type=int, default=2)
    ap.add_argument("--sat-thr", type=int, default=80,
                    help="Saturation below this is considered desaturated (no extra sharpen)")
    ap.add_argument("--feather", type=int, default=15,
                    help="Mask feather radius in pixels")
    ap.add_argument("--ext", nargs="+", default=[".png", ".jpg", ".jpeg"])
    args = ap.parse_args()

    inp = Path(args.inp); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted([f for f in inp.iterdir() if f.suffix.lower() in args.ext])
    for i, f in enumerate(files, 1):
        pil = Image.open(f).convert("RGB")
        rgb = np.array(pil)
        sharp_pil = pil.filter(ImageFilter.UnsharpMask(radius=args.radius,
                                                       percent=args.pct,
                                                       threshold=args.threshold))
        sharp = np.array(sharp_pil)
        mask = soft_saturation_mask(rgb, args.sat_thr, args.feather)[..., None]
        out_arr = (sharp.astype(np.float32) * mask + rgb.astype(np.float32) * (1 - mask))
        out_arr = np.clip(out_arr, 0, 255).astype(np.uint8)
        Image.fromarray(out_arr).save(out / (f.stem + ".png"))
        print(f"[{i}/{len(files)}] {f.name}")
    print("done")


if __name__ == "__main__":
    main()
