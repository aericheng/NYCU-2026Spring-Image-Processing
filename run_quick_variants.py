"""Quick variants that reuse existing FFTformer outputs (no GPU passes needed).

1. 09 γ=1.0 + masked text sharpen
   - Source: results/abl_no_gamma_full/09_*.png (already has γ=1.0 + CLAHE + sat + unsharp)
   - Apply masked text sharpen on top
   - Output: results/pipe_09_no_gamma_text_sharp/

2. Unsharp percent sweep on Pipe A pre-unsharp output
   - Source: results/abl_no_unsharp_fft/ (Pipe A preproc + FFTf, no unsharp yet)
   - Variants: percent=100, 160, 200 (current Pipe A uses 130)
   - Output: results/unsharp_sweep_<N>_full/
"""
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageFilter

PROJ = Path(r"C:\Users\user\Desktop\NYCU\影像處理\Term Project")
RESULTS = PROJ / "results"


# ---------- (1) 09 γ=1.0 + masked text sharpen ----------
print("\n=== Variant 1: 09 γ=1.0 + masked text sharpen ===")
src_09 = RESULTS / "abl_no_gamma_full" / "09_White_Truck_Zoom_Blur_Rain.png"
assert src_09.exists(), src_09
out_dir = RESULTS / "pipe_09_no_gamma_text_sharp"
out_dir.mkdir(parents=True, exist_ok=True)


def soft_saturation_mask(rgb_u8, sat_thr=70, feather=15):
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    m = np.clip((s - sat_thr) / max(1, 255 - sat_thr), 0, 1)
    k = feather
    return cv2.GaussianBlur(m, (2 * k + 1, 2 * k + 1), k / 2)


pil = Image.open(src_09).convert("RGB")
rgb = np.array(pil)
sharp_pil = pil.filter(ImageFilter.UnsharpMask(radius=3, percent=220, threshold=2))
sharp = np.array(sharp_pil)
mask = soft_saturation_mask(rgb, sat_thr=70, feather=15)[..., None]
out_arr = sharp.astype(np.float32) * mask + rgb.astype(np.float32) * (1 - mask)
out_arr = np.clip(out_arr, 0, 255).astype(np.uint8)
out_path = out_dir / "09_White_Truck_Zoom_Blur_Rain.png"
Image.fromarray(out_arr).save(out_path)
print(f"  wrote {out_path}")


# ---------- (2) Unsharp percent sweep ----------
print("\n=== Variant 2: Unsharp percent sweep (100/160/200) ===")
src_fft = RESULTS / "abl_no_unsharp_fft"

for pct in (100, 160, 200):
    out_dir = RESULTS / f"unsharp_sweep_{pct}_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_fft.iterdir()):
        if f.suffix.lower() != ".png":
            continue
        im = Image.open(f).convert("RGB")
        im2 = im.filter(ImageFilter.UnsharpMask(radius=2, percent=pct, threshold=2))
        im2.save(out_dir / f.name)
    print(f"  unsharp_{pct}: 15 files written")

print("\n[done] quick variants complete")
