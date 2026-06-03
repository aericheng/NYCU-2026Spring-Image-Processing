"""Cosmetic post-processing pass (presentation layer, not a restoration step).

For each final image: filmic contrast -> teal/orange split-tone -> vibrance ->
bloom on highlights (neon / wet reflections) -> subject-local clarity -> vignette.
Implemented in OpenCV / NumPy (no external image editor).
Outputs results/wow/graded/<name>.png + <name>_before_after.jpg
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJ = Path(r"C:\Users\user\Desktop\NYCU\影像處理\Term Project")
SUB = PROJ / "final_submissions" / "SUPIR_2026-06-03"
WOW = PROJ / "results" / "wow"
OUT = WOW / "graded"

# name -> (input path, subject box for local clarity)
JOBS = {
    "08_KFC": (WOW / "08_KFC_Rider_Rainy_Night_Delivery" / "scale2_composite.png", (0.34, 0.40, 0.56, 0.66)),
    "05_RedTaxi": (WOW / "05_Red_Taxi_Through_City_Lights" / "scale2_composite.png", (0.30, 0.50, 0.72, 0.80)),
    "15_Face": (WOW / "15_Photographer_Reflected_In_Night_Glass" / "whole_final.png", (0.36, 0.37, 0.66, 0.61)),
}


def grade(rgb_u8, box):
    H, W = rgb_u8.shape[:2]
    x = rgb_u8.astype(np.float32) / 255.0
    # 1. filmic contrast (gentle S-curve around 0.5)
    x = np.clip((x - 0.5) * 1.13 + 0.5, 0, 1)
    # 2. teal-orange split-tone
    luma = x @ np.array([0.299, 0.587, 0.114], np.float32)
    sh = np.clip(1 - luma * 1.8, 0, 1)[..., None]
    hi = np.clip(luma * 1.8 - 0.8, 0, 1)[..., None]
    teal = np.array([0.00, 0.35, 0.50], np.float32)
    warm = np.array([0.55, 0.32, 0.00], np.float32)
    x = np.clip(x + sh * teal * 0.05 + hi * warm * 0.05, 0, 1)
    # 3. vibrance (HSV saturation boost, stronger where less saturated)
    hsv = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.18, 0, 255)
    x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    # 4. bloom on highlights (neon + wet reflections)
    luma = x @ np.array([0.299, 0.587, 0.114], np.float32)
    bright = np.clip(luma - 0.62, 0, 1) / 0.38
    k = max(7, (min(H, W) // 120) * 2 + 1)
    glow = cv2.GaussianBlur(bright, (k, k), 0)[..., None]
    x = 1 - (1 - x) * (1 - glow * 0.45)          # screen blend
    x = np.clip(x, 0, 1)
    # 5. subject-local clarity (large-radius unsharp) + a touch more sat
    x0, y0, x1, y1 = [round(v * d) for v, d in zip(box, (W, H, W, H))]
    sub = x[y0:y1, x0:x1, :]
    blur = cv2.GaussianBlur(sub, (0, 0), sigmaX=max(3, (x1 - x0) // 60))
    sub = np.clip(sub + (sub - blur) * 0.6, 0, 1)   # clarity
    hsv2 = cv2.cvtColor((sub * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv2[:, :, 1] = np.clip(hsv2[:, :, 1] * 1.08, 0, 255)
    x[y0:y1, x0:x1, :] = cv2.cvtColor(hsv2.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    # 6. vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W / 2, H / 2
    r = np.sqrt(((xx - cx) / (W / 2)) ** 2 + ((yy - cy) / (H / 2)) ** 2)
    vig = np.clip((r - 0.6) / 0.8, 0, 1)
    x = x * (1 - 0.16 * vig[..., None])
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("arialbd.ttf", 30)
    except Exception:
        font = ImageFont.load_default()
    names = sys.argv[1:] if len(sys.argv) > 1 else list(JOBS)
    for name in names:
        inp, box = JOBS[name]
        if not inp.exists():
            print(f"[skip] missing {inp}")
            continue
        rgb = np.array(Image.open(inp).convert("RGB"))
        g = grade(rgb, box)
        Image.fromarray(g).save(OUT / f"{name}.png")
        # before/after (downscaled)
        def ds(arr, L=1500):
            h, w = arr.shape[:2]; s = L / max(h, w)
            return cv2.resize(arr, (round(w*s), round(h*s))) if s < 1 else arr
        a = ds(rgb); b = ds(g); Hh = max(a.shape[0], b.shape[0]); LH = 40; G = 10
        canvas = Image.new("RGB", (a.shape[1]+G+b.shape[1], Hh+LH), (15, 15, 15))
        d = ImageDraw.Draw(canvas); d.rectangle([0, 0, canvas.width, LH], fill=(0, 0, 0))
        d.text((10, 6), "BEFORE (deblur only)", fill=(255, 235, 60), font=font)
        d.text((a.shape[1]+G+10, 6), "AFTER (+ cinematic grade)", fill=(60, 255, 120), font=font)
        canvas.paste(Image.fromarray(a), (0, LH)); canvas.paste(Image.fromarray(b), (a.shape[1]+G, LH))
        canvas.save(OUT / f"{name}_before_after.jpg", quality=92)
        print(f"[graded] {name} -> {OUT/(name+'.png')}", flush=True)
    print("[done]")


if __name__ == "__main__":
    main()
