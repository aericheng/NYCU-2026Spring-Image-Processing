"""Reproduce the two submitted night-blur face restorations.

Pipeline per face (see report/Report.pdf, section 3):
  crop native face -> preprocess (gamma/CLAHE/saturation)
  -> FFTformer RealBlur-J (motion deblur, provides clean structure)
  -> SUPIR v0F (diffusion prior, high control, synthesises face detail)
  -> finalize (man: feather-composite onto the blurry frame; woman: tight crop)
  -> light OpenCV/NumPy grade.

Outputs land in results/final2/<key>/ and the two FINAL_graded.png files.
Requires: the FFTformer/ checkout (deblur env) and a running ComfyUI SUPIR
server on :8188 (comfy env) — see README. Saving uses PIL because cv2.imwrite
fails on this non-ASCII project path.
"""
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

PROJ = Path(r"C:\Users\user\Desktop\NYCU\影像處理\Term Project")
IMAGES = PROJ / "Images"
OUT = PROJ / "results" / "final2"
SUB = PROJ / "final_submissions" / "Faces_2026-06-04"
DEBLUR = Path(r"C:\ProgramData\miniconda3\envs\deblur\python.exe")
COMFY = Path(r"C:\ProgramData\miniconda3\envs\comfy\python.exe")
FFT_DIR = PROJ / "FFTformer"
WEIGHTS = "pretrain_model/Realblur/net_g_Realblur_J.pth"
SUPIR_API = PROJ / "scripts" / "supir_api.py"

# Per-face configuration. box/sub_box are (x0,y0,x1,y1) fractions of the frame.
FACES = {
    "10_man": dict(
        stem="10_Crowded_Night_Market_Face_Glow",
        submit="01_NightMarket_Man__FINAL.png",
        box=(0.33, 0.40, 0.52, 0.80), pre=(0.85, 3.0, 1.10),
        fft_maxside=0,                       # native (small crop)
        supir=dict(steps=10, cfg=1.5, scale=2.0, control=1.0, seed=123),
        prompt="a young man face, sharp clear facial features, eyes nose mouth, "
               "detailed natural skin, photorealistic portrait, night market crowd",
        finalize="composite",                # paste sharp face onto the blurry frame
        ellipse=dict(cx=0.55, cy=0.64, rx=0.30, ry=0.40),
        portrait=(0.295, 0.35, 0.55, 0.83),  # final framing on the full frame
    ),
    "14_woman": dict(
        stem="14_Yellow_Chair_Alley_Motion_Portrait",
        submit="02_NeonAlley_Woman__FINAL.png",
        box=(0.28, 0.20, 0.66, 0.62), pre=(0.78, 3.5, 1.15),
        fft_maxside=1024,                    # large kernel -> downscale into range
        supir=dict(steps=14, cfg=1.3, scale=1.5, control=1.0, seed=42),
        prompt="a young woman face, sharp clear symmetric eyes, eyebrows, nose, lips, "
               "hoop earring, detailed natural skin, photorealistic portrait, warm neon night",
        finalize="crop",                     # tight crop of the SUPIR result
        sub_box=(0.45, 0.10, 0.86, 0.60),    # excludes the lower-left ghost artifact
    ),
}


def gamma(a, g):
    t = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(a, t)


def clahe(a, clip, grid=8):
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    L2 = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2RGB)


def saturate(a, s):
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def preprocess(rgb, p):
    g, c, s = p
    return saturate(clahe(gamma(rgb, g), c), s)


def light_grade(rgb):
    H, W = rgb.shape[:2]
    x = rgb.astype(np.float32) / 255.0
    x = np.clip((x - 0.5) * 1.09 + 0.5, 0, 1)                       # gentle filmic contrast
    luma = x @ np.array([0.299, 0.587, 0.114], np.float32)
    hi = np.clip(luma * 1.6 - 0.6, 0, 1)[..., None]
    sh = np.clip(1 - luma * 1.7, 0, 1)[..., None]
    x = np.clip(x + hi * np.array([0.04, 0.02, 0.0], np.float32)
                  - sh * np.array([0.0, 0.01, 0.025], np.float32), 0, 1)   # warm hi / cool sh
    hsv = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.08, 0, 255)
    x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(x, (0, 0), sigmaX=max(2, min(H, W) // 90))
    x = np.clip(x + (x - blur) * 0.35, 0, 1)                        # subtle clarity
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vig = np.clip((r - 0.55) / 0.85, 0, 1)
    x = x * (1 - 0.18 * vig[..., None])                            # face-focus vignette
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def run(cmd, cwd=None):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=False)


def restore(key, cfg):
    d = OUT / key
    d.mkdir(parents=True, exist_ok=True)
    raw = np.array(Image.open(IMAGES / f"{cfg['stem']}.jpg").convert("RGB"))
    H, W = raw.shape[:2]
    bx = [round(v * dim) for v, dim in zip(cfg["box"], (W, H, W, H))]
    x0, y0, x1, y1 = bx
    full_pre = preprocess(raw, cfg["pre"])
    crop_pre = full_pre[y0:y1, x0:x1].copy()

    # Stage 1: FFTformer
    fin = d / "_fft_in"
    if fin.exists():
        shutil.rmtree(fin)
    fin.mkdir(parents=True)
    Image.fromarray(crop_pre).save(fin / "face.png")
    fout = d / "fft"
    flags = ["--max-side", str(cfg["fft_maxside"])] if cfg["fft_maxside"] else []
    run([str(DEBLUR), "run_fftformer.py", "-i", str(fin), "-o", str(fout),
         "-w", WEIGHTS] + flags, cwd=str(FFT_DIR))
    fft_png = fout / "face.png"

    # Stage 2: SUPIR v0F (needs the ComfyUI server up)
    s = cfg["supir"]
    sup_png = d / "supir.png"
    run([str(COMFY), str(SUPIR_API), "--in", str(fft_png), "--out", str(sup_png),
         "--supir", "v0F", "--steps", str(s["steps"]), "--cfg", str(s["cfg"]),
         "--scale", str(s["scale"]), "--control", str(s["control"]),
         "--seed", str(s["seed"]), "--tiled", "--prompt", cfg["prompt"]])
    sup = np.array(Image.open(sup_png).convert("RGB"))

    # Finalize
    if cfg["finalize"] == "composite":
        cw, ch = x1 - x0, y1 - y0
        sup_c = cv2.resize(sup, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
        e = cfg["ellipse"]
        mask = np.zeros((ch, cw), np.float32)
        cv2.ellipse(mask, (int(e["cx"] * cw), int(e["cy"] * ch)),
                    (int(e["rx"] * cw), int(e["ry"] * ch)), 0, 0, 360, 1.0, -1)
        fk = int(min(cw, ch) * 0.12) | 1
        mask = cv2.GaussianBlur(mask, (fk, fk), fk / 3)[..., None]
        comp = full_pre.astype(np.float32).copy()
        comp[y0:y1, x0:x1] = sup_c * mask + comp[y0:y1, x0:x1] * (1 - mask)
        comp = np.clip(comp, 0, 255).astype(np.uint8)
        p = cfg["portrait"]
        px = [round(p[0] * W), round(p[1] * H), round(p[2] * W), round(p[3] * H)]
        final_ungraded = comp[px[1]:px[3], px[0]:px[2]]
    else:  # crop
        cw, ch = x1 - x0, y1 - y0
        sup_c = cv2.resize(sup, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
        sb = cfg["sub_box"]
        sx = [round(sb[0] * cw), round(sb[1] * ch), round(sb[2] * cw), round(sb[3] * ch)]
        final_ungraded = sup_c[sx[1]:sx[3], sx[0]:sx[2]]

    final = light_grade(final_ungraded)
    Image.fromarray(final).save(d / f"{key}_FINAL_graded.png")
    Image.fromarray(final_ungraded).save(d / f"{key}_FINAL_ungraded.png")
    SUB.mkdir(parents=True, exist_ok=True)
    shutil.copy2(d / f"{key}_FINAL_graded.png", SUB / cfg["submit"])
    print(f"[done] {key} -> {SUB / cfg['submit']}", flush=True)


def main():
    for key, cfg in FACES.items():
        print("=" * 60 + f"\n{key}\n" + "=" * 60, flush=True)
        restore(key, cfg)
    print("\n[all done] finals under", OUT)


if __name__ == "__main__":
    main()
