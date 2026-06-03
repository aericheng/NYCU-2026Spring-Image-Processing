"""Two-stage subject restoration batch: crop subject (RAW native) -> FFTformer
(structure) -> SUPIR (generative detail). Outputs per-subject stage0/1/2 + montage.

Pipeline per subject:
  stage0_raw   : native crop of the visual-center subject from the RAW image
  stage1_fft   : FFTformer --max-side 1024 on the crop (clean structure, kernel in range)
  stage2_supir : SUPIR (two-stage input = stage1) ??v0F for fidelity; faces also v0Q

SUPIR runs against the already-running ComfyUI server (supir_api.py).
"""
import shutil
import subprocess
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJ = Path(r"C:\Users\user\Desktop\NYCU\影像處理\Term Project")
IMAGES = PROJ / "Images"
OUT = PROJ / "results" / "supir_batch"
DEBLUR = Path(r"C:\ProgramData\miniconda3\envs\deblur\python.exe")
COMFY = Path(r"C:\ProgramData\miniconda3\envs\comfy\python.exe")
FFT_DIR = PROJ / "FFTformer"
WEIGHTS = "pretrain_model/Realblur/net_g_Realblur_J.pth"
SUPIR_API = PROJ / "scripts" / "supir_api.py"

SUBJECTS = {
    "07_Yellow_Taxi_Neon_Rain_Street": dict(
        box=(0.10, 0.56, 0.62, 0.76), versions=["v0F"],
        prompt="yellow taxi car side, sharp license plate text and numbers, glossy car door, photorealistic, crisp detail"),
    "08_KFC_Rider_Rainy_Night_Delivery": dict(
        box=(0.34, 0.40, 0.56, 0.66), versions=["v0F"],
        prompt="red KFC delivery box with Colonel Sanders logo and Chinese text, sharp focus, photorealistic, crisp"),
    "09_White_Truck_Zoom_Blur_Rain": dict(
        box=(0.18, 0.27, 0.56, 0.49), versions=["v0F"],
        prompt="white Biffa waste truck with red Biffa.co.uk logo lettering, sharp text, photorealistic, crisp"),
    "14_Yellow_Chair_Alley_Motion_Portrait": dict(
        box=(0.32, 0.25, 0.64, 0.49), versions=["v0F", "v0Q"],
        prompt="a person face, sharp facial features, eyes nose mouth, detailed skin, photorealistic portrait"),
    "15_Photographer_Reflected_In_Night_Glass": dict(
        box=(0.36, 0.37, 0.66, 0.61), versions=["v0F", "v0Q"],
        prompt="a man face, sharp facial features, detailed skin, photorealistic portrait, night street"),
}
RAW = {k: k + ".jpg" for k in SUBJECTS}


def run(cmd, cwd=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, check=False)
    if r.returncode != 0:
        print(f"[warn] rc={r.returncode}", flush=True)
    return r.returncode


def crop_box(im, box):
    w, h = im.size
    x0, y0, x1, y1 = box
    return im.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fft_in = OUT / "_fft_in"
    fft_out = OUT / "_fft_out"
    if fft_in.exists():
        shutil.rmtree(fft_in)
    fft_in.mkdir(parents=True, exist_ok=True)

    # stage0: crop subjects
    for stem, c in SUBJECTS.items():
        raw = Image.open(IMAGES / RAW[stem]).convert("RGB")
        crop = crop_box(raw, c["box"])
        d = OUT / stem
        d.mkdir(parents=True, exist_ok=True)
        crop.save(d / "stage0_raw.png")
        crop.save(fft_in / f"{stem}.png")
        print(f"[crop] {stem}: {crop.size}", flush=True)

    # stage1: FFTformer on all crops at once (max-side 1024)
    run([str(DEBLUR), "run_fftformer.py", "-i", str(fft_in), "-o", str(fft_out),
         "-w", WEIGHTS, "--max-side", "1024"], cwd=str(FFT_DIR))
    for stem in SUBJECTS:
        src = fft_out / f"{stem}.png"
        if src.exists():
            shutil.copy2(src, OUT / stem / "stage1_fft.png")

    # stage2: SUPIR on stage1 (two-stage), per version
    for stem, c in SUBJECTS.items():
        s1 = OUT / stem / "stage1_fft.png"
        if not s1.exists():
            print(f"[skip supir] missing {s1}", flush=True)
            continue
        for ver in c["versions"]:
            outp = OUT / stem / f"stage2_supir_{ver}.png"
            run([str(COMFY), str(SUPIR_API), "--in", str(s1), "--out", str(outp),
                 "--supir", ver, "--steps", "10", "--cfg", "1.5", "--scale", "1.0",
                 "--control", "1.0", "--prompt", c["prompt"]])

    # montages
    try:
        font = ImageFont.truetype("arialbd.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    LH = 32
    for stem, c in SUBJECTS.items():
        d = OUT / stem
        panels = [("RAW", d / "stage0_raw.png"), ("FFTformer", d / "stage1_fft.png")]
        for ver in c["versions"]:
            panels.append((f"FFT?îUPIR {ver}", d / f"stage2_supir_{ver}.png"))
        loaded = [(l, Image.open(p).convert("RGB")) for l, p in panels if p.exists()]
        if not loaded:
            continue
        H = max(i.height for _, i in loaded)
        # normalize heights to H
        norm = []
        for l, i in loaded:
            if i.height != H:
                i = i.resize((round(i.width * H / i.height), H), Image.LANCZOS)
            norm.append((l, i))
        GAP = 8
        W = sum(i.width for _, i in norm) + GAP * (len(norm) - 1)
        canvas = Image.new("RGB", (W, H + LH), (15, 15, 15))
        dr = ImageDraw.Draw(canvas)
        x = 0
        for l, i in norm:
            dr.rectangle([x, 0, x + i.width, LH], fill=(0, 0, 0))
            dr.text((x + 6, 5), l, fill=(255, 235, 60), font=font)
            canvas.paste(i, (x, LH))
            x += i.width + GAP
        canvas.save(OUT / f"{stem}__montage.jpg", quality=92)
        print(f"[montage] {stem}", flush=True)

    print("\n[done] supir batch ->", OUT, flush=True)


if __name__ == "__main__":
    main()
