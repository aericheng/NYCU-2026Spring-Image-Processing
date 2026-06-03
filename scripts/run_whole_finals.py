"""Whole-image final pipeline for 08 + 07: FFT whole -> SUPIR whole (tiled @ 4096)
-> upscale to native. Negative prompt kills the SUPIR faux-signature artifact.

Outputs: results/final_supir/<stem>/whole_final.png (native), whole_final_view.jpg,
         <stem>_whole_before_after.jpg
"""
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJ = Path(r"C:\Users\user\Desktop\NYCU\敶勗???\Term Project")
IMAGES = PROJ / "Images"
FS = PROJ / "results" / "final_supir"
DEBLUR = Path(r"C:\ProgramData\miniconda3\envs\deblur\python.exe")
COMFY = Path(r"C:\ProgramData\miniconda3\envs\comfy\python.exe")
FFT_DIR = PROJ / "FFTformer"
WEIGHTS = "pretrain_model/Realblur/net_g_Realblur_J.pth"
SUPIR_API = PROJ / "scripts" / "supir_api.py"

NPROMPT = ("blurry, motion blur, out of focus, low quality, noise, jpeg artifacts, "
           "signature, watermark, autograph, logo overlay, frame, border, text watermark")

JOBS = {
    "08_KFC_Rider_Rainy_Night_Delivery":
        "red KFC delivery box with Colonel Sanders logo and Chinese text, sharp focus, photorealistic, crisp, night street panning shot",
    "07_Yellow_Taxi_Neon_Rain_Street":
        "yellow taxi car, sharp license plate, glossy car body, neon city night, photorealistic, crisp, panning shot",
}
TARGET = 4096

try:
    FONT = ImageFont.truetype("arialbd.ttf", 30)
except Exception:
    FONT = ImageFont.load_default()


def run(cmd, cwd=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd, check=False).returncode


def main():
    for stem, prompt in JOBS.items():
        d = FS / stem; d.mkdir(parents=True, exist_ok=True)
        raw = Image.open(IMAGES / f"{stem}.jpg").convert("RGB")
        W, H = raw.size
        s = TARGET / max(W, H)
        small = raw.resize((round(W * s), round(H * s)), Image.LANCZOS)
        fin = d / "_wf_in"; fin.mkdir(exist_ok=True)
        small.save(fin / f"{stem}.png")
        fout = d / "_wf_fft"
        run([str(DEBLUR), "run_fftformer.py", "-i", str(fin), "-o", str(fout),
             "-w", WEIGHTS, "--tile", "768", "--overlap", "96"], cwd=str(FFT_DIR))
        sup = d / "_wf_supir.png"
        run([str(COMFY), str(SUPIR_API), "--in", str(fout / f"{stem}.png"), "--out", str(sup),
             "--supir", "v0F", "--steps", "10", "--cfg", "1.5", "--scale", "1.0",
             "--control", "1.0", "--tiled", "--prompt", prompt, "--nprompt", NPROMPT])
        # upscale to native
        res = Image.open(sup).convert("RGB").resize((W, H), Image.LANCZOS)
        res.save(d / "whole_final.png")
        # view + before/after
        def ds(im, L=1500):
            w, h = im.size; sc = L / max(w, h)
            return im.resize((round(w*sc), round(h*sc)), Image.LANCZOS) if sc < 1 else im
        ds(res).save(d / "whole_final_view.jpg", quality=92)
        a = ds(raw); b = ds(res); Hh = max(a.height, b.height); LH = 40; G = 10
        c = Image.new("RGB", (a.width+G+b.width, Hh+LH), (15, 15, 15)); dr = ImageDraw.Draw(c)
        dr.rectangle([0, 0, c.width, LH], fill=(0, 0, 0))
        dr.text((10, 6), "BEFORE (raw)", fill=(255, 235, 60), font=FONT)
        dr.text((a.width+G+10, 6), "AFTER (whole FFT->SUPIR)", fill=(60, 255, 120), font=FONT)
        c.paste(a, (0, LH)); c.paste(b, (a.width+G, LH))
        c.save(FS / f"{stem}_whole_before_after.jpg", quality=92)
        print(f"[done] {stem} -> whole_final.png ({W}x{H})", flush=True)
    print("\n[all done]", flush=True)


if __name__ == "__main__":
    main()
