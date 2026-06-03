"""Re-run 08/05 subject SUPIR scale x2 with TILED sampling (fixes OOM) + composite."""
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

PROJ = Path(r"C:\Users\user\Desktop\NYCU\敶勗???\Term Project")
IMAGES = PROJ / "Images"; BATCH = PROJ / "results" / "supir_batch"; WOW = PROJ / "results" / "wow"
COMFY = Path(r"C:\ProgramData\miniconda3\envs\comfy\python.exe")
SUPIR_API = PROJ / "scripts" / "supir_api.py"


def run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=False).returncode


def feather(w, h, f):
    ax = np.ones(w); ay = np.ones(h); r = np.linspace(0, 1, max(1, f))
    ax[:f] = r; ax[-f:] = r[::-1]; ay[:f] = r; ay[-f:] = r[::-1]
    return (ay[:, None] * ax[None, :]).astype(np.float32)


def composite(stem, box, subj, out):
    raw = Image.open(IMAGES / f"{stem}.jpg").convert("RGB"); W, H = raw.size
    x0, y0, x1, y1 = [round(v*d) for v, d in zip(box, (W, H, W, H))]
    s = np.array(subj.resize((x1-x0, y1-y0), Image.LANCZOS)).astype(np.float32)
    base = np.array(raw).astype(np.float32)
    f = max(8, int(min(x1-x0, y1-y0)*0.09)); a = feather(x1-x0, y1-y0, f)[..., None]
    base[y0:y1, x0:x1, :] = s*a + base[y0:y1, x0:x1, :]*(1-a)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(out)
    print(f"[composite] {stem} -> {out.name}", flush=True)


for stem, box, prompt in [
    ("08_KFC_Rider_Rainy_Night_Delivery", (0.34, 0.40, 0.56, 0.66),
     "red KFC delivery box, Colonel Sanders logo, sharp focus, photorealistic, crisp, fine fabric texture"),
    ("05_Red_Taxi_Through_City_Lights", (0.30, 0.50, 0.72, 0.80),
     "red taxi car, sharp body windows wheels, license plate, neon night, photorealistic, crisp"),
]:
    out2 = WOW / stem / "supir_scale2.png"; out2.parent.mkdir(parents=True, exist_ok=True)
    run([str(COMFY), str(SUPIR_API), "--in", str(BATCH / stem / "stage1_fft.png"),
         "--out", str(out2), "--supir", "v0F", "--steps", "10", "--cfg", "1.5",
         "--scale", "2.0", "--control", "1.0", "--tiled", "--prompt", prompt])
    if out2.exists():
        composite(stem, box, Image.open(out2).convert("RGB"), WOW / stem / "scale2_composite.png")
print("[scale2 done]", flush=True)
