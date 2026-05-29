"""Build the I+ submission compare jpgs + standalone PNGs.

For Plan I+ (the chosen submission), generates:
  final_submissions/I_plus_pipeA_strengthened/<stem>_compare.jpg
      input | output side-by-side, near-native resolution (max-side 1800)
  final_submissions/I_plus_pipeA_strengthened/<stem>__<method>.png
      the standalone deblurred PNG (the upload file)

Run once after results/pipe_09_v2_text_sharp/ and results/pipe_08_v2_final/
finish; the outputs are then committed.
"""
from pathlib import Path
import shutil
from PIL import Image, ImageDraw, ImageFont

PROJ = Path(r"C:\Users\user\Desktop\NYCU\影像處理\Term Project")
SUBMIT = PROJ / "final_submissions"
SUBMIT.mkdir(parents=True, exist_ok=True)

PLAN_KEY = "I_plus_pipeA_strengthened"
PLAN_ITEMS = [
    ("09_White_Truck_Zoom_Blur_Rain",     "pipe_09_v2_text_sharp", "Pipe A v2 (γ=1.0) + masked text sharpen"),
    ("08_KFC_Rider_Rainy_Night_Delivery", "pipe_08_v2_final",      "Pipe A v2 (γ=0.7, CLAHE 5, unsharp 200)"),
]


def load_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def resize_max(im, max_side):
    if max(im.size) <= max_side:
        return im
    s = max_side / max(im.size)
    return im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.LANCZOS)


def make_pair_image(stem, method_dir, method_label, max_side=1800):
    inp_path = PROJ / "Images" / f"{stem}.jpg"
    out_path = PROJ / "results" / method_dir / f"{stem}.png"
    inp = resize_max(Image.open(inp_path).convert("RGB"), max_side)
    out = resize_max(Image.open(out_path).convert("RGB"), max_side)
    h = max(inp.size[1], out.size[1])
    if inp.size[1] != h:
        inp = inp.resize((int(inp.size[0] * h / inp.size[1]), h), Image.LANCZOS)
    if out.size[1] != h:
        out = out.resize((int(out.size[0] * h / out.size[1]), h), Image.LANCZOS)
    gap = 12
    band = 36
    w = inp.size[0] + out.size[0] + gap
    canvas = Image.new("RGB", (w, h + band), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    f = load_font(18)
    canvas.paste(inp, (0, band))
    draw.text((10, 8), f"input  ({inp.size[0]}x{inp.size[1]})", fill="white", font=f)
    canvas.paste(out, (inp.size[0] + gap, band))
    draw.text((inp.size[0] + gap + 10, 8),
              f"{method_label}  ({out.size[0]}x{out.size[1]})",
              fill="white", font=f)
    return canvas


plan_dir = SUBMIT / PLAN_KEY
plan_dir.mkdir(parents=True, exist_ok=True)

for stem, mdir, mlabel in PLAN_ITEMS:
    pair_big = make_pair_image(stem, mdir, mlabel, max_side=1800)
    cmp_path = plan_dir / f"{stem}_compare.jpg"
    pair_big.save(cmp_path, quality=92)
    print(f"  cmp: {cmp_path.relative_to(SUBMIT)}")

    src = PROJ / "results" / mdir / f"{stem}.png"
    dst = plan_dir / f"{stem}__{mdir}.png"
    shutil.copy2(src, dst)
    print(f"  png: {dst.relative_to(SUBMIT)}")

print("done")
