"""Build a single grid comparison of every sharpening method per image.

Layout (4 rows x 2 cols):
    Row 0:  Original                |  Frequency
    Row 1:  Spatial 4n  zero-pad    |  Spatial 4n  replicate-pad
    Row 2:  Spatial 8n  zero-pad    |  Spatial 8n  replicate-pad
    Row 3:  Spatial LoG zero-pad    |  Spatial LoG replicate-pad
"""
from __future__ import annotations

import os
import sys
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPATIAL_DIR = os.path.join(ROOT, "results", "spatial")
FREQ_DIR = os.path.join(ROOT, "results", "frequency")
OUT_DIR = os.path.join(ROOT, "results", "comparison")

LABEL_HEIGHT = 36
GAP = 8
COLS = 2


def load_font(size: int = 22) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def panel_paths(name: str) -> list[tuple[str, str]]:
    return [
        ("Original",                  os.path.join(SPATIAL_DIR, f"{name}_original.png")),
        ("Frequency",                 os.path.join(FREQ_DIR,    f"{name}_sharpened_freq.png")),
        ("Spatial 4n  zero-pad",      os.path.join(SPATIAL_DIR, f"{name}_sharpened_4n_zero.png")),
        ("Spatial 4n  replicate-pad", os.path.join(SPATIAL_DIR, f"{name}_sharpened_4n_replicate.png")),
        ("Spatial 8n  zero-pad",      os.path.join(SPATIAL_DIR, f"{name}_sharpened_8n_zero.png")),
        ("Spatial 8n  replicate-pad", os.path.join(SPATIAL_DIR, f"{name}_sharpened_8n_replicate.png")),
        ("Spatial LoG zero-pad",      os.path.join(SPATIAL_DIR, f"{name}_sharpened_log_zero.png")),
        ("Spatial LoG replicate-pad", os.path.join(SPATIAL_DIR, f"{name}_sharpened_log_replicate.png")),
    ]


def grid_compare(name: str, out_path: str) -> bool:
    panels = panel_paths(name)
    for _, p in panels:
        if not os.path.exists(p):
            print(f"skip {name}: missing {os.path.basename(p)}")
            return False

    images = [Image.open(p).convert("RGB") for _, p in panels]
    base_w, base_h = images[0].size
    images = [im.resize((base_w, base_h)) for im in images]

    rows = (len(panels) + COLS - 1) // COLS
    total_w = base_w * COLS + GAP * (COLS - 1)
    total_h = (base_h + LABEL_HEIGHT) * rows + GAP * (rows - 1)

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    font = load_font(22)
    draw = ImageDraw.Draw(canvas)

    for i, ((label, _), im) in enumerate(zip(panels, images)):
        r, c = divmod(i, COLS)
        x = c * (base_w + GAP)
        y = r * (base_h + LABEL_HEIGHT + GAP)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (base_w - tw) // 2, y + 6),
                  label, fill=(0, 0, 0), font=font)
        canvas.paste(im, (x, y + LABEL_HEIGHT))

    canvas.save(out_path)
    return True


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    names = sorted(
        f.split("_original")[0]
        for f in os.listdir(SPATIAL_DIR)
        if f.endswith("_original.png")
    )
    for name in names:
        out = os.path.join(OUT_DIR, f"{name}_comparison.png")
        if grid_compare(name, out):
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
