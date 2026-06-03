"""Reproduce the MUSIQ numbers in the report for the two final images.

Computes MUSIQ (No-Reference) at three stages to separate the contribution of
deblurring from the cosmetic grade:
  RAW           -> original input
  DEBLUR-ONLY   -> FFTformer + SUPIR x2 subject composite, BEFORE cinematic grade
  FINAL         -> after cinematic grade (the uploaded image)

All scored at 1536px long side (RAW is downsampled to match) so the comparison
is at a common scale. Saves report/musiq_final.csv.
"""
import csv
from pathlib import Path

import torch
import torchvision.transforms as T
import pyiqa
from PIL import Image

PROJ = Path(r"C:\Users\user\Desktop\NYCU\影像處理\Term Project")
LONG = 1536

ITEMS = {
    "08_KFC": {
        "RAW":    PROJ / "Images" / "08_KFC_Rider_Rainy_Night_Delivery.jpg",
        "DEBLUR": PROJ / "results" / "wow" / "08_KFC_Rider_Rainy_Night_Delivery" / "scale2_composite.png",
        "FINAL":  PROJ / "final_submissions" / "SUPIR_2026-06-03" / "08_KFC_Rider__FINAL_graded.png",
    },
    "05_RedTaxi": {
        "RAW":    PROJ / "Images" / "05_Red_Taxi_Through_City_Lights.jpg",
        "DEBLUR": PROJ / "results" / "wow" / "05_Red_Taxi_Through_City_Lights" / "scale2_composite.png",
        "FINAL":  PROJ / "final_submissions" / "SUPIR_2026-06-03" / "05_Red_Taxi__FINAL_graded.png",
    },
}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    musiq = pyiqa.create_metric("musiq", device=dev)
    to_t = T.ToTensor()
    rows = [["image", "stage", "MUSIQ"]]
    for name, stages in ITEMS.items():
        print(f"\n{name}")
        for stage, path in stages.items():
            im = Image.open(path).convert("RGB")
            w, h = im.size
            s = LONG / max(w, h)
            if s < 1:
                im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
            t = to_t(im).unsqueeze(0).to(dev)
            score = float(musiq(t))
            print(f"  {stage:7s} MUSIQ={score:.2f}")
            rows.append([name, stage, f"{score:.2f}"])
    out = PROJ / "report" / "musiq_final.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
