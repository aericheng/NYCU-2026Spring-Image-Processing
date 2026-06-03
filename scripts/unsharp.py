"""Apply unsharp mask post-process to a folder of images."""
import argparse
from pathlib import Path
from PIL import Image, ImageFilter

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--inp", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--radius", type=float, default=2.0, help="Gaussian radius for unsharp")
    p.add_argument("--percent", type=int, default=120, help="Strength percent (100 = identity-ish, 200 = strong)")
    p.add_argument("--threshold", type=int, default=2, help="Don't sharpen low-contrast edges below this")
    args = p.parse_args()

    inp = Path(args.inp); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted([f for f in inp.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    for i, f in enumerate(files, 1):
        im = Image.open(f).convert("RGB")
        sharpened = im.filter(ImageFilter.UnsharpMask(radius=args.radius,
                                                     percent=args.percent,
                                                     threshold=args.threshold))
        sharpened.save(out / (f.stem + ".png"))
        print(f"[{i}/{len(files)}] {f.name}")
    print("done")

if __name__ == "__main__":
    main()
