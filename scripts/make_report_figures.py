"""Generate all figures for the rewritten report into report/figures/ (prefix rpt_)."""
import os
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJ = Path(r"C:\Users\user\Desktop\NYCU\影像處理\Term Project")
R = PROJ / "results"
SUB = PROJ / "final_submissions" / "SUPIR_2026-06-03"
DT = Path(r"C:\Users\user\DiffTSR\testset")
FIG = PROJ / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def font(sz):
    for n in ("arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def label_strip(images_labels, label_h=34, gap=8, fsz=22, vertical=False):
    ims = [(l, Image.open(p).convert("RGB")) for l, p in images_labels]
    f = font(fsz)
    if vertical:
        W = max(i.width for _, i in ims)
        H = sum(i.height + label_h for _, i in ims)
        c = Image.new("RGB", (W, H), (15, 15, 15)); d = ImageDraw.Draw(c); y = 0
        for l, i in ims:
            d.rectangle([0, y, W, y + label_h], fill=(0, 0, 0)); d.text((8, y + 6), l, fill=(255, 235, 60), font=f)
            y += label_h; c.paste(i, (0, y)); y += i.height
    else:
        H = max(i.height for _, i in ims)
        norm = [(l, i.resize((round(i.width * H / i.height), H), Image.LANCZOS) if i.height != H else i) for l, i in ims]
        W = sum(i.width for _, i in norm) + gap * (len(norm) - 1)
        c = Image.new("RGB", (W, H + label_h), (15, 15, 15)); d = ImageDraw.Draw(c); x = 0
        for l, i in norm:
            d.rectangle([x, 0, x + i.width, label_h], fill=(0, 0, 0)); d.text((x + 6, 5), l, fill=(255, 235, 60), font=f)
            c.paste(i, (x, label_h)); x += i.width + gap
    return c


def cp(src, name):
    if Path(src).exists():
        shutil.copy2(src, FIG / name); print("[copy]", name)
    else:
        print("[MISS]", src)


# --- f1 pipeline diagram ---
def pipeline_diagram():
    steps = ["RAW\nnight-blur\n(6-8K)", "Crop\nsubject", "Stage 1\nFFTformer\n(deblur\nstructure)",
             "Stage 2\nSUPIR x2\n(generative\ndetail)", "Feather\ncomposite\n(bg blur kept)",
             "Cinematic\ngrade +\nbloom", "FINAL"]
    bw, bh, gap, pad = 200, 150, 60, 30
    W = len(steps) * bw + (len(steps) - 1) * gap + pad * 2
    H = bh + pad * 2 + 40
    c = Image.new("RGB", (W, H), (250, 250, 250)); d = ImageDraw.Draw(c)
    f = font(22); ft = font(26)
    d.text((pad, 8), "Two-stage generative subject restoration pipeline", fill=(20, 20, 20), font=ft)
    x = pad; y = pad + 30
    cols = [(208, 64, 64), (224, 132, 48), (48, 120, 200), (140, 64, 200), (40, 150, 110), (210, 150, 40), (30, 30, 30)]
    for i, s in enumerate(steps):
        d.rounded_rectangle([x, y, x + bw, y + bh], radius=16, fill=cols[i])
        lines = s.split("\n")
        ty = y + bh // 2 - len(lines) * 13
        for ln in lines:
            tw = d.textlength(ln, font=f)
            d.text((x + bw / 2 - tw / 2, ty), ln, fill=(255, 255, 255), font=f); ty += 26
        if i < len(steps) - 1:
            ax = x + bw + gap // 2
            d.polygon([(x + bw + 8, y + bh / 2 - 10), (ax + 14, y + bh / 2), (x + bw + 8, y + bh / 2 + 10)], fill=(80, 80, 80))
            d.line([(x + bw, y + bh / 2), (ax + 6, y + bh / 2)], fill=(80, 80, 80), width=4)
        x += bw + gap
    c.save(FIG / "rpt_f1_pipeline.png"); print("[fig] rpt_f1_pipeline.png")


# --- f2 resolution strategy (09 zoom needs downscale; 08 mild needs hi-res) ---
def resolution_fig():
    a = Image.open(R / "v3" / "09_White_Truck_Zoom_Blur_Rain__ab.png").convert("RGB")
    b = Image.open(R / "v3" / "08_KFC_Rider_Rainy_Night_Delivery__ab.png").convert("RGB")
    Wt = 760
    a = a.resize((Wt, round(a.height * Wt / a.width)), Image.LANCZOS)
    b = b.resize((Wt, round(b.height * Wt / b.width)), Image.LANCZOS)
    f = font(22); LH = 30
    c = Image.new("RGB", (Wt * 2 + 20, max(a.height, b.height) + LH), (15, 15, 15)); d = ImageDraw.Draw(c)
    d.text((6, 5), "09 zoom-blur: downscale (up1024) WINS", fill=(255, 235, 60), font=f)
    d.text((Wt + 26, 5), "08 mild-blur: hi-res tiling (t3072) WINS", fill=(255, 235, 60), font=f)
    c.paste(a, (0, LH)); c.paste(b, (Wt + 20, LH))
    c.save(FIG / "rpt_f2_resolution.jpg", quality=92); print("[fig] rpt_f2_resolution.jpg")


# --- f8 DiffTSR: verify (works) + all15 (fails on our data) ---
def difftsr_fig():
    # verify samples
    samples = ["0", "1", "2", "4"]
    f = font(18); cw, ch = 360, 90; LH = 22
    vc = Image.new("RGB", (cw, LH + len(samples) * (ch * 2 + 6 + LH)), (12, 12, 12)); dd = ImageDraw.Draw(vc); y = 0
    for s in samples:
        dd.text((4, y + 2), "#" + s + " LR-in / DiffTSR-out", fill=(255, 235, 60), font=f); y += LH
        try:
            a = Image.open(DT / "0_lr_synth" / (s + ".png")).convert("RGB").resize((cw, ch))
            b = Image.open(DT / "0_sr_verify" / (s + ".png")).convert("RGB").resize((cw, ch))
            vc.paste(a, (0, y)); vc.paste(b, (0, y + ch + 6)); y += ch * 2 + 6 + 8
        except Exception as e:
            print("verify miss", e)
    vc.save(FIG / "rpt_f8a_difftsr_works.jpg", quality=93)
    # all15 grid (reuse builder logic, 5 cols x 3 rows)
    stems = sorted([p.stem for p in (DT / "all15_sr").glob("*.png")])
    cols = 5; cw2, ch2 = 300, 75; LH2 = 20
    rows = (len(stems) + cols - 1) // cols
    cellH = LH2 + ch2 * 2 + 4
    gc = Image.new("RGB", (cols * (cw2 + 6), rows * (cellH + 6)), (12, 12, 12)); dg = ImageDraw.Draw(gc); f2 = font(15)
    for i, s in enumerate(stems):
        cx = (i % cols) * (cw2 + 6); cy = (i // cols) * (cellH + 6)
        dg.text((cx + 3, cy + 1), "#" + s.split("_")[0] + " in/out", fill=(255, 235, 60), font=f2)
        a = Image.open(DT / "all15_lr" / (s + ".png")).convert("RGB").resize((cw2, ch2))
        b = Image.open(DT / "all15_sr" / (s + ".png")).convert("RGB").resize((cw2, ch2))
        gc.paste(a, (cx, cy + LH2)); gc.paste(b, (cx, cy + LH2 + ch2 + 2))
    gc.save(FIG / "rpt_f8b_difftsr_all15.jpg", quality=92)
    print("[fig] rpt_f8 difftsr")


def main():
    pipeline_diagram()
    resolution_fig()
    cp(R / "supir_batch" / "08_KFC_Rider_Rainy_Night_Delivery__montage.jpg", "rpt_f3_fftformer_vs_supir.jpg")
    cp(R / "final_supir" / "08_KFC_Rider_Rainy_Night_Delivery" / "subject_compare_4096.jpg", "rpt_f4_crop_vs_whole.jpg")
    cp(SUB / "08_RAW_vs_FINAL.jpg", "rpt_f5_result_08.jpg")
    cp(SUB / "05_RAW_vs_FINAL.jpg", "rpt_f6_result_05.jpg")
    cp(R / "wow" / "graded" / "08_KFC_before_after.jpg", "rpt_f7_grade.jpg")
    cp(R / "wow" / "15_Photographer_Reflected_In_Night_Glass" / "whole_final.png", "rpt_f9_face_fail.png")
    difftsr_fig()
    print("\n[done] figures in", FIG)


if __name__ == "__main__":
    main()
