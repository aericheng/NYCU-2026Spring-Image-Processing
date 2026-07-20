"""
HW4 Task 2 - Frequency Domain Laplacian Sharpening.

Pipeline (per the spec)
-----------------------
    1. F = fft2(f);  F_shift = fftshift(F)             # shift zero-freq to center
    2. Build H(u,v) on the centered M x N frequency grid
    3. G_shift = H * F_shift                            # filter the spectrum
    4. g = real(ifft2(ifftshift(G_shift)))             # ifftshift, ifft, take real
    5. sharpened = f + c * g                            # combine with original

The spec explicitly allows numpy.fft for this task.

Outputs in ../results/frequency/:
    <name>_original.png
    <name>_spectrum.png            log(1 + |F_shift|)
    <name>_filter.png              the filter H(u,v) (visualised)
    <name>_filtered_spectrum.png   log(1 + |H * F_shift|)
    <name>_laplacian_freq.png      signed Laplacian response (0 -> mid gray)
    <name>_sharpened_freq.png      g = f + c * Laplacian(f)
    <name>_comparison.png          2x3 grid summary for the report
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# UTF-8 stdout for CJK paths on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (  # noqa: E402
    load_image,
    save_image,
    normalize_uint8_signed,
    normalize_uint8_minmax,
    ensure_dir,
    list_images,
)


# ---------------------------------------------------------------------------
# 1. Build the Laplacian filter H(u,v)
# ---------------------------------------------------------------------------
def build_laplacian_filter(M: int, N: int,
                           positive_center: bool = True,
                           normalize: bool = True) -> np.ndarray:
    """Construct H(u,v) on the *centered* (post-fftshift) M x N grid.

    Mathematical Laplacian (Gonzalez & Woods, eq. 4.9-4):
        H_lap(u, v) = -4 pi^2 * D^2(u, v)
        D^2(u, v)   = (u - M/2)^2 + (v - N/2)^2

    `positive_center=True` returns the *negation* (+4 pi^2 D^2). Combined with
    the sharpening rule  g = f + c * IFFT(H * F)  this matches the spatial
    sharpening of Task 1 sign-for-sign (Task 1 uses positive-center kernels).

    `normalize=True` divides by max|H| so H lies in [0, 1] (or [-1, 0]).
    Without this step |H| grows like ~M^2 + N^2 and `c` would have to be
    rescaled per image. Normalised filters are equivalent up to a global
    constant absorbed into c.
    """
    u = np.arange(M, dtype=np.float64) - M / 2.0
    v = np.arange(N, dtype=np.float64) - N / 2.0
    U, V = np.meshgrid(u, v, indexing="ij")
    D2 = U * U + V * V
    H = 4.0 * (np.pi ** 2) * D2          # +4 pi^2 D^2  (positive center)
    if not positive_center:
        H = -H
    if normalize:
        m = float(np.max(np.abs(H)))
        if m > 0.0:
            H = H / m
    return H


# ---------------------------------------------------------------------------
# 2. Apply filter via FFT
# ---------------------------------------------------------------------------
def _filter_gray(img: np.ndarray, H: np.ndarray):
    """Filter one channel. Returns (response, F_shift, G_shift)."""
    f = img.astype(np.float64)
    F = np.fft.fft2(f)
    F_shift = np.fft.fftshift(F)
    G_shift = H * F_shift
    G = np.fft.ifftshift(G_shift)
    g = np.fft.ifft2(G)
    return np.real(g), F_shift, G_shift


def laplacian_response_freq(img: np.ndarray,
                            normalize_filter: bool = True) -> np.ndarray:
    """Frequency-domain Laplacian response for 2D or per-channel 3D image."""
    if img.ndim == 2:
        H = build_laplacian_filter(*img.shape,
                                   positive_center=True,
                                   normalize=normalize_filter)
        return _filter_gray(img, H)[0]
    if img.ndim == 3:
        h, w, c = img.shape
        H = build_laplacian_filter(h, w,
                                   positive_center=True,
                                   normalize=normalize_filter)
        chans = [_filter_gray(img[..., k], H)[0] for k in range(c)]
        return np.stack(chans, axis=-1)
    raise ValueError(f"Unsupported ndim: {img.ndim}")


def sharpen_freq(img: np.ndarray, c: float = 1.0,
                 normalize_filter: bool = True) -> np.ndarray:
    """g = f + c * Laplacian_freq(f). Caller clips/casts to uint8."""
    return img.astype(np.float64) + c * laplacian_response_freq(
        img, normalize_filter=normalize_filter)


# ---------------------------------------------------------------------------
# 3. Driver
# ---------------------------------------------------------------------------
def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def _spectrum_log_uint8(F_shift: np.ndarray) -> np.ndarray:
    """log(1 + |F|) -> uint8 (mean across channels for color)."""
    if F_shift.ndim == 3:
        mag = np.mean(np.abs(F_shift), axis=-1)
    else:
        mag = np.abs(F_shift)
    return normalize_uint8_minmax(np.log1p(mag))


def process_image(image_path: str, output_dir: str,
                  c: float = 1.0, normalize_filter: bool = True) -> None:
    name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n=== Processing {name} ===")

    img = load_image(image_path, as_gray=False)
    is_gray = (img.ndim == 2)
    print(f"  Loaded shape={img.shape}, is_gray={is_gray}")

    save_image(img, os.path.join(output_dir, f"{name}_original.png"))

    H = build_laplacian_filter(*img.shape[:2],
                               positive_center=True,
                               normalize=normalize_filter)

    t0 = time.perf_counter()
    if is_gray:
        resp, F_shift, G_shift = _filter_gray(img, H)
    else:
        resp_chs, F_chs, G_chs = [], [], []
        for k in range(img.shape[2]):
            r, fs, gs = _filter_gray(img[..., k], H)
            resp_chs.append(r); F_chs.append(fs); G_chs.append(gs)
        resp = np.stack(resp_chs, axis=-1)
        F_shift = np.stack(F_chs, axis=-1)
        G_shift = np.stack(G_chs, axis=-1)
    t1 = time.perf_counter()
    sharp = img.astype(np.float64) + c * resp
    t2 = time.perf_counter()
    print(f"  freq laplacian: {t1 - t0:.3f}s  sharpen: {t2 - t1:.3f}s")

    # Save artifacts
    spec_vis = _spectrum_log_uint8(F_shift)
    save_image(spec_vis, os.path.join(output_dir, f"{name}_spectrum.png"))

    fil_vis = normalize_uint8_minmax(np.abs(H))
    save_image(fil_vis, os.path.join(output_dir, f"{name}_filter.png"))

    fspec_vis = _spectrum_log_uint8(G_shift)
    save_image(fspec_vis, os.path.join(output_dir, f"{name}_filtered_spectrum.png"))

    resp_for_vis = resp if is_gray else np.mean(resp, axis=-1)
    resp_vis = normalize_uint8_signed(resp_for_vis)
    save_image(resp_vis, os.path.join(output_dir, f"{name}_laplacian_freq.png"))

    save_image(sharp, os.path.join(output_dir, f"{name}_sharpened_freq.png"))

    # Comparison grid (2x3): the report showpiece.
    cmap = "gray" if is_gray else None
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes[0, 0].imshow(_to_uint8(img), cmap=cmap)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(spec_vis, cmap="gray")
    axes[0, 1].set_title("Spectrum  log(1+|F|)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(fil_vis, cmap="gray")
    axes[0, 2].set_title("Laplacian filter |H(u,v)|")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(fspec_vis, cmap="gray")
    axes[1, 0].set_title("Filtered spectrum  log(1+|H*F|)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(resp_vis, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Laplacian response (spatial)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(_to_uint8(sharp), cmap=cmap)
    axes[1, 2].set_title(f"Sharpened   g = f + {c} * response")
    axes[1, 2].axis("off")

    fig.suptitle(f"Frequency Domain Sharpening - {name}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(output_dir, f"{name}_comparison.png"), dpi=120)
    plt.close(fig)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.abspath(os.path.join(here, "..", "images"))
    output_dir = os.path.abspath(os.path.join(here, "..", "results", "frequency"))
    ensure_dir(output_dir)

    files = list_images(images_dir)
    if not files:
        print(f"[!] No images found in {images_dir}")
        print("    Drop a few test images in there (jpg/png/bmp) and re-run.")
        return 1

    print(f"Images dir: {images_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Found {len(files)} image(s).")

    for f in files:
        process_image(f, output_dir, c=30.0, normalize_filter=True)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
