"""
HW4 Task 1 - Spatial Domain Laplacian Sharpening.

Restrictions (from spec)
------------------------
* Do NOT use built-in convolution helpers
  (e.g. cv2.filter2D, scipy.signal.convolve2d).
* Implement convolution + boundary handling from scratch with basic
  matrix operations only.

What this script does
---------------------
For every image in ../images/, run four configurations
    {4-neighbor, 8-neighbor} x {zero-padding, replicate-padding}
and write to ../results/spatial/:
    <name>_original.png
    <name>_laplacian_<n>n_<pad>.png   (signed response, 0 -> mid gray)
    <name>_sharpened_<n>n_<pad>.png   (g = f + Laplacian(f))
    <name>_comparison.png             (matplotlib grid for the report)

Run from anywhere:
    python src/spatial_laplacian.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# Make stdout UTF-8 so paths containing CJK characters print on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

# Allow running both as a module and as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (  # noqa: E402
    load_image,
    save_image,
    normalize_uint8_signed,
    ensure_dir,
    list_images,
)


# ---------------------------------------------------------------------------
# 1. Boundary handling (manual, no library padding)
# ---------------------------------------------------------------------------
def pad_image_2d(img: np.ndarray, pad_h: int, pad_w: int,
                 mode: str = "zero") -> np.ndarray:
    """Pad a 2D array.

    mode:
        'zero'      -> outside pixels are 0
        'replicate' -> outside pixels copy the nearest border pixel
    """
    if img.ndim != 2:
        raise ValueError(f"pad_image_2d expects 2D array, got ndim={img.ndim}")
    H, W = img.shape
    out = np.zeros((H + 2 * pad_h, W + 2 * pad_w), dtype=np.float64)
    out[pad_h:pad_h + H, pad_w:pad_w + W] = img

    if mode == "zero":
        return out

    if mode == "replicate":
        if pad_h > 0:
            out[:pad_h, pad_w:pad_w + W] = img[0:1, :]              # top
            out[pad_h + H:, pad_w:pad_w + W] = img[-1:, :]          # bottom
        if pad_w > 0:
            out[pad_h:pad_h + H, :pad_w] = img[:, 0:1]              # left
            out[pad_h:pad_h + H, pad_w + W:] = img[:, -1:]          # right
        if pad_h > 0 and pad_w > 0:
            out[:pad_h, :pad_w] = img[0, 0]                         # TL
            out[:pad_h, pad_w + W:] = img[0, -1]                    # TR
            out[pad_h + H:, :pad_w] = img[-1, 0]                    # BL
            out[pad_h + H:, pad_w + W:] = img[-1, -1]               # BR
        return out

    raise ValueError(f"Unknown padding mode: {mode!r}")


def pad_image(img: np.ndarray, pad_h: int, pad_w: int,
              mode: str = "zero") -> np.ndarray:
    """Pad a 2D (HxW) or 3D (HxWxC) image. Each channel is padded independently."""
    if img.ndim == 2:
        return pad_image_2d(img, pad_h, pad_w, mode)
    if img.ndim == 3:
        chans = [pad_image_2d(img[..., c], pad_h, pad_w, mode)
                 for c in range(img.shape[2])]
        return np.stack(chans, axis=-1)
    raise ValueError(f"Unsupported ndim: {img.ndim}")


# ---------------------------------------------------------------------------
# 2. Manual 2D convolution
# ---------------------------------------------------------------------------
def convolve2d_gray(img: np.ndarray, kernel: np.ndarray,
                    padding: str = "zero") -> np.ndarray:
    """2D convolution on a single-channel image.

    Implementation
    --------------
    Vectorized over pixels but loops over the (small) kernel positions.
    For a k x k kernel this is k*k full-array additions. We flip the
    kernel so this is true convolution; for the symmetric Laplacian
    masks below the flip is a no-op but keeps the function general.
    """
    if img.ndim != 2:
        raise ValueError(f"convolve2d_gray expects 2D image, got ndim={img.ndim}")
    if kernel.ndim != 2:
        raise ValueError("kernel must be 2D")

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = pad_image_2d(img.astype(np.float64), pad_h, pad_w, mode=padding)
    H, W = img.shape

    kf = np.asarray(kernel, dtype=np.float64)[::-1, ::-1]  # convolution flip
    out = np.zeros((H, W), dtype=np.float64)

    for i in range(kh):
        for j in range(kw):
            w = kf[i, j]
            if w != 0.0:
                out += w * padded[i:i + H, j:j + W]
    return out


def convolve2d(img: np.ndarray, kernel: np.ndarray,
               padding: str = "zero") -> np.ndarray:
    """Convolve a 2D or 3D (per-channel) image. Output keeps input shape."""
    if img.ndim == 2:
        return convolve2d_gray(img, kernel, padding)
    if img.ndim == 3:
        chans = [convolve2d_gray(img[..., c], kernel, padding)
                 for c in range(img.shape[2])]
        return np.stack(chans, axis=-1)
    raise ValueError(f"Unsupported ndim: {img.ndim}")


# ---------------------------------------------------------------------------
# 3. Laplacian kernels
# ---------------------------------------------------------------------------
def get_laplacian_kernel(neighbor: int = 4,
                         positive_center: bool = True) -> np.ndarray:
    """Return a 3x3 Laplacian mask.

    Mathematical Laplacian (negative center):
        4-nbr:   [[ 0, 1, 0], [ 1,-4, 1], [ 0, 1, 0]]
        8-nbr:   [[ 1, 1, 1], [ 1,-8, 1], [ 1, 1, 1]]

    Negated form (positive center) lets sharpening be written as
        g = f + c * response
    instead of the equivalent g = f - c * response.
    """
    if neighbor == 4:
        k = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]], dtype=np.float64)
    elif neighbor == 8:
        k = np.array([[1, 1, 1],
                      [1, -8, 1],
                      [1, 1, 1]], dtype=np.float64)
    else:
        raise ValueError("neighbor must be 4 or 8")
    return -k if positive_center else k


def get_log_kernel_5x5(positive_center: bool = True) -> np.ndarray:
    """Return a 5x5 Laplacian-of-Gaussian (LoG) mask.

    Pre-smooths with a Gaussian then takes the discrete Laplacian, so it is
    less noise-sensitive than the bare 3x3 Laplacian. Kernel sums to 0.
    """
    k = np.array([[ 0,  0,  1,  0,  0],
                  [ 0,  1,  2,  1,  0],
                  [ 1,  2,-16,  2,  1],
                  [ 0,  1,  2,  1,  0],
                  [ 0,  0,  1,  0,  0]], dtype=np.float64)
    return -k if positive_center else k


def get_kernel(kernel_type: str, positive_center: bool = True) -> np.ndarray:
    if kernel_type == "4n":
        return get_laplacian_kernel(4, positive_center=positive_center)
    if kernel_type == "8n":
        return get_laplacian_kernel(8, positive_center=positive_center)
    if kernel_type == "log":
        return get_log_kernel_5x5(positive_center=positive_center)
    raise ValueError(f"Unknown kernel_type: {kernel_type!r}")


# ---------------------------------------------------------------------------
# 4. Laplacian response & sharpening
# ---------------------------------------------------------------------------
def laplacian_response(img: np.ndarray, kernel_type: str = "4n",
                       padding: str = "replicate") -> np.ndarray:
    """Return the Laplacian response using a positive-center kernel."""
    kernel = get_kernel(kernel_type, positive_center=True)
    return convolve2d(img, kernel, padding=padding)


def sharpen(img: np.ndarray, kernel_type: str = "4n",
            padding: str = "replicate", c: float = 1.0) -> np.ndarray:
    """Sharpen with g = f + c * Laplacian(f). Returns float (un-clipped)."""
    return img.astype(np.float64) + c * laplacian_response(
        img, kernel_type=kernel_type, padding=padding)


# ---------------------------------------------------------------------------
# 5. Driver
# ---------------------------------------------------------------------------
def _to_uint8_for_show(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def process_image(image_path: str, output_dir: str,
                  c: float = 1.0) -> None:
    """Run all four configs on a single image and save outputs + comparison."""
    name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n=== Processing {name} ===")

    img = load_image(image_path, as_gray=False)
    is_gray = (img.ndim == 2)
    print(f"  Loaded shape={img.shape}, is_gray={is_gray}")

    save_image(img, os.path.join(output_dir, f"{name}_original.png"))

    # Per-kernel sharpening strength.
    # The textbook 4n / 8n / 5x5-LoG kernels have center magnitudes 4, 8, 16,
    # so an unscaled c=1 makes 8n ~2x and LoG ~4x stronger than 4n. We pick c
    # per kernel so the visible sharpening level is comparable across configs.
    configs = [
        ("4n",  "zero",      1.00),
        ("4n",  "replicate", 1.00),
        ("8n",  "zero",      0.50),
        ("8n",  "replicate", 0.50),
        ("log", "zero",      0.25),
        ("log", "replicate", 0.25),
    ]

    rows = len(configs)
    fig, axes = plt.subplots(rows, 3, figsize=(13, 3.6 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    cmap = "gray" if is_gray else None

    for r, (kernel_type, padding, kc) in enumerate(configs):
        tag = f"{kernel_type}_{padding}"
        print(f"  [{tag} c={kc}]", end=" ")
        t0 = time.perf_counter()
        lap = laplacian_response(img, kernel_type=kernel_type, padding=padding)
        t1 = time.perf_counter()
        sharp = img + kc * lap
        t2 = time.perf_counter()
        print(f"laplacian={t1 - t0:.3f}s  sharpen={t2 - t1:.3f}s")

        lap_for_vis = lap if is_gray else np.mean(lap, axis=-1)
        lap_vis = normalize_uint8_signed(lap_for_vis)
        save_image(lap_vis,
                   os.path.join(output_dir, f"{name}_laplacian_{tag}.png"))
        save_image(sharp,
                   os.path.join(output_dir, f"{name}_sharpened_{tag}.png"))

        axes[r, 0].imshow(_to_uint8_for_show(img), cmap=cmap)
        axes[r, 0].set_title("Original")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(lap_vis, cmap="gray", vmin=0, vmax=255)
        axes[r, 1].set_title(f"Laplacian  ({kernel_type}, {padding})")
        axes[r, 1].axis("off")

        axes[r, 2].imshow(_to_uint8_for_show(sharp), cmap=cmap)
        axes[r, 2].set_title(f"Sharpened  ({kernel_type}, {padding}, c={kc})")
        axes[r, 2].axis("off")

    fig.suptitle(f"Spatial Domain Sharpening - {name}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(output_dir, f"{name}_comparison.png"), dpi=120)
    plt.close(fig)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.abspath(os.path.join(here, "..", "images"))
    output_dir = os.path.abspath(os.path.join(here, "..", "results", "spatial"))
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
        process_image(f, output_dir, c=1.8)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
