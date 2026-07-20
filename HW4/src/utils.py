"""
Shared I/O / visualization helpers for HW4.

Notes
-----
* Images are loaded as float64 in [0, 255] for safe arithmetic.
* `save_image` clips to [0, 255] and writes uint8 (used for final results).
* `normalize_uint8_signed` is for visualizing signed responses (e.g. the
  Laplacian itself), mapping 0 -> mid-gray so positive/negative edges read
  as bright/dark.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_image(path: str, as_gray: bool = False) -> np.ndarray:
    """Load an image as float64 in [0, 255].

    Parameters
    ----------
    path : str
        Image file path.
    as_gray : bool
        If True, force-convert to single-channel.
    """
    img = Image.open(path)
    if as_gray:
        img = img.convert("L")
    else:
        # Drop alpha if present; keep grayscale grayscale.
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float64)
    return arr


def save_image(arr: np.ndarray, path: str) -> None:
    """Clip to [0, 255], cast to uint8, write file."""
    a = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(a).save(path)


def normalize_uint8_minmax(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 255] uint8."""
    a = arr.astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros(a.shape, dtype=np.uint8)
    return (((a - lo) / (hi - lo)) * 255.0).astype(np.uint8)


def normalize_uint8_signed(arr: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Map a signed response so that 0 -> 128 (gray), preserving sign info.

    Uses a robust scale (percentile of |values|) instead of the absolute max,
    so a handful of spike-edge pixels in natural photos don't compress the
    bulk of the image to a flat gray.
    """
    a = arr.astype(np.float64)
    m = float(np.percentile(np.abs(a), percentile))
    if m < 1e-12:
        return np.full(a.shape, 128, dtype=np.uint8)
    return np.clip((a / m) * 127.5 + 127.5, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_images(directory: str,
                exts: Iterable[str] = (".jpg", ".jpeg", ".jfif", ".png",
                                       ".bmp", ".tif", ".tiff",
                                       ".avif", ".webp")) -> list[str]:
    """List image files in `directory` sorted alphabetically."""
    if not os.path.isdir(directory):
        return []
    files = []
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith(tuple(exts)):
            files.append(os.path.join(directory, f))
    return files
