"""
Internal sanity tests for spatial_laplacian.py.
Not part of the submission — verifies our manual convolution matches
scipy's reference output on synthetic and random data.
"""
from __future__ import annotations

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spatial_laplacian import (  # noqa: E402
    pad_image_2d,
    convolve2d_gray,
    convolve2d,
    get_laplacian_kernel,
    laplacian_response,
    sharpen,
)


def assert_close(a, b, msg, atol=1e-9):
    if not np.allclose(a, b, atol=atol):
        diff = np.abs(a - b)
        raise AssertionError(
            f"{msg}\n  max|diff|={diff.max()}  shape a={a.shape} b={b.shape}")
    print(f"  OK  {msg}")


# 1) Padding correctness vs. np.pad
def test_padding():
    print("[test] padding")
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(7, 9)).astype(np.float64)
    for ph, pw in [(0, 0), (1, 1), (2, 3), (3, 2)]:
        z_ours = pad_image_2d(img, ph, pw, "zero")
        z_np = np.pad(img, ((ph, ph), (pw, pw)), mode="constant")
        assert_close(z_ours, z_np, f"zero pad ph={ph} pw={pw}")

        r_ours = pad_image_2d(img, ph, pw, "replicate")
        r_np = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
        assert_close(r_ours, r_np, f"replicate pad ph={ph} pw={pw}")


# 2) Convolution correctness vs. scipy.signal.convolve2d (used only for testing)
def test_convolution():
    print("[test] 2D convolution vs scipy")
    try:
        from scipy.signal import convolve2d as sp_conv
    except Exception as e:
        print(f"  SKIP (scipy unavailable: {e})")
        return
    rng = np.random.default_rng(1)
    img = rng.standard_normal((20, 25)) * 100 + 128
    kernel = rng.standard_normal((3, 3))

    # zero padding -> scipy 'same' with fillvalue=0
    ours = convolve2d_gray(img, kernel, padding="zero")
    ref = sp_conv(img, kernel, mode="same", boundary="fill", fillvalue=0)
    assert_close(ours, ref, "zero padding 3x3")

    # replicate padding -> boundary='symm' is symmetric, not edge.
    # scipy doesn't have edge replicate; manually pad then 'valid' convolve.
    pad = 1
    padded = np.pad(img, pad, mode="edge")
    ref_rep = sp_conv(padded, kernel, mode="valid")
    ours_rep = convolve2d_gray(img, kernel, padding="replicate")
    assert_close(ours_rep, ref_rep, "replicate padding 3x3")

    # 5x5 kernel
    k5 = rng.standard_normal((5, 5))
    ours5 = convolve2d_gray(img, k5, padding="zero")
    ref5 = sp_conv(img, k5, mode="same", boundary="fill", fillvalue=0)
    assert_close(ours5, ref5, "zero padding 5x5")


# 3) Laplacian kernels
def test_laplacian_kernel():
    print("[test] laplacian kernels")
    k4 = get_laplacian_kernel(4, positive_center=True)
    k8 = get_laplacian_kernel(8, positive_center=True)
    expected_k4 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=float)
    expected_k8 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float)
    assert_close(k4, expected_k4, "4-neighbor positive-center")
    assert_close(k8, expected_k8, "8-neighbor positive-center")
    # zero-sum (DC = 0)
    assert abs(k4.sum()) < 1e-12 and abs(k8.sum()) < 1e-12, "Laplacian sums to 0"
    print("  OK  kernels sum to zero")


# 4) On a flat region the response is zero (DC kills Laplacian)
def test_flat_region():
    print("[test] flat region -> zero response")
    img = np.full((30, 40), 100.0)
    for n in (4, 8):
        for pad in ("zero", "replicate"):
            r = laplacian_response(img, neighbor=n, padding=pad)
            if pad == "replicate":
                assert_close(r, np.zeros_like(r),
                             f"flat n={n} pad={pad}")
            else:
                # zero padding: only borders are nonzero (boundary pulls toward 0)
                interior = r[1:-1, 1:-1]
                assert_close(interior, np.zeros_like(interior),
                             f"flat interior n={n} pad={pad}")


# 5) Impulse response = kernel (for replicate padding, no boundary effect at center)
def test_impulse():
    print("[test] impulse response equals kernel")
    img = np.zeros((11, 11))
    img[5, 5] = 1.0
    for n in (4, 8):
        r = laplacian_response(img, neighbor=n, padding="zero")
        k = get_laplacian_kernel(n, positive_center=True)
        # Convolution with delta -> kernel centered at impulse
        # (because we flipped the kernel, true convolution).
        center_patch = r[4:7, 4:7]
        assert_close(center_patch, k, f"impulse n={n}")


# 6) Color-image path runs and matches per-channel application
def test_color_path():
    print("[test] 3D image == per-channel 2D")
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(15, 18, 3)).astype(np.float64)
    out_3d = laplacian_response(img, 4, "replicate")
    for c in range(3):
        out_c = laplacian_response(img[..., c], 4, "replicate")
        assert_close(out_3d[..., c], out_c, f"channel {c}")


# 7) Sharpening on a flat region returns the original
def test_sharpen_flat():
    print("[test] sharpen on flat region == identity")
    img = np.full((20, 20), 123.0)
    out = sharpen(img, neighbor=4, padding="replicate", c=1.0)
    assert_close(out, img, "flat sharpen identity")


def main():
    test_padding()
    test_convolution()
    test_laplacian_kernel()
    test_flat_region()
    test_impulse()
    test_color_path()
    test_sharpen_flat()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
