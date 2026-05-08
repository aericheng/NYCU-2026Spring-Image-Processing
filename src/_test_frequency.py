"""
Internal sanity tests for frequency_laplacian.py. Not part of the submission.
"""
from __future__ import annotations

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frequency_laplacian import (  # noqa: E402
    build_laplacian_filter,
    laplacian_response_freq,
    sharpen_freq,
)


def assert_close(a, b, msg, atol=1e-9):
    if not np.allclose(a, b, atol=atol):
        diff = np.abs(np.asarray(a) - np.asarray(b))
        raise AssertionError(
            f"{msg}\n  max|diff|={diff.max()}  shape a={np.shape(a)} b={np.shape(b)}")
    print(f"  OK  {msg}")


def test_filter_basic():
    print("[test] H(u,v) construction")
    H = build_laplacian_filter(8, 6, positive_center=True, normalize=False)
    assert H.shape == (8, 6), f"shape {H.shape}"
    assert H.dtype == np.float64
    # Center (u=4, v=3) is DC -> H = 0
    assert abs(H[4, 3]) < 1e-12, f"DC center = {H[4, 3]}"
    # Positive center -> non-DC values are >= 0
    assert (H >= -1e-12).all(), "positive_center should give H >= 0"
    print("  OK  shape, dtype, DC=0, positivity")

    H_neg = build_laplacian_filter(8, 6, positive_center=False, normalize=False)
    assert_close(H_neg, -H, "negation matches positive_center=False")


def test_filter_normalization():
    print("[test] H normalization")
    H = build_laplacian_filter(64, 64, normalize=True)
    m = float(np.max(np.abs(H)))
    assert abs(m - 1.0) < 1e-12, f"max|H|={m}"
    print("  OK  max|H| == 1 after normalize=True")

    H_un = build_laplacian_filter(64, 64, normalize=False)
    assert np.max(np.abs(H_un)) > 1.0
    print("  OK  un-normalized H has |max| > 1")


def test_fft_round_trip():
    print("[test] FFT round trip identity")
    rng = np.random.default_rng(0)
    f = rng.standard_normal((37, 41)) * 100 + 128
    F = np.fft.fft2(f)
    g = np.real(np.fft.ifft2(F))
    assert_close(f, g, "ifft2(fft2(f)) == f", atol=1e-8)


def test_flat_image_zero_response():
    print("[test] flat image -> zero response")
    f = np.full((30, 40), 100.0)
    r = laplacian_response_freq(f)
    assert_close(r, np.zeros_like(r), "flat image response", atol=1e-9)


def test_dc_of_response_is_zero():
    print("[test] DC component of response == 0")
    rng = np.random.default_rng(1)
    f = rng.integers(0, 256, size=(48, 48)).astype(np.float64)
    r = laplacian_response_freq(f)
    # H(0,0) = 0 means the mean of the response should vanish.
    mean = r.mean()
    assert abs(mean) < 1e-9, f"mean={mean}"
    print(f"  OK  mean response = {mean:.3e}")


def test_color_channels_independent():
    print("[test] 3D image == per-channel 2D")
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(20, 25, 3)).astype(np.float64)
    out_3d = laplacian_response_freq(img)
    for c in range(3):
        out_c = laplacian_response_freq(img[..., c])
        assert_close(out_3d[..., c], out_c, f"channel {c}")


def test_sharpen_flat_identity():
    print("[test] sharpen on flat image == identity")
    f = np.full((20, 20), 70.0)
    out = sharpen_freq(f, c=1.0)
    assert_close(out, f, "flat sharpen identity", atol=1e-9)


def test_high_freq_amplified():
    print("[test] high-frequency input is amplified, DC is preserved")
    M = N = 32
    # Image = DC + a single high-frequency cosine
    yy, xx = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
    dc_level = 100.0
    hf = 50.0 * np.cos(np.pi * xx)  # frequency M/2 along x: highest possible
    f = dc_level + hf
    out = sharpen_freq(f, c=1.0, normalize_filter=True)
    # DC should be preserved exactly (filter is 0 at DC).
    assert abs(out.mean() - dc_level) < 1e-9, \
        f"DC drift {out.mean()} vs {dc_level}"
    # High-frequency amplitude should grow (sharpening).
    hf_amp_in = (f - dc_level).std()
    hf_amp_out = (out - dc_level).std()
    assert hf_amp_out > hf_amp_in, \
        f"hf_amp didn't grow: {hf_amp_in} -> {hf_amp_out}"
    print(f"  OK  DC preserved; HF amplitude {hf_amp_in:.2f} -> {hf_amp_out:.2f}")


def test_response_is_real_for_real_input():
    print("[test] response is real-valued for real input")
    rng = np.random.default_rng(4)
    f = rng.standard_normal((33, 29)) * 50 + 128
    r = laplacian_response_freq(f)
    assert r.dtype == np.float64 and not np.iscomplexobj(r)
    print("  OK  response dtype = float64")


def main():
    test_filter_basic()
    test_filter_normalization()
    test_fft_round_trip()
    test_flat_image_zero_response()
    test_dc_of_response_is_zero()
    test_color_channels_independent()
    test_sharpen_flat_identity()
    test_high_freq_amplified()
    test_response_is_real_for_real_input()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
