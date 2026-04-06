import os
import numpy as np
import matplotlib.pyplot as plt
from utils import load_gray, save_img, calc_histogram


def histogram_specification(src: np.ndarray, ref: np.ndarray):
    def get_cdf(img):
        hist = calc_histogram(img)
        cdf  = np.cumsum(hist).astype(np.float64)
        cdf /= cdf[-1]
        return cdf

    src_cdf = get_cdf(src)
    ref_cdf = get_cdf(ref)
    mapping = np.interp(src_cdf, ref_cdf, np.arange(256)).astype(np.uint8)
    return mapping[src], src_cdf, ref_cdf, mapping


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    src = load_gray("Q2_src.jpg")
    ref = load_gray("Q2_ref.jpg")

    out, src_cdf, ref_cdf, mapping = histogram_specification(src, ref)
    save_img("output/Q2_output.jpg", out)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Q2 Histogram Specification – Images", fontsize=14)
    axes[0].imshow(src, cmap="gray", vmin=0, vmax=255); axes[0].set_title("Source");    axes[0].axis("off")
    axes[1].imshow(ref, cmap="gray", vmin=0, vmax=255); axes[1].set_title("Reference"); axes[1].axis("off")
    axes[2].imshow(out, cmap="gray", vmin=0, vmax=255); axes[2].set_title("Output");    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig("output/Q2_images.png", dpi=150)
    plt.close()


    x = np.arange(256)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Q2 Histogram Specification – CDF & Mapping", fontsize=14)

    axes[0].plot(x, src_cdf, color="steelblue")
    axes[0].set_title("Source CDF")
    axes[0].set_xlabel("Pixel value"); axes[0].set_ylabel("CDF")
    axes[0].set_xlim([0, 255]); axes[0].set_ylim([0, 1])

    axes[1].plot(x, ref_cdf, color="darkorange")
    axes[1].set_title("Reference CDF")
    axes[1].set_xlabel("Pixel value"); axes[1].set_ylabel("CDF")
    axes[1].set_xlim([0, 255]); axes[1].set_ylim([0, 1])

    axes[2].plot(x, mapping, color="green")
    axes[2].set_title("Mapping Curve (src → ref)")
    axes[2].set_xlabel("Input (src pixel value)")
    axes[2].set_ylabel("Output (ref pixel value)")
    axes[2].set_xlim([0, 255]); axes[2].set_ylim([0, 255])

    plt.tight_layout()
    plt.savefig("output/Q2_curves.png", dpi=150)
    plt.close()


    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Q2 Histogram Specification – Histograms", fontsize=14)
    for ax, img, title, color in zip(
        axes,
        [src, ref, out],
        ["Source Histogram", "Reference Histogram", "Output Histogram"],
        ["steelblue", "darkorange", "green"],
    ):
        ax.bar(x, calc_histogram(img), width=1, color=color, edgecolor="none")
        ax.set_title(title)
        ax.set_xlabel("Pixel value"); ax.set_ylabel("Count")
        ax.set_xlim([0, 255])
    plt.tight_layout()
    plt.savefig("output/Q2_histograms.png", dpi=150)
    plt.close()