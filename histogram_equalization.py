import os
import numpy as np
import matplotlib.pyplot as plt
from utils import load_gray, save_img, calc_histogram, plot_histogram


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    hist = calc_histogram(img)
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0].min()
    total_px = img.size
    mapping = np.round((cdf - cdf_min) / (total_px - cdf_min) * 255).astype(np.uint8)
    return mapping[img]


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    img = load_gray("Q1.jpg")
    img_eq = histogram_equalization(img)
    save_img("output/Q1_equalized.jpg", img_eq)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Q1 Histogram Equalization", fontsize=14)

    axes[0, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original"); axes[0, 0].axis("off")
    axes[0, 1].imshow(img_eq, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Equalized"); axes[0, 1].axis("off")
    plot_histogram(calc_histogram(img), "Histogram (before)", axes[1, 0])
    plot_histogram(calc_histogram(img_eq), "Histogram (after)", axes[1, 1])

    plt.tight_layout()
    plt.savefig("output/Q1_result.png", dpi=150)
    plt.close()