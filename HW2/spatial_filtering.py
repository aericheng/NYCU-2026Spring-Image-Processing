import os
import numpy as np
import matplotlib.pyplot as plt
from utils import load_gray, save_img


def apply_filter(img: np.ndarray, ksize: int, mode: str = "mean") -> np.ndarray:
    pad = ksize // 2
    padded = np.pad(img, pad, mode="edge")
    out = np.zeros_like(img, dtype=np.uint8)

    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            region = padded[r:r + ksize, c:c + ksize]
            if mode == "mean":
                out[r, c] = np.clip(np.round(np.mean(region)), 0, 255)
            else:
                out[r, c] = np.median(region)
    return out


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    img = load_gray("Q3.jpg")

    configs = [
        ("Mean 3x3",   3, "mean"),
        ("Mean 5x5",   5, "mean"),
        ("Median 3x3", 3, "median"),
        ("Median 5x5", 5, "median"),
    ]

    results = {}
    for name, ksize, mode in configs:
        print(f"  {name}...")
        results[name] = apply_filter(img, ksize, mode)
        save_img(f"output/Q3_{name.replace(' ', '_')}.jpg", results[name])

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Q3 Spatial Filtering Comparison", fontsize=14)
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
    for ax, (name, res) in zip(axes[1:], results.items()):
        ax.imshow(res, cmap="gray"); ax.set_title(name); ax.axis("off")
    plt.tight_layout()
    plt.savefig("output/Q3_comparison.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Q3 Mean vs Median – Difference", fontsize=14)
    for row, (ksize_str, mean_k, median_k) in enumerate([
        ("3×3", "Mean 3x3", "Median 3x3"),
        ("5×5", "Mean 5x5", "Median 5x5"),
    ]):
        diff = np.abs(
            results[mean_k].astype(np.int16) - results[median_k].astype(np.int16)
        ).astype(np.uint8)
        axes[row, 0].imshow(results[mean_k], cmap="gray"); axes[row, 0].set_title(f"Mean {ksize_str}"); axes[row, 0].axis("off")
        axes[row, 1].imshow(results[median_k], cmap="gray"); axes[row, 1].set_title(f"Median {ksize_str}"); axes[row, 1].axis("off")
        axes[row, 2].imshow(diff, cmap="hot"); axes[row, 2].set_title(f"Difference {ksize_str}"); axes[row, 2].axis("off")
    plt.tight_layout()
    plt.savefig("output/Q3_diff.png", dpi=150)
    plt.close()