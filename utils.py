import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"找不到影像：{path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def save_img(path: str, img: np.ndarray) -> None:
    cv2.imwrite(path, img)
    print(f"已儲存：{path}")


def calc_histogram(img: np.ndarray) -> np.ndarray:
    hist = np.zeros(256, dtype=np.int64)
    for val in range(256):
        hist[val] = np.sum(img == val)
    return hist


def plot_histogram(hist: np.ndarray, title: str, ax: plt.Axes) -> None:
    ax.bar(np.arange(256), hist, width=1, color="steelblue", edgecolor="none")
    ax.set_title(title)
    ax.set_xlim([0, 255])
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Count")