# Introduction to Image Processing — NYCU, Spring 2026

國立陽明交通大學「影像處理概論」課程作業與專題集

## Course Info ｜ 課程資訊

- **Institution:** National Yang Ming Chiao Tung University (NYCU 國立陽明交通大學)
- **Semester:** Spring 2026（114學年度第2學期）
- **Course:** Introduction to Image Processing 影像處理概論（undergraduate course, 資訊共同）
- **Instructor:** 陳冠文 Kuan-Wen Chen
- **Field:** Image Processing / Computer Vision

## Overview

This repository collects two homework assignments and one term project from
the NYCU Introduction to Image Processing undergraduate course. The homework
assignments implement classic spatial- and frequency-domain techniques from
scratch with NumPy: histogram equalization and specification, mean/median
spatial filtering, and Laplacian sharpening compared side-by-side in both the
spatial and FFT-based frequency domains. The term project tackles a harder,
open-ended restoration problem — recovering recognizable faces from real
nighttime photos degraded by motion blur — by combining a frequency-domain
deblurring transformer with a diffusion-based generative face prior.
本作業集涵蓋直方圖均衡化與規定化、均值／中值空間濾波、拉普拉斯銳化的空間域與頻率域比較等從零實作的
影像處理核心技術，並以「夜間運動模糊人臉修復」期末專題延伸至生成式先驗輔助的影像復原。

## Contents ｜ 內容

| Folder | Topic | Summary |
|--------|-------|---------|
| [HW2](HW2/) | Histogram & Spatial Filtering（直方圖與空間濾波）| Histogram equalization (CDF-based intensity remapping) and histogram specification (CDF matching via interpolation) for contrast enhancement and tone transfer, plus mean/median spatial filtering (3×3, 5×5) compared for noise removal versus edge preservation — all implemented from scratch with NumPy. Result plots in [`HW2/output/`](HW2/output/). |
| [HW4](HW4/) | Image Sharpening: Spatial & Frequency Domain Laplacian（影像銳化：空間域與頻率域拉普拉斯）| Laplacian sharpening implemented in two equivalent paths: a hand-written spatial-domain convolution (4-neighbor / 8-neighbor / 5×5 LoG kernels, zero and replicate padding) and a frequency-domain FFT pipeline, with a discussion comparing linear vs. cyclic convolution and computational complexity. Result grids in [`HW4/results/`](HW4/results/). |
| [Project](Project/) | Term Project: Nighttime Motion-Blur Face Restoration（期末專題：夜間運動模糊人臉修復）| Restoration of faces from real nighttime, low-light photos with large-scale motion blur: a frequency-domain deblurring transformer (FFTformer) provides structural recovery, followed by a diffusion-based generative face prior (SUPIR) to recover high-frequency facial detail, composited back with feathering; includes an explicit fidelity statement distinguishing recovered structure from generated detail. Submitted results in [`Project/final_submissions/`](Project/final_submissions/). |

## Notes

Each folder is self-contained and keeps its own original README with exact
setup instructions, dependencies, and how to run each part — please refer to
the subfolder README before running any code. Commit history for each
assignment/project has been preserved from its original individual repository.
