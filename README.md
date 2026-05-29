# 影像處理 Term Project — 夜間運動模糊影像修復

NYCU 影像處理 2026 春 · Term Project

**組員**：113950011　鄭名翔　·　[id]　[teammate]　·　[id]　[teammate]

---

## 任務

題目給定 15 張夜間或低光環境下拍攝、含有 motion blur 的真實照片，要求從近年發表的開源 deblurring 方法中挑選並改進，最終繳交 2 張視覺最佳的修復結果。

挑戰：解析度落在 6K~8K、低光 + noise + 大尺度 blur kernel、無 ground truth（不能用 PSNR / SSIM）。

---

## 方法

主力是 **FFTformer**（CVPR 2023，RealBlur-J pretrained）。在它之上我們設計了一條手刻的 preprocessing pipeline，稱為 **Pipe A v2**：

```
Input
  → gamma 0.75 (per-image)
  → CLAHE clip = 5.0, grid = 8×8
  → HSV saturation × 1.2
  → resize to max-side = 1024
  → FFTformer (RealBlur-J pretrained)
  → resize back to native
  → unsharp mask (radius = 2, percent = 200)
Output
```

設計哲學：**把 out-of-distribution（OOD）的夜間影像分布推回 FFTformer 的訓練分布**。整套 pipeline 不需要 training data、不更新權重，純粹用 classical image processing 達成 distribution shift 的效果。

針對個別影像的瓶頸再加上 **Saturation-Masked Sharpening**——用 HSV 飽和度作為自動 spatial mask，只在高彩度區（紅色 logo、霓虹文字）追加 unsharp，避免在暗區放大 noise。

---

## 主要結果

15 張影像的 4 個 NR-IQA 指標平均（NIQE / MUSIQ / MANIQA / NRQM）：

| Method | MUSIQ ↑ | Δ vs input |
|---|---|---|
| Input（原圖） | 28.22 | — |
| FFTformer 單獨 | 25.38 | −2.84 |
| Chain（DarkIR → FFTformer） | 24.95 | −3.27 |
| **Pipe A v2**（本方法） | **33.74** | **+5.52** |

**Chain control experiment**：把 DarkIR 這種 trained low-light enhancement 放在 FFTformer 前，MUSIQ 反而從 25.38 *下降* 到 24.95；而手刻 Pipe A v2 卻能把同一個 FFTformer 從 25.38 *拉升* 到 33.74。**Handcrafted preprocessing 優於 trained preprocessing**。

### Plan I+ v2 繳交版

| 影像 | 設定 | MUSIQ | Δ vs 原圖 |
|---|---|---|---|
| **09 White Truck** | γ=1.0 + Pipe A v2 + masked text sharpen | **36.41** | **+11.39**（單張最大）|
| **08 KFC Rider** | γ=0.70 + Pipe A v2 + saturation 1.25 | 32.32 | +3.14 |

繳交檔放在 `final_submissions/I_plus_pipeA_strengthened/`。

---

## Repo 結構

```
.
├── README.md                       本檔案
├── report/
│   ├── Report_full.md              書面報告（markdown source）
│   ├── Report_full.pdf             書面報告（PDF，13 pages）
│   ├── PPT_outline.md              簡報大綱
│   ├── PPT_script.md               簡報講稿
│   ├── iqa_table.md                完整 IQA 數據與解讀
│   ├── iqa_*.csv                   per-image 量化分數
│   └── figures/                    所有報告用圖檔
├── final_submissions/
│   └── I_plus_pipeA_strengthened/  Plan I+ v2 繳交檔（08 + 09）
├── compare/v1_vs_v2/               v1 vs v2 native-resolution crops
├── preprocess.py                   gamma + CLAHE + saturation
├── masked_sharpen.py               saturation-masked unsharp
├── unsharp.py                      基本 unsharp
├── run_pipea_v2.py                 Pipe A v2 主管線
├── run_ablation.py                 ingredient ablation runner
├── run_clahe_sweep.py              CLAHE clip parameter sweep
├── run_quick_variants.py           unsharp sweep + 09 γ=1.0
├── compute_iqa.py                  主 IQA scoring
├── compute_iqa_ablation.py         ablation IQA
├── compute_iqa_tier1.py            sweep IQA
├── plot_iqa.py                     bar chart 視覺化
├── plot_pipeline.py                Pipe A pipeline diagram
├── build_plans.py                  生成繳交資料夾
└── md_to_pdf.py                    markdown → PDF（headless Chrome）
```

不放進 repo 但需要的（由 `.gitignore` 排除）：

- `Images/`：15 張原始影像（題目提供）
- `results/`：所有方法的 inference 輸出（~3 GB，可由腳本重跑）
- `FFTformer/` / `MISCFilter/` / `DarkIR/`：upstream model code，請到原 repo 取得
- `Term Project.pdf`：作業規格（課程材料）

---

## 復現流程

### Environment

NVIDIA GPU（Blackwell sm_120 需要 PyTorch 2.11.0+cu128）+ cupy 13.x。

```bash
conda create -n deblur python=3.11
conda activate deblur
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install pyiqa pymupdf cupy-cuda12x basicsr matplotlib pillow opencv-python markdown
```

幾個相容性 patch 要踩：
- `basicsr` 需要 monkey-patch（移除已被廢棄的 `torchvision.transforms.functional_tensor` import）
- FFTformer 用 `importlib.util.spec_from_file_location` 繞過破損的 `basicsr` import chain
- MISCFilter 的 inline CUDA kernel 要改寫成 cupy 12+ 的 `RawModule` API

### Model weights

- **FFTformer**：[`kkkls/FFTformer`](https://github.com/kkkls/FFTformer)，使用 `pretrain_model/Realblur/net_g_Realblur_J.pth`
- **DarkIR**：[`cidautai/DarkIR`](https://github.com/cidautai/DarkIR)（ablation 用）
- **MISCFilter**：[`ChengxuLiu/MISCFilter`](https://github.com/ChengxuLiu/MISCFilter)（ablation 用）

### 跑主管線（產生繳交檔）

```bash
python run_pipea_v2.py            # Pipe A v2 baseline + 08 / 09 繳交版
python build_plans.py             # 生成 final_submissions/ 內的 compare jpg + PNG
```

### Ablation

```bash
python run_ablation.py            # ingredient ablation（拿掉每個成分）
python compute_iqa_ablation.py    # 評分

python run_clahe_sweep.py         # CLAHE clip sweep
python run_quick_variants.py      # unsharp sweep + 09 γ=1.0 variant
python compute_iqa_tier1.py       # 評分
```

### 主 IQA 表與視覺化

```bash
python compute_iqa.py             # 8 methods × 4 metrics × 15 images
python plot_iqa.py                # bar chart → report/figures/
python plot_pipeline.py           # Pipe A diagram → report/figures/
```

### 生成書面報告 PDF

```bash
python md_to_pdf.py               # report/Report_full.md → Report_full.pdf
```

---

## 限制與未來方向

- **Glass reflection 場景無法處理**（15 號攝影師反射、02 號紅燈反射）——這本質上是 layer separation 問題，需要 reflection removal 類方法
- **極大尺度 shake blur** 仍是 open problem（06 號招牌震動，blur kernel 太大）
- NIQE 與 NRQM 對所有 aggressive processing 都有偏差，這是 NR-IQA 領域的 known limitation

未來方向：用真實 paired night-blur dataset 對 FFTformer 做 few-shot fine-tuning；對 glass reflection 場景整合 deep reflection removal；在更多 deblur backbones 上驗證「handcrafted vs trained preprocessing」的差異。
