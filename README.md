# 影像處理 Term Project — 夜間運動模糊影像修復

NYCU 影像處理 2026 春 · 期末專題

**組員**：113950011 鄭名翔　·　[id] [teammate]　·　[id] [teammate]

---

## 任務

針對 15 張夜間或低光環境拍攝、含有運動模糊的真實照片，從近年發表的開源 deblurring 方法中挑選並改進，繳交 2 張視覺最佳的修復結果。

挑戰：6K~8K 解析度、低光 + 雜訊 + 大尺度模糊核、無 ground truth。

---

## 方法概述

主力是 **FFTformer**（CVPR 2023，RealBlur-J 預訓練）。在它之上設計了一條手刻前處理管線 **Pipe A v2**：

```
Input → γ=0.75 → CLAHE clip 5.0 → HSV saturation × 1.2
      → resize max-side 1024 → FFTformer
      → resize 回原尺寸 → unsharp pct 200
Output
```

設計哲學：**把 OOD 的夜間影像分布推回 FFTformer 訓練分布**。整套管線不需要訓練資料、不更新權重，純粹用經典影像處理達成 distribution shift。

針對個別影像的瓶頸加上區域強化：**Saturation-Masked Sharpening**——以 HSV 飽和度為自動空間遮罩，只在高彩度區追加 unsharp，避免在暗區放大雜訊。

---

## 主要結果

15 張影像 × 4 個 NR-IQA 指標（MUSIQ、MANIQA、NRQM、NIQE）平均：

| 方法 | MUSIQ↑ | Δ vs input |
|---|---|---|
| Input（原圖） | 28.22 | — |
| FFTformer 單獨 | 25.38 | −2.84 |
| Chain（DarkIR → FFTformer） | 24.95 | −3.27 |
| **Pipe A v2**（本方法） | **33.74** | **+5.52** |

**Chain 控制組**：訓練過的 DarkIR 放在 FFTformer 前讓 MUSIQ *下降* 0.43；手刻 Pipe A v2 把同一個 FFTformer 拉升 +8.36。**手刻前處理優於訓練式前處理**。

### 繳交版（Plan I+ v2）

| 影像 | 設定 | MUSIQ | Δ vs 原圖 |
|---|---|---|---|
| **09 White Truck** | γ=1.0 + Pipe A v2 + masked text sharpen | **36.41** | **+11.39**（單張最大）|
| **08 KFC Rider** | γ=0.70 + Pipe A v2 + sat 1.25 | 32.32 | +3.14 |

繳交檔位於 `final_submissions/I_plus_pipeA_strengthened/`。

---

## Repo 結構

```
.
├── README.md                       本檔
├── report/
│   ├── Report_full.md              書面報告（markdown source）
│   ├── Report_full.pdf             書面報告（PDF，13 頁）
│   ├── PPT_outline.md              簡報大綱
│   ├── PPT_script.md               簡報講稿
│   ├── iqa_table.md                完整 IQA 數據
│   ├── iqa_*.csv                   per-image 量化分數
│   └── figures/                    所有報告圖檔
├── final_submissions/
│   └── I_plus_pipeA_strengthened/  Plan I+ v2 繳交檔（08 + 09）
├── preprocess.py                   gamma + CLAHE + saturation 前處理
├── masked_sharpen.py               飽和度遮罩 unsharp
├── unsharp.py                      基本 unsharp
├── run_pipea_v2.py                 v2 主管線編排
├── run_ablation.py                 ingredient ablation
├── run_clahe_sweep.py              CLAHE clip 掃描
├── run_quick_variants.py           unsharp 掃描 + 09 γ=1.0
├── compute_iqa.py                  主 IQA 評分
├── compute_iqa_ablation.py         ablation IQA
├── compute_iqa_tier1.py            sweep IQA
├── plot_iqa.py                     bar chart
├── plot_pipeline.py                Pipe A diagram
├── build_plans.py                  生成繳交資料夾
└── md_to_pdf.py                    markdown → PDF
```

不在 repo 但需要的（用 .gitignore 排除）：
- `Images/`：15 張原始影像（題目提供，非公開）
- `results/`：所有方法的輸出（~3 GB，可由腳本重新生成）
- `FFTformer/` / `MISCFilter/` / `DarkIR/`：upstream 模型程式碼，到原 repo 取得
- `Term Project.pdf`：作業規格（課程材料）

---

## 復現流程

### 環境

NVIDIA GPU（Blackwell sm_120 建議用 PyTorch 2.11.0+cu128）+ cupy 13.x。

```bash
conda create -n deblur python=3.11
conda activate deblur
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install pyiqa pymupdf cupy-cuda12x basicsr matplotlib pillow opencv-python
```

basicsr 需要 monkey-patch（移除已被廢棄的 `torchvision.transforms.functional_tensor` import）；FFTformer 用 `importlib.util.spec_from_file_location` 繞過破損的 import chain；MISCFilter 的 inline CUDA kernel 要改寫成 cupy 12+ 的 `RawModule` API。

### 模型

- FFTformer：[`kkkls/FFTformer`](https://github.com/kkkls/FFTformer)，使用 `pretrain_model/Realblur/net_g_Realblur_J.pth` 權重
- DarkIR：[`cidautai/DarkIR`](https://github.com/cidautai/DarkIR)（ablation 用）
- MISCFilter：[`ChengxuLiu/MISCFilter`](https://github.com/ChengxuLiu/MISCFilter)（ablation 用）

### 主管線（產生繳交檔）

```bash
python run_pipea_v2.py            # 跑 Pipe A v2 baseline + 08/09 繳交版
python build_plans.py             # 生成 final_submissions/ 內的 compare jpg + PNG
```

### Ablation

```bash
python run_ablation.py            # ingredient ablation（拿掉每個成分）
python compute_iqa_ablation.py    # 評分

python run_clahe_sweep.py         # CLAHE clip 掃描
python run_quick_variants.py      # unsharp 掃描 + 09 γ=1.0
python compute_iqa_tier1.py       # 評分
```

### 主 IQA 表 + 視覺化

```bash
python compute_iqa.py             # 8 方法 × 4 metric × 15 images
python plot_iqa.py                # bar chart → report/figures/
python plot_pipeline.py           # Pipe A diagram → report/figures/
```

### 報告 PDF

```bash
python md_to_pdf.py               # report/Report_full.md → report/Report_full.pdf
```

---

## 限制與未來方向

- **玻璃反射場景無法處理**（15 號攝影師反射、02 號紅燈反射）——本質是 layer separation 問題，需要 reflection removal 類方法
- **極大尺度 shake blur** 仍是開放問題（06 號招牌震動）
- NIQE / NRQM 對所有積極處理都有偏差，這是 IQA 領域的已知問題

未來：用真實 paired night-blur 資料集對 FFTformer 做 few-shot fine-tuning；整合 reflection removal；在更多 deblur backbone 上驗證「手刻 vs 訓練式前處理」的差異。
