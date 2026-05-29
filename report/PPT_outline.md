# PPT 大綱 — 影像處理期末專題

> 用途：互評上傳 + 5 分鐘課堂報告共用。建議 8–10 張投影片。
> 排版風格：左圖右文 / 大圖小字，PPT 不要塞滿。

---

## Slide 1 — 標題頁

- 題目：**夜間運動模糊影像修復：以 Pipe A 為核心的逐圖細調管線**
- 副標：Night-Blur Image Deblurring with a Universal Preprocessing Pipeline and Per-Image Strengthening
- 課程 / 學期：影像處理 2026 春
- 學號 / 姓名
- 日期：2026-06-12

---

## Slide 2 — 問題與資料

**任務**：給定 15 張夜間 / 低光運動模糊照片，從 paper-based 開源方法挑選並改進，繳交 2 張最佳結果，公開互評。

**輸入資料特性**（圖片：縮圖 3×5 grid 顯示 Images/01.jpg ~ 15.jpg）：
- 解析度大：3000×4000 ~ 8000×5300
- 模糊類型多：linear motion、zoom blur、panning、shake、玻璃反射混雜
- 共同挑戰：**低光 + 雜訊 + 大尺度模糊核**，主流 GoPro/RealBlur 預訓練模型 OOD（out-of-domain）

> 對照：標準 deblur benchmark 多為白天光線、合成模糊。

---

## Slide 3 — 候選方法比較

表格：

| 方法 | 年份 | 訓練集 | 我們實測表現 | 備註 |
|---|---|---|---|---|
| DarkIR | CVPR 2025 | LOL / RealBlur | **僅曝光提升，不能去模糊** | 適合作為前處理 |
| MISCFilter | CVPR 2024 | GoPro | 文字邊緣略改善，色偏 | VRAM 需 mini-batch tiling |
| **FFTformer** (RealBlur-J) | CVPR 2023 | RealBlur-J | **去模糊最強**，但細節在 native 解析度會弱化 | 採用 max-side 1024 + 上採回原尺寸 |
| Real-ESRGAN | ICCV 2021 | 合成 | 加銳但會產生 cartoon artifact | 文字場景反而變糊 |

**結論**：FFTformer 作主力，搭配自訂前處理放大其作用範圍。

---

## Slide 4 — 我們的方法 Pipe A v2（核心創新）

Pipeline 圖（流程方塊）：使用 `report/pipe_a_diagram.png`

```
Input (raw night-blur)
   ↓ gamma 0.75 (per-image)
   ↓ CLAHE clip 5.0, grid 8 (boost local contrast on L channel)
   ↓ saturation × 1.2 (HSV)
   ↓ resize max-side 1024
   ↓ FFTformer (RealBlur-J weights)
   ↓ resize back to native
   ↓ unsharp mask (r=2, pct=200, thr=2)
Output
```

**為什麼這樣設計**：
- **CLAHE + Unsharp 是核心**（ablation 證實：拿掉各使 MUSIQ −3.85 / −2.94）
- max-side 1024 避免 FFTformer 在 8K 大圖上失效（receptive field 不夠）
- v2 來自參數掃描：CLAHE 3→5、Unsharp 130→200（§4.6.4），15 張平均 MUSIQ 30.96 → 33.74

---

## Slide 5 — 逐圖細調（Per-Image Strengthening）

**觀察**：同一條 Pipe A 不能對每張圖都最佳。

**對策**：保留 Pipe A v2 作骨架，針對個別瓶頸調 gamma：

- **09 White Truck (Biffa)**：曝光中等 → **γ=1.0**（跳過 gamma，避免壓平動態範圍）+ saturation mask 銳化（pct=220 r=3，sat>70）。MUSIQ 28.71 → **36.41**（+7.70）
- **08 KFC Rider**：曝光偏暗 → **γ=0.7**（更積極拉亮）+ saturation 1.25。MUSIQ 31.59 → **32.32**（+0.73）

> 創新故事：**通用管線（Pipe A v2）＋ per-image gamma 條件調整 ＝ 雙層方法**
> Gamma ablation 揭示「per-image conditional」的設計依據

---

## Slide 6 — 結果：09 White Truck

放 `final_submissions/I_plus_pipeA_strengthened/09_White_Truck_Zoom_Blur_Rain_compare.jpg`

說明：
- 紅色 "Biffa" 字樣由完全糊掉變為可辨識
- 背景樹冠、車燈光暈線條、車身細節浮現
- **MUSIQ 25.02 → 36.41（+11.39，全 15 張最大提升）**

---

## Slide 7 — 結果：08 KFC Rider

放 `final_submissions/I_plus_pipeA_strengthened/08_KFC_Rider_Rainy_Night_Delivery_compare.jpg`

說明：
- 騎手側臉、頭盔結構從一團糊變立體
- 左側 KFC 商店招牌恢復可閱讀，柯內爾上校 logo 清晰
- **MUSIQ 29.18 → 32.32（+3.14）**

---

## Slide 8 — 量化評估（IQA Ablation）

**插入圖**：`report/iqa_bar.png`（4 個 subplot，Pipe A v2 / Plan I+ v2 紅色 highlight）

或表格：

| Method | NIQE ↓ | MUSIQ ↑ | MANIQA ↑ | NRQM ↑ |
|---|---|---|---|---|
| Input | 5.31 | 28.22 | 0.174 | 6.04 |
| FFTformer | 7.05 | 25.38 | 0.194 | 3.20 |
| Chain (DK→FFTf) | 6.78 | 24.95 | 0.191 | 3.30 |
| Pipe A v1 | 6.35 | 30.96 | 0.192 | 4.06 |
| **Pipe A v2** | 6.23 | **33.74** | **0.199** | 4.49 |
| **Plan I+ v2** | **6.05** | **34.37** | 0.151 | 4.40 |

加一句：「**v1 → v2 四個指標全面改善**；Chain ablation 證明手刻前處理優於訓練式」

---

## Slide 9 — 失敗案例與限制

並排兩張小圖：
- **15 Photographer Glass Reflection**：玻璃反射混合，所有方法都無法分離真實影像與反射
- **09 native crop**：FFTformer 在 1024 推論回放會略失原生細節（已用 unsharp 補償）

教訓：
- 預訓練模型對 OOD blur kernel（zoom + rain streak）仍敏感
- No-reference IQA 無法區分「銳化過度」與「真實清晰」，須輔以主觀比對

---

## Slide 10 — 結論

**貢獻**：
1. 設計 **Pipe A v2** 通用前處理管線（CLAHE 5 + Unsharp 200），15 張平均 MUSIQ 33.74（+5.52 vs input）
2. 引入 **saturation-masked sharpening**，09 文字區獲得 +11.39 MUSIQ
3. **Ingredient ablation + 參數掃描** 驗證設計：v1 → v2 四指標全面改善

**侷限**：玻璃反射、zoom-blur 仍是開放問題，需要 reference-blur paired data 重新訓練。

**未來**：使用實拍 paired dataset 微調 FFTformer，或結合 diffusion-based deblurring（TP-Diff）。

---

## 講稿提示（5 分鐘版本）

- 0:00–0:30 — 自我介紹 + 題目
- 0:30–1:00 — 任務描述、為什麼難
- 1:00–2:00 — 候選方法、我們選 FFTformer 的理由
- 2:00–3:00 — Pipe A pipeline 走一遍
- 3:00–4:00 — 兩張結果展示（09 & 08）
- 4:00–4:30 — IQA 表格 + 失敗案例
- 4:30–5:00 — 結論

> 重點：**不要花太多時間講候選方法被淘汰的細節**，把時間留給 Pipe A 與兩張結果展示。
