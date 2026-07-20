# 課堂報告投影片大綱（5 分鐘）

用途：6/11 上傳 PPT + 6/12 課堂報告。建議 9 張投影片，左圖右文、字少圖大。

---

## Slide 1 — 標題
- 題目：夜間運動模糊的人臉修復：回歸式去模糊結合生成式先驗
- 組員：113950011 鄭名翔、[id] [teammate]、[id] [teammate]
- 影像處理 Term Project

## Slide 2 — 問題與目標
- 15 張夜間真實照片，6K–8K，含 motion blur，無 ground truth（不能用 PSNR/SSIM）
- 作業範例正是 blurry face → sharp face，我們以此為目標
- 目標：在主體可信還原的前提下，做出前後差異最明顯的 before/after（挑模糊最重但仍可修復的主體）
- 圖：Images 縮圖 grid

## Slide 3 — 基礎方法與它的天花板
- 成功執行 FFTformer（CVPR 2023，frequency-domain transformer，RealBlur-J 權重）
- 也比較過 DarkIR（只提亮）、MISCFilter（去模糊有限）
- 回歸式學 blur→sharp；臉的高頻被抹除後，輸出偏軟、帶噪（接近上限）
- 圖：face_fft_vs_supir（RAW｜FFTformer｜+SUPIR）的前兩格

## Slide 4 — 方法：回歸去模糊 + 生成式臉部先驗
- 裁臉 → FFTformer（去模糊、給乾淨結構）→ SUPIR-v0F（擴散先驗補臉部高頻，高 control 貼合原五官）→ 羽化合成 → 輕度調色
- 兩階段互補：先把 kernel 縮回範圍給乾淨結構，再補細節；high control 控制幻覺
- 圖：face_pipeline

## Slide 5 — 結果
- 繳交 2 張不同場景的「模糊路人 → 清晰真人臉」
- 圖：face_result_man、face_result_woman（before/after）
- #2（女子）上半臉為生成，於 Slide 6 揭露

## Slide 6 — 誠實面：真實修復 vs 生成合成（課堂重點）
- 臉部細節是擴散先驗合成（prior-guided，與範例 GFPGAN/GPEN 同類），非解卷積
- #1 男子：五官 layout 真實、高 control 緊錨；高頻仍為先驗合成
- #2 女子：眼睛/上半臉為生成（原圖被運動重影破壞）；生成受原圖約束（不同 seed 大致一致，屬觀察非保證）
- 重點：高 NR-IQA 不代表真實還原
- 圖：face_seed_consistency

## Slide 7 — 解決的實作困難（課堂重點）
- 16 GB 顯存跑 6–8K：FFTformer overlap-blend tiling；SUPIR tiled VAE / tiled sampling
- Blackwell sm_120：需 PyTorch 2.11+cu128；ComfyUI/SUPIR 與 FFTformer 相依衝突 → 兩個隔離 conda env
- 生成式幻覺控制：裁臉（背景不亂修）+ 高 control + 裁掉幻覺邊緣（14 整圖會生第二張臉）

## Slide 8 — 驗證過、但沒採用的方向
- 純車輛去模糊（05、08）：真實去模糊但對比太小，不符合「最明顯 before/after」的挑選目標
- 玻璃反射（15）：layer separation 非模糊，去不掉
- 整張 SUPIR：把背景人群幻覺成扭曲臉
- 圖：face_rejected
- 重點：說明什麼有效、什麼無效、為什麼

## Slide 9 — 結論與限制
- 貢獻：回歸→生成兩階段人臉修復 + 誠實的 fidelity 界線
- 限制：玻璃反射需 layer separation；被重影破壞的五官無法真實還原，只能可信生成
- 未來：臉部專用修復（fidelity 旋鈕）、真實 paired night-blur 微調、reflection removal

---

## 講稿時間分配（5 分鐘）
- 0:00–0:40 題目與目標、作業範例就是臉
- 0:40–1:30 FFTformer 與回歸天花板
- 1:30–2:30 方法（兩階段）+ 兩張結果
- 2:30–3:40 誠實面（真實 vs 生成）+ 解決的困難
- 3:40–4:30 驗證過但沒採用的方向
- 4:30–5:00 結論與限制

提醒：誠實區分「真實修復 vs 生成」是本題的加分重點，務必講清楚 #2 女子的眼睛是生成的。
