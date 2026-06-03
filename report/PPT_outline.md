# 課堂報告投影片大綱（5 分鐘）

用途：6/11 上傳 PPT + 6/12 課堂報告。建議 9 張投影片，左圖右文、字少圖大。

---

## Slide 1 — 標題

- 題目：夜間運動模糊影像修復——從回歸式去模糊到生成式主體修復
- 組員：113950011 鄭名翔、[id] [teammate]、[id] [teammate]
- 影像處理 Term Project

## Slide 2 — 問題與資料

- 15 張夜間真實照片，6K–8K，含 motion blur，沒有 ground truth（不能用 PSNR/SSIM）
- 模糊種類混雜：panning 追焦、zoom blur、手震、玻璃反射多重曝光
- 難點：低光 + 雜訊 + 大尺度 blur kernel + 高解析度，夜景對多數模型是 OOD
- 圖：Images 縮圖 grid

## Slide 3 — 基礎方法與它的天花板

- 我們成功執行 FFTformer（CVPR 2023，frequency-domain transformer，RealBlur-J 權重）
- 也比較過 DarkIR（只提亮不去模糊）、MISCFilter（去模糊有限）
- 觀察：回歸式模型學 blur→sharp 映射；當高頻被運動模糊抹除，缺乏依據時輸出偏平滑（推測接近其上限）
- 圖：rpt_f3（RAW｜FFTformer｜SUPIR）：FFTformer 偏平滑

## Slide 4 — 改進一：解析度策略

- FFTformer attention 近線性，但 6–8K 的中間 activation 仍超出 16GB 顯存；常見「縮 1024 再放大」在 native 其實更糊
- 發現：最佳去模糊解析度與 blur kernel 大小成反比
  - 重度 zoom blur（09）要 downscale 把 kernel 縮回訓練範圍
  - 輕度 panning（07/08）要高解析度 tiling 才留得住真細節
- per-image 自適應選解析度
- 圖：rpt_f2

## Slide 5 — 改進二：生成式主體修復

- 題目要的是把視覺中心主體變清楚，整張變清楚既難又沒必要（背景的追焦模糊可保留作速度感）
- 用 SUPIR（CVPR 2024，SDXL 擴散先驗）合成回歸模型變不出來的真實細節
- 兩階段 FFTformer→SUPIR：FFT 給乾淨結構，SUPIR 補高頻；低 cfg + Wavelet 護色保持忠實
- 完整管線：裁主體 → FFTformer → SUPIR ×2 → 羽化合成 → 後製調色（自寫 OpenCV/NumPy 程式，非 Photoshop）
- 圖：rpt_f1 pipeline

## Slide 6 — 解決的實作困難（課堂重點）

- 16 GB 顯存跑 6–8K：FFTformer attention 的 activation 爆顯存 → overlap-blend tiling；SUPIR 用 tiled VAE / tiled sampling，scale ×2 也要 tiled 才不 OOM
- Blackwell sm_120 GPU：舊 torch 跑不動，需 PyTorch 2.11+cu128；ComfyUI/SUPIR 與 FFTformer 相依衝突 → 開兩個隔離 conda env（deblur / comfy）
- 生成式幻覺控制：低 cfg + Wavelet 護色 + 裁主體（背景不亂修）
- 重點：這些是「解決困難」的得分點

## Slide 7 — 結果

- 繳交 08 KFC 外送員、05 紅色計程車
- 主體補回 FFTformer 輸出缺少的高頻紋理，背景保留追焦光條
- MUSIQ（去模糊本身的貢獻）：08 35.4→42.3、05 28.6→51.0；調色略降分，是為人眼觀感非衝指標
- 圖：rpt_f5、rpt_f6（RAW vs FINAL）

## Slide 8 — 驗證過、但沒採用的方向

- 整張 SUPIR vs 裁主體：整張會把背景動態模糊修成靜態雜亂、主體解析度也降，裁主體較好（rpt_f4）
- 人臉：對 15 號整張修復，玻璃反射的多重人影讓模型幻覺出多張臉，失敗
- DiffTSR 中文文字修復：模型對平面招牌很強，但本資料集文字不是裝飾品牌字、就是發光霓虹、或已可讀，method 與 data 不匹配
- 重點：說明什麼有效、什麼無效、為什麼

## Slide 9 — 結論與限制

- 貢獻：解析度策略（resolution = f(kernel)）+ 回歸→生成式兩階段主體修復管線
- 限制：玻璃反射需 layer separation；生成式有人臉走樣、中文品牌字無法還原（資訊已毀）
- 未來：真實 paired night-blur 微調、reflection removal、中文字形感知的文字修復

---

## 講稿時間分配（5 分鐘）

- 0:00–0:40 題目與資料、為什麼難
- 0:40–1:30 FFTformer 與回歸天花板
- 1:30–2:30 兩個改進（解析度策略、生成式主體修復）
- 2:30–3:40 兩張結果（08、05）
- 3:40–4:30 驗證過但沒採用的方向
- 4:30–5:00 結論與限制

提醒：別花太多時間講被淘汰的候選方法，把時間留給兩個改進與結果。
