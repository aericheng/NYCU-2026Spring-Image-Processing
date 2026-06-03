# 繳交檔案說明

## 上傳的兩張

- `08_KFC_Rider__FINAL_graded.png` — KFC 外送員
- `05_Red_Taxi__FINAL_graded.png` — 紅色計程車

兩張的流程：裁出視覺中心主體 → FFTformer 去模糊（提供結構）→ SUPIR 生成式修復（scale ×2，補回高頻細節）→ 羽化合成回原圖（背景保留追焦動態模糊）→ 電影感調色（對比、分色、霓虹輝光、主體 clarity、暈影）。調色屬呈現層處理，去模糊本身是真實的。

對照圖：`08_RAW_vs_FINAL.jpg`、`05_RAW_vs_FINAL.jpg`。

## 08 的備選版本

FINAL 版的 08 箱體為整箱 SUPIR 均勻銳化（箱上中文區為生成結果，非原字）。若希望中文區更保守，可改用：

- `08_KFC_Rider__A_FFTsoft_chinese.png` — 中文區改用 FFTformer 柔化（不生成假字），其餘維持 SUPIR 銳化。

## 為什麼用生成式

回歸式去模糊（FFTformer）只能還原模糊尚未抹除的細節；當高頻已被運動模糊破壞，它只能輸出柔和的猜測。SUPIR（CVPR 2024，SDXL 擴散先驗）以擴散先驗合成合理的高頻紋理，補回回歸式去模糊難以還原的細節（屬生成結果，不保證與真實一致）。完整方法、對照實驗與量化評估見 `report/Report.pdf`。
