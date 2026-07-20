# 繳交檔案說明 — 夜間運動模糊人臉修復

## 上傳的兩張

- `01_NightMarket_Man__FINAL.png` — 夜市人群中的男子（來源：影像 10）
- `02_NeonAlley_Woman__FINAL.png` — 霓虹巷弄中的女子（來源：影像 14）

對照圖：`01_Man_BEFORE_AFTER.jpg`、`02_Woman_BEFORE_AFTER.jpg`。

## fidelity（重要）

兩張皆為「模糊路人 → 清晰真人臉」的修復（FFTformer 去模糊 + SUPIR v0F 擴散先驗補細節）。
**#2 女子的眼睛 / 上半臉為 SUPIR 生成**（原圖該區被運動重影破壞，無法真實還原）。#1 男子的五官 layout 在原圖中真實存在、由高 control 緊錨，但高頻細節同樣由先驗合成（prior-guided，非逐像素還原）；差別在約束程度，不在真實 vs 生成。

完整方法、對照實驗與 fidelity 誠實聲明見 `report/Report.pdf`（第 5 節）。
