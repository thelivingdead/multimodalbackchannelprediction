# Experiment log

| date | purpose | split | result |
| --- | --- | --- | --- |
| 2026-08-15 | gold annotation | 15 DEV + 15 TEST | 30/30 labelled in `annotation_sheet.csv` and `events.csv`. |
| 2026-08-16 | rule TEST (clip F1) | TEST once | Frozen axis x, thr 16.35°. TEST P/R/F1 = 0.64 / 0.70 / 0.67 (TP7 FP4 TN1 FN3). |
| 2026-08-18 | 1D CNN TEST | TRAIN 80 pseudo; DEV epoch; TEST once | Pseudo 70/10. CNN TEST P/R/F1 = 0.70 / 0.70 / 0.70 (TP7 FP3 TN2 FN3). Ablation D diverged. No VideoMAE. |
