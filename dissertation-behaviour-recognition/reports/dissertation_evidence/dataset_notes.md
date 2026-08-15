# Dataset notes

- Intended gold: 30 × ~1 min videos, 15 DEV / 15 TEST.
- Online gold sheet filled 2026-08-15: 30/30 labelled in `data/gold/annotation_sheet.csv` and imported to `data/gold/events.csv`.
- Labels: 1 = clear nod (only gold positive), 0 = unclear. YouTube clock times; p0=LEFT, p1=RIGHT.
- Two times sit outside the planned watch window (kept as labelled, not silently edited): `Ak2Bm8mfL3w` 1:57–1:58 vs window 13:34–14:34; `Zrer1sqWzOQ` 4:48–4:49 vs window 4:56–5:56.
- Predictions stay empty until matching RealTalk EMOCA `.pkl` exists. Do not download the 299 GB dataset or 23.6 GB `emoca.tar.gz`.
- FPS: 25 (RealTalk documented; stored in config/meta).
