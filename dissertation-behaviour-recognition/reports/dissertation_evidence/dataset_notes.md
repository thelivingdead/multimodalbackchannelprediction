# Dataset notes

- Gold set: 30 × ~1 min RealTalk windows, 15 DEV / 15 TEST.
- Labels collected 2026-08-15 in `data/gold/annotation_sheet.csv` and imported to `data/gold/events.csv`.
- Classes: `1` = clear nod (gold positive), `0` = unclear. Times are YouTube clock; `p0` = LEFT, `p1` = RIGHT; 25 fps.
- Two labelled times sit outside the planned watch window and were kept as recorded: `Ak2Bm8mfL3w` 1:57–1:58 (window 13:34–14:34); `Zrer1sqWzOQ` 4:48–4:49 (window 4:56–5:56).
- Pose / EMOCA files are not in this repository. Predicted intervals stay empty until matching `.pkl` files exist.
