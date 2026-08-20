# Annotation audit — out-of-window events

**Date:** 20 August 2026
**Scope:** the two gold windows whose recorded nod interval falls outside the watched/analysed window: `Ak2Bm8mfL3w` (DEV) and `Zrer1sqWzOQ` (TEST).
**Outcome:** both labels are kept exactly as recorded. No ground truth was changed. The discrepancies are documented here so the TEST metrics can be interpreted correctly.

## Method

For every one of the 30 gold rows, three independent records were cross-checked:

1. `data/gold/annotation_sheet.csv` — YouTube-clock watch window (`watch_from`–`watch_until`) and nod interval (`nod_start`–`nod_end`).
2. `data/gold/events.csv` — the same nod interval in seconds.
3. `features/gold/<sample_id>.npz` — the EMOCA pose actually extracted (frame range, valid ratio), checked against `data/gold_annotations.csv`.

28 of 30 rows are fully consistent: the nod interval lies inside the watch window, and the extracted frame window equals `watch_from`–`watch_until` at 25 fps. The two rows below are the only exceptions; both are also flagged programmatically by `outside_window` in `results/gold_annotation_summary.json`.

## Case 1 — `Ak2Bm8mfL3w` (`gold_015`, DEV, label 1 = clear nod, person p0 / LEFT)

| Record | Value |
| --- | --- |
| Watch window (sheet) | 13:34–14:34 (814–874 s) |
| Nod interval (sheet + events.csv) | 1:57–1:58 (117–118 s) |
| Extracted frames (`gold_015.npz`) | 20350–21849 = 814.0–874.0 s at 25 fps |
| Offset of nod vs window | ~697 s (11 min 37 s) before the window start |
| Pose coverage (valid_ratio) | 0.999 |
| Sheet URL | `…&t=813` (window start); the export in `results/predicted_vs_annotated.csv` uses `&t=117` (nod time) |

**Finding.** The recorded nod time (1:57–1:58) is inconsistent with the watched and feature-extracted window (13:34–14:34). The frame window in `data/gold_annotations.csv` matches the watch window, so the pose features cover 13:34–14:34, not 1:57–1:58. A plausible cause is a transcription error in the nod minute field (e.g. 13:57 recorded as 1:57), but this cannot be confirmed without re-watching the video, and the label is therefore left untouched.

**Impact.** DEV-only sample. The frozen rule's DEV recall is 1.0, i.e. this window's amplitude score (≥ 16.35°) was treated as a correct positive during tuning. If the nod in fact occurred outside the extracted window, this row's DEV label may be noisy; because DEV was used for threshold selection, the effect is absorbed into the frozen threshold and does not alter the untouched TEST evaluation. No correction applied.

## Case 2 — `Zrer1sqWzOQ` (`gold_018`, TEST, label 1 = clear nod, person p0 / LEFT)

| Record | Value |
| --- | --- |
| Watch window (sheet) | 4:56–5:56 (296–356 s) |
| Nod interval (sheet + events.csv) | 4:48–4:49 (288–289 s) |
| Extracted frames (`gold_018.npz`) | 7400–8899 = 296.0–356.0 s at 25 fps |
| Offset of nod vs window | 7–8 s **before** the window start |
| Pose coverage (valid_ratio) | 1.000 |
| Rule score / CNN probability | 4.32° (< 16.35° threshold → pred 0) / 0.168 (< 0.45 → pred 0) |

**Finding.** The recorded nod (4:48–4:49) ends 7 s before the analysed window begins (4:56–5:56). If the nod truly occurred at 4:48–4:49, the window on which both models were scored does not contain the nod, so the clip is a structural false negative rather than a model error. Both models indeed miss this clip: it is one of the three shared FN cases in the locked TEST confusion matrices (rule TP7 FP4 TN1 FN3; CNN TP7 FP3 TN2 FN3).

**Impact.** This row is part of TEST and is included in the verified headline numbers (rule F1 0.67, CNN F1 0.70). Correcting or removing it would change those numbers, so the label and window are kept exactly as recorded; this note is the documentation of the caveat. A future re-annotation pass could re-watch 4:30–6:00 and re-align the window, but any such change must be accompanied by a fresh, single TEST re-score and a new version of the results artifacts — none of that was done here.

## Secondary observation — export URLs

`results/predicted_vs_annotated.csv` builds its YouTube link from the **nod time** (`t=` seconds), while `data/gold/annotation_sheet.csv` links to the **window start**. For the 28 consistent rows this is a helpful deep link; for the two rows above it produces a URL that points outside the analysed window (e.g. `Ak2Bm8mfL3w&t=117`). This is a cosmetic consequence of the two timing discrepancies, not an additional error.

## What was verified as consistent

- All 30 extracted gold feature files match `data/gold_annotations.csv` on `sample_id`, `video_id`, `person`, and exact frame range (1500 frames each).
- All 30 nod intervals are 1–2 s; mean 1.1 s.
- No other row has a nod interval outside its watch window.
- Ground-truth files (`data/gold/annotation_sheet.csv`, `data/gold/events.csv`, `data/gold_annotations.csv`, `data/splits/*`) were **not modified** by this audit.
