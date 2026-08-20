# Split integrity and leakage audit

**Date:** 20 August 2026
**Question:** can any information from the 15 TEST windows have influenced the rule threshold, the CNN weights, the CNN epoch/threshold selection, or the pseudo-labels?
**Conclusion:** no leakage detected at the video, window, feature-normalisation, or model-selection level. TEST was scored once, with configurations frozen on DEV. Two data-quality observations (not leakage) are listed in §5.

All checks below were re-executed on the repository as it stands; the commands are noted so they can be repeated.

## 1. DEV / TEST separation (gold set)

| Check | Result |
| --- | --- |
| `data/splits/gold_dev.txt` ∩ `data/splits/gold_test.txt` | **empty** (15 + 15 distinct video ids) |
| `data/splits/gold_split.csv` per-video assignment | consistent with the two lists, one row per video |
| `tests/test_invariants.py::test_split_files_no_overlap` | passes (`pytest -q`) |
| `scripts/09_tune_rule.py` guard | aborts with `LEAK: test ids in dev` if the lists ever intersect |
| `results/gold_dataset_summary.json` → `video_overlap` | `[]` (recomputed at every full run) |
| Sample ids | `gold_001`–`gold_015` = DEV, `gold_016`–`gold_030` = TEST, disjoint by construction |
| `data/splits/planned_dev.txt` / `planned_test.txt` | byte-identical to the gold lists — the planned split *is* the frozen split, no post-hoc change |

## 2. Feature–label alignment (gold)

Each `features/gold/gold_0XX.npz` was checked against `data/gold_annotations.csv`: `sample_id`, `video_id`, `person`, and exact frame range (`start_frame`–`end_frame` − 1, i.e. 1500 frames = 60 s at 25 fps) match for **all 30/30** clips. Pose coverage (`valid_ratio`) ranges from 0.868 (`gold_008`) to 1.000; median ≈ 0.98.

## 3. Pseudo-TRAIN pool vs gold

The CNN is trained only on rule-labelled pseudo clips. Verified directly from the committed npz metadata:

| Check | Result |
| --- | --- |
| Pseudo clips | 80 (`features/pseudo/pseudo_00001–00080.npz`) |
| Unique source videos | 80 (no duplicates) |
| Overlap with the 30 gold videos | **none** (intersection is empty) |
| Person channel | all `p0` (pseudo pool is LEFT-listener only — a known sampling bias, see §5) |
| `valid_ratio` | min 0.196, mean 0.84 (one low-coverage clip retained, see §5) |

The extraction code enforces the separation structurally: in `scripts/run_full_experiment.py::stream_emoca`, a video id present in the gold set is only ever written to `features/gold/`; pseudo clips are drawn only from videos **not** in the gold set.

Pseudo-labels are produced by the **frozen** DEV-tuned rule (`results/rule_selected_config.json`), applied after the rule was fixed; the rule's threshold never saw TEST, so the pseudo-labels carry no TEST information. Label balance of the pseudo pool: 70 nod / 10 unclear (`results/pseudo_labels.csv`).

## 4. Model and threshold selection

| Choice | Made on | Evidence |
| --- | --- | --- |
| Rule axis (x) + amplitude threshold (16.35°) | DEV only (quantile search, `results/rule_dev_threshold_search.csv`) | `results/rule_selected_config.json`, frozen before TEST scoring; reused verbatim on re-runs |
| CNN feature normalisation (mean/std) | pseudo TRAIN only | `build_matrix(..., mean, std)` applies TRAIN statistics unchanged to DEV and TEST; stored in `models/normalization.json` at train time |
| CNN best epoch (9) | DEV F1 per epoch | `results/training_history.csv` |
| CNN decision threshold (0.45) | DEV sweep 0.20–0.80 | same file, `dev_probability_threshold` |
| TEST evaluation | once, with the frozen rule / best-on-DEV checkpoint | `results/rule_test_metrics.json`, `results/classifier_test_metrics.json` |

Ablation feature sets A–D follow the identical protocol; D (with expression coefficients) diverged (`loss = nan`) and is excluded from all reported tables.

## 5. Observations that are **not** leakage (documented for completeness)

- **Complementary side imbalance.** DEV has 10 p0 / 5 p1 clips, TEST 5 p0 / 10 p1. The split was fixed before any model output existed, so this is a sampling property, not leakage; it is noted in `reports/dissertation_evidence/limitations.md`.
- **Pseudo pool is p0-only and includes one low-coverage clip** (`valid_ratio` 0.196). This biases the pseudo-label distribution towards the LEFT listener and slightly noisy pose; it affects training-data quality, not TEST contamination.
- **`results/feature_quality.csv` lists 23 of 30 gold rows.** It is appended per streaming session and resumed sessions skip already-extracted clips; the authoritative coverage check is the npz-level verification in §2 (30/30).
- **Two out-of-window nod annotations** (`Ak2Bm8mfL3w` DEV, `Zrer1sqWzOQ` TEST) are auditing findings about label–window alignment, not leakage; see `reports/annotation_audit.md`. They are unchanged.

## 6. Bottom line

The verified TEST numbers — rule F1 0.67 (P 0.64, R 0.70; TP7 FP4 TN1 FN3) and 1D CNN F1 0.70 (P 0.70, R 0.70; TP7 FP3 TN2 FN3) — rest on a clean split: no shared videos between DEV/TRAIN/TEST, TRAIN-only normalisation, DEV-only selection, single TEST scoring.
