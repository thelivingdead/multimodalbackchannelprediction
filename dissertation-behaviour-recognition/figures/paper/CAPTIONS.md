# Figure captions (paste-ready)

Numbers below are copied from locked json/csv. None of these figures is a new GOLD TEST score. Pose Euler and RGB face crops are two **visual representations** of the same listener window, not two sensory modalities. All classifiers predict a **window-level 0/1 label**, not a future event (not forecasting).

Print width ~12–14 cm. Files are PNG (300 dpi) and PDF under `dissertation-behaviour-recognition/figures/paper/`.

---

## README teaser (GitHub lead): 3 s listener heads

**Files:** `teaser_windowed_heads.png` / `.jpg` / `.svg`

**Caption.** Two labelled 3 s TEST windows: a strip of listener faces and the matching Euler trace. Grey band = annotated gesture; dashed vertical = annotated onset; frames are sampled inside the annotated interval and placed at their true times on the 0-3 s axis. Yaw and Pitch here are EMOCA rotation channels y and x; the anatomical mapping was verified in the methods (Chapter 4), not assumed from the channel names. (a) Head shake, `gold_023`, 15.0 to 18.0 s, watch RIGHT; annotated shake 16.0 to 17.0 s. Dashed horizontal lines mark the frozen yaw amplitude threshold (peak-to-peak vs 4.091°). (b) Head nod, `gold_030`, 21.0 to 24.0 s, watch RIGHT; annotated nod 22.0 to 23.0 s. The vertical bracket is the Savitzky-Golay peak-to-peak the return-ratio rule uses (amplitude >= 1.49°); the callout is return ratio <= 0.21. Pose slicing matches `start_frame_relative` at 25 fps; there is no one-second index error. The annotator marked the trough and the return (the descent begins just before onset), which is why the frozen nod rule uses return ratio rather than amplitude alone. Listener crops use the official RealTalk box for the gold person (p0 = LEFT, p1 = RIGHT), one box held for the window. Clip ids are in this caption only, not on the figure. Withdrawn largest-face Haar is not used. This figure does not rescore TEST. Distinct from `teaser_shake_windowed` (pose chart, no faces) and from `teaser_backchannel` (superseded 60 s nod protocol). RealTalk stills follow the RealTalk terms.

**Source.** `data/windowed_annotations/shake_events_windowed_test.csv` (`gold_023` shake 16.0 to 17.0 s); `data/windowed_annotations/nod_events_windowed_test.csv` (`gold_030` nod 22.0 to 23.0 s); `features/gold/gold_023.npz`, `features/gold/gold_030.npz`. Frozen shake tau from `results/windowed_shake/baselines_bacc/metrics.json` (axis y, 4.091°). Frozen nod thresholds from `scripts/evaluate_windowed_nod_return_ratio_test.py` (1.492°, 0.213). Script: `scripts/plot_teaser_windowed_heads.py`. This is the GitHub README lead. Distinct from `teaser_shake_windowed` (pose chart, no faces) and from `teaser_backchannel` (superseded 60 s nod protocol).

---

## Pose-only 3 s shake yaw chart (not the GitHub lead)

**Files:** `teaser_shake_windowed.png` / `.jpg` / `.svg`

**Caption.** Locked TEST 3 s yaw rule on `gold_028` (shake-only). Top: EMOCA yaw (axis y). Bottom: per-window peak-to-peak amplitude against the DEV-selected threshold τ = 4.091°. Orange = annotated shake; green = rule positive. The printed interval 0.654 [0.525, 0.794] is the 15-clip TEST balanced accuracy, not this clip alone. Pose only; no face crops. Do not caption this figure as showing a moving head. The old face teaser `teaser_backchannel.jpg` is the superseded 60 s nod protocol (pitch, τ = 16.35°) and is not the README lead.

**Source.** `results/windowed_shake/baselines_bacc/metrics.json` (axis 1 = y, `dev_selected_window_threshold` = 4.091°); `predictions.csv` TEST rows for `gold_028`; `data/windowed_annotations/shake_events_windowed_test.csv`; `features/gold/gold_028.npz`.

---

## Fig. A — Nod TEST F1

**Files:** `nod_test_f1.png` / `nod_test_f1.pdf`

**Caption.** Held-out GOLD TEST performance for binary head-nod recognition (n = 15 windows, scored once). Bars are F1; whiskers are 95% bootstrap CIs (1000 resamples, seed 42) from `results/tables/bootstrap_ci.csv`. Precision and recall are printed to the right of each bar. Pose rule F1 = 0.67 (P 0.64, R 0.70); pose CNN F1 = 0.70 (P 0.70, R 0.70); frozen VideoMAE F1 = 0.57 (P 0.55, R 0.60); fine-tuned VideoMAE n=80 F1 = 0.82 (P 0.75, R 0.90); fine-tuned VideoMAE n=200 F1 = 0.63 (P 0.67, R 0.60). Inputs are visual representations of the same clip (EMOCA Euler vs RGB 16×224×224 face crops), not two sensory modalities.

**Source.** `results/tables/main_results.csv` (and `results/tables/main_results.md`).

---

## Fig. B — Nod TEST confusion counts

**Files:** `nod_test_confusion.png` / `nod_test_confusion.pdf`

**Caption.** Confusion counts on the same GOLD TEST nod split (n = 15, scored once). TN/FP/FN/TP are taken from the locked metric json for each system. The n=200 VideoMAE run is a scaling ablation, not the RGB headline.

**Source.** `results/rule_test_metrics.json`; `results/classifier_test_metrics.json`; `results/videomae_frozen_head/metrics.json` (`test_metrics`); `results/videomae_finetuned/metrics.json` (`test_metrics`); `results/videomae_finetuned_n200/metrics.json` (`test_metrics`).

---

## Fig. C — Shake TEST F1

**Files:** `shake_test_f1.png` / `shake_test_f1.pdf`

**Caption.** Held-out GOLD TEST performance for binary head-shake recognition (n = 15 windows, scored once). The locked pose rule used Euler **axis z** (roll), τ = 11.15°, not geometric yaw. Always-shake (predict 1 on every TEST clip) has F1 = 0.64 and is drawn as a dashed reference line. Pose rule (z) F1 = 0.70; pose CNN trained 75 pos / 5 neg F1 = 0.64 (TN = 0); frozen VideoMAE 75/5 F1 = 0.60; fine-tuned VideoMAE 75/5 F1 = 0.60. This figure does not re-score TEST.

**Source.** `results/shake/rule_test_metrics.json`; `results/shake/majority_baseline/metrics.json` (`always_positive`); `results/shake/cnn/metrics.json` (`test_metrics`); `results/shake/videomae_frozen_head/metrics.json` (`test_metrics`); `results/shake/videomae_finetuned/metrics.json` (`test_metrics`). Axis: `results/shake/rule_selected_config.json` (`axis_name` = z).

---

## Fig. D — Shake TEST confusion counts

**Files:** `shake_test_confusion.png` / `shake_test_confusion.pdf`

**Caption.** Confusion counts on GOLD TEST head-shake (n = 15, scored once). Always-shake and the 75/5 pose CNN both have TN = 0 (all TEST clips predicted shake). Frozen and fine-tuned VideoMAE 75/5 share the same 2×2 counts (TP 6, FP 7, TN 1, FN 1) but are not identical clip-by-clip decisions.

**Source.** Same json files as Fig. C. Clip-level note: `results/shake/majority_baseline/metrics.json` (`videomae_locked_test_compare`).

---

## Fig. E — Shake DEV-only F1 (TEST not scored)

**Files:** `shake_dev_only_f1.png` / `shake_dev_only_f1.pdf`

**Caption.** Head-shake **DEV-only** comparison on GOLD DEV (n = 15; 10 shake+ / 5 shake−). GOLD TEST was **not** scored for the search runs. The dashed line is the always-shake baseline on this split (F1 = 0.80; TP 10, FP 5, TN 0, FN 0). Hatched bars collapsed on DEV (predicted-positive rate > 0.85 or TN = 0). Among non-collapsed search runs, pose CNN 40/40 was selected on DEV (F1 = 0.818, P = 0.75, R = 0.90). Locked 75/5 CNN DEV F1 = 0.870 is collapsed (recall = 1). This panel must not be reported as a TEST result.

**Source.** `results/shake/dev_search/summary.csv` and `results/shake/dev_search/comparison_dev.md`. CNN 40/40: `results/shake/dev_search/cnn_40_40/metrics_dev.json`.

---

## Fig. F — Euler power spectrum (illustration of the signal)

**Files:** `euler_signal_spectrum.png` / `euler_signal_spectrum.pdf`

**Caption.** Mean power spectrum of EMOCA Euler traces on the 30 gold windows (DEV+TEST). **Illustration of the signal, not a detector:** scored systems in this dissertation are amplitude rules, a 1D CNN, and VideoMAE, not an FFT classifier. Axis mapping: **x = pitch (nod-like)**, **y = yaw (shake-like)**, **z = roll (tilt-like)**. The locked shake TEST rule used **z**; geometric left–right shake on GOLD DEV is **y**. Nod grouping uses gold nod labels on axis x; shake grouping uses gold shake labels on axes y and z. Not a TEST metric.

**Source.** `features/gold/gold_*.npz` (`rotation_xyz`) and `data/gold/shake_annotation_sheet.csv`. Mapping: `results/shake/dev_search/axis_audit_conclusion.json` (`literature_mapping`, `locked_rule_axis`, `geometric_shake_axis`).

The older two-panel file `euler_power_spectrum.png` had a cramped title/footer; prefer this three-axis figure.

---

## Fig. G — Nod-only vs shake-only Euler traces

**Files:** `euler_nod_vs_shake_traces.png` / `euler_nod_vs_shake_traces.pdf`

**Caption.** EMOCA Euler x/y/z over the 60 s gold window for one **nod-only** DEV clip (`gold_009`, nod=1, shake=0) and one **shake-only** DEV clip (`gold_004`, nod=0, shake=1). **Illustration of the signal, not the classifier.** x = pitch (nod-like), y = yaw (shake-like), z = roll (tilt-like). The locked TEST shake rule used z; geometric shake is y.

**Source.** `features/gold/gold_009.npz`, `features/gold/gold_004.npz`; labels from `data/gold/shake_annotation_sheet.csv`.

---

## Fig. H — DEV axis audit (mean rule amplitude)

**Files:** `euler_axis_audit_dev.png` / `euler_axis_audit_dev.pdf`

**Caption.** Mean oscillatory rule amplitude (°) on GOLD DEV (n = 15), by Euler axis. Left: shake− (n = 5) vs shake+ (n = 10). Right: exclusive labels, shake-only (n = 4) vs nod-only (n = 3). **Illustration of the signal, not the detector.** On exclusive DEV clips, nod-only energy is highest on x (pitch); geometric shake is y (yaw). The locked TEST rule nevertheless used z. Not a TEST F1.

**Source.** `results/shake/dev_search/axis_audit_conclusion.json` (`axis_summary`).

---

## Fig. I — DEV per-clip rule amplitude

**Files:** `euler_axis_strips_dev.png` / `euler_axis_strips_dev.pdf`

**Caption.** Per-clip rule amplitude on GOLD DEV (n = 15). Points are individual windows; the horizontal bar is the group mean. Panel titles mark geometric yaw (y) versus the locked TEST rule axis (z). Not a TEST metric.

**Source.** `results/shake/v2/axis_audit/dev_axis_stats.csv`.

---

## Fig. J — Two visual representations of the same window

**Files:** `visual_representations.png` / `visual_representations.pdf`

**Caption.** The study compares two **visual representations** of the same Columbia RealTalk listener window (~60 s, 25 fps): EMOCA Euler rotation (x, y, z) and RGB face crops (16×224×224). These are not two sensory modalities (there is no audio in this figure). Each stream produces a clip-level 0/1 window label (nod or shake, depending on the experiment), not a forecast of a future backchannel.

---

## Fig. K — Nod qualitative TEST cases

**Files:** `nod_qualitative_cases.png` / `nod_qualitative_cases.pdf`

**Caption.** Pitch (Euler x) on four GOLD TEST windows, with locked 0/1 predictions from the pose rule, pose CNN, frozen VideoMAE, and fine-tuned VideoMAE n=80. Predictions are read from existing TEST CSVs (scored once); this figure does not re-evaluate TEST.

**Source.** `features/gold/gold_{016,017,018,024}.npz`; `results/rule_test_predictions.csv`; `results/classifier_test_predictions.csv`; `results/videomae_frozen_head/predictions.csv`; `results/videomae_finetuned/predictions_test.csv`.

---

## Fig. L — Nod vs unclear pitch traces

**Files:** `nod_vs_unclear_pitch.png` / `nod_vs_unclear_pitch.pdf`

**Caption.** What a nod looks like in this study: EMOCA axis x (pitch) on a gold nod TEST window (`gold_016`, peak-to-peak ≈ 64.5°) versus a gold unclear TEST window (`gold_024`, peak-to-peak ≈ 13.5°). The dashed line is the frozen nod-rule threshold τ = 16.35°. Illustration of the pose signal, not a new score.

**Source.** `features/gold/gold_016.npz`, `features/gold/gold_024.npz`. Threshold τ = 16.35° from `results/rule_selected_config.json` (`selected_amplitude_threshold` = 16.3538°).

---

## Fig. M — Frozen VideoMAE training (DEV F1)

**Files:** `nod_videomae_frozen_head_training.png` / `nod_videomae_frozen_head_training.pdf`

**Caption.** Training loss and **DEV** F1 by epoch for the frozen VideoMAE head (nod). The star marks early stopping (best epoch 10, DEV F1 = 0.90 from `metrics.json`). This is not a TEST curve.

**Source.** `results/videomae_frozen_head/training_history.csv`; `results/videomae_frozen_head/metrics.json`.

---

## Fig. N — Fine-tuned VideoMAE training (DEV F1)

**Files:** `nod_videomae_finetuned_training.png` / `nod_videomae_finetuned_training.pdf`

**Caption.** Training loss and **DEV** F1 by epoch for VideoMAE fine-tuned on the last four blocks (nod). Early stop from `metrics.json`. Not a TEST curve.

**Source.** `results/videomae_finetuned/training_history.csv`; `results/videomae_finetuned/metrics.json`.

---

## Figures that were messy (do not paste those copies)

| Old file | Problem |
| --- | --- |
| `figures/paper/euler_power_spectrum.png` | Long suptitle + footer overlapping the axes; only x and z shown |
| `figures/paper/error_cases.png` | Top-row x labels overlapping bottom-row titles |
| `figures/paper/nod_vs_unclear.png` | Suptitle overlapping panel titles |
| `figures/paper/test_clip_grid.png` | Footer overlapping column labels |
| `figures/model_comparison_f1.png` | F1 numbers struck through by CI whiskers |
| `figures/videomae_training_curve.png` | Legend covering the DEV F1 line |
| `figures/videomae_finetuned_training_curve.png` | Legend covering the DEV F1 line |
| `figures/shake_v2/dev_axis_class_bars.png` | Panel y-labels colliding; red–blue legend inside axes |
| `figures/shake_v2/dev_traces_shake_pos.png` | Clip IDs overlapping traces; cramped 10×3 grid |
| `figures/gold_visuals/pose_traces_extracted.png` | Subplot titles colliding with x labels of the row above |
| `figures/classifier_confusion_matrix.jpg` | Numeric ticks −0.5…1.5 instead of class names |
| `figures/shake_axis_audit/gold_*_xyz.png` | Cramped y labels (`x (°) pitch (nod-like)` against ticks) |

Use the new `figures/paper/` stems above instead.

---

## Fig. O — MFCC / concat fusion F1 (GOLD DEV)

**Files:** `audio_dev_mfcc_f1.png` / `audio_dev_mfcc_f1.pdf`

**Caption.** Exploratory GOLD DEV nod results (n = 15 windows). GOLD TEST was not scored. Thresholds were selected on this same DEV split. Always-nod F1 = 0.75 (P 0.60, R 1.00); MFCC audio LR F1 = 0.73 (P 0.62, R 0.89); frozen RGB + LR F1 = 0.86 (P 0.75, R 1.00); RGB+audio concat LR F1 = 0.78 (P 0.64, R 1.00). Mixed conversation audio. These DEV F1 values are not locked TEST scores.

**Source.** `results/tables/multimodal_ablation.md`; `results/audio_visual_fusion_dev.csv`.

---

## Fig. P — Frozen HuBERT F1 (GOLD DEV)

**Files:** `audio_dev_hubert_f1.png` / `audio_dev_hubert_f1.pdf`

**Caption.** Exploratory GOLD DEV nod results (n = 15) at a fixed probability threshold of 0.5. GOLD TEST was not scored. Always-nod F1 = 0.75 (P 0.60, R 1.00); frozen HuBERT + LR F1 = 0.89 (P 0.89, R 0.89); frozen RGB + LR F1 = 0.82 (P 0.69, R 1.00); 50/50 HuBERT+RGB probability fusion F1 = 0.80 (P 0.73, R 0.89). Encoder: frozen `facebook/hubert-base-ls960` (768-D, 10 s chunks, mean pool). Mixed conversation audio. TRAIN = existing 80 pose-derived pseudo-labels. RGB here uses threshold 0.5, not the DEV-selected threshold in Fig. O.

**Source.** `results/hubert_dev/hubert_dev_metrics.json`; `results/hubert_dev/hubert_rgb_fusion_metrics.json`; `results/hubert_dev/multimodal_dev_comparison.csv`.

---

## Fig. Q — Audio DEV confusion counts

**Files:** `audio_dev_confusion.png` / `audio_dev_confusion.pdf`

**Caption.** Confusion counts for the GOLD DEV audio experiments in Figs O–P (n = 15). GOLD TEST was not scored. MFCC and concat rows use DEV-selected thresholds; HuBERT rows use threshold 0.5.

**Source.** Same files as Figs O–P.

---

## Fig. R — HuBERT TRAIN-label permutation null (GOLD DEV)

**Files:** `permutation_null.png` / `permutation_null.pdf`

**Caption.** Null distribution from 1000 TRAIN-label permutations of the frozen HuBERT logistic head, evaluated on GOLD DEV (n = 15). GOLD TEST was not scored. Feature vectors and DEV labels were held fixed; only the 80 pseudo TRAIN labels were shuffled (counts preserved). Each panel is an independent histogram (25 bins) of the permutation metric; the dotted line is the permutation mean and the dashed line is the observed HuBERT value at threshold 0.5. Balanced accuracy p = 0.008 (7/1000); F1 p = 0.009 (8/1000). The two nulls are not centred at the same place (mean BA ≈ 0.518, mean F1 ≈ 0.673). This is a development-only figure.

**Source.** `results/hubert_dev/permutation_metrics.csv`; `results/hubert_dev/permutation_summary.json`.

---

## Fig. S — Acoustic DEV comparison (F1 and balanced accuracy)

**Files:** `acoustic_dev_comparison.png` / `acoustic_dev_comparison.pdf`

**Caption.** GOLD DEV nod comparison (n = 15) at a fixed probability threshold of 0.5. GOLD TEST was not scored. Bars are F1 and balanced accuracy for the always-positive baseline, frozen RGB VideoMAE, equal-weight (50/50) HuBERT+RGB fusion, and frozen HuBERT on mixed conversation audio. The dashed line at 0.5 is chance balanced accuracy. These DEV scores are not locked TEST scores.

**Source.** `results/hubert_dev/multimodal_dev_comparison.csv`.

---

## GOLD TEST figures already in this folder (do not re-score)

These files exist under `figures/paper/`. They are locked TEST plots (or TEST-window illustrations). None of the new audio figures above is a GOLD TEST score.

| Stem | Files present | Role in thesis |
| --- | --- | --- |
| `teaser_windowed_heads` | `.png` `.jpg` `.svg` | GitHub README lead. 3 s face figure: TEST shake `gold_023` 15 to 18 s and nod `gold_030` 21 to 24 s |
| `teaser_shake_windowed` | `.png` `.jpg` `.svg` | Pose-only 3 s yaw rule on TEST `gold_028`. Not the GitHub lead |
| `nod_test_f1` | `.png` `.pdf` | Fig. A — nod GOLD TEST F1 (n = 15, scored once) |
| `nod_test_confusion` | `.png` `.pdf` | Fig. B — nod GOLD TEST confusion counts |
| `shake_test_f1` | `.png` `.pdf` | Fig. C — shake GOLD TEST F1 (n = 15, scored once) |
| `shake_test_confusion` | `.png` `.pdf` | Fig. D — shake GOLD TEST confusion counts |
| `nod_qualitative_cases` | `.png` `.pdf` | Fig. K — pitch traces on four GOLD TEST windows; predictions from existing TEST CSVs |
| `nod_vs_unclear_pitch` | `.png` `.pdf` | Fig. L — TEST-window illustration (`gold_016` vs `gold_024`); not a new score |

Older TEST-related files in the same folder (do not paste; listed as messy above): `test_metrics.png` / `.jpg`, `test_clip_grid.png` / `.jpg`, `error_cases.png` / `.jpg`, `nod_vs_unclear.png` / `.jpg`.

