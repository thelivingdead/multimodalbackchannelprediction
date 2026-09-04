# Figure captions (copy into Word)

All paths are relative to `dissertation-behaviour-recognition/`. Prefer **JPG** for Word. PNG duplicates exist for some gold plots. The GitHub README lead is Figure 15b (`figures/paper/teaser_windowed_heads`). The 60 s VideoMAE TEST scores (frozen 0.57, fine-tuned 0.82) belong to the earlier clip protocol, not the 3 s GitHub headline.

Insert **after** the paragraph that first mentions the result. Keep DEV plots out of the Results headline section.

---

## Use these

**Figure 1.** Distribution of human labels on the 30-window gold set: 19 clear nod (`1`) and 11 unclear (`0`). File: `figures/gold_visuals/label_counts.jpg` (alternative: `figures/gold_label_distribution.jpg`).

**Figure 2.** Gold labels by split. DEV: 9 nod / 6 unclear. TEST: 10 nod / 5 unclear. File: `figures/gold_visuals/labels_by_split.jpg` (alternative: `figures/gold_split_distribution.jpg`).

**Figure 3.** Gold labels by listener side (LEFT = p0, RIGHT = p1). File: `figures/gold_visuals/labels_by_person.jpg`.

**Figure 4.** Overview of the 30 watch windows used for annotation. File: `figures/gold_visuals/clip_overview.jpg`.

**Figure 5.** Example Euler traces for a gold-positive DEV window (clear nod). Axes are stored as \(x,y,z\) degrees and are not assumed to be anatomical pitch/yaw/roll in the caption. File: `figures/example_positive_rotation.jpg`.

**Figure 6.** Example Euler traces for a gold-negative DEV window (unclear). File: `figures/example_negative_rotation.jpg`.

**Figure 7.** DEV-only threshold search for the pose rule on the selected axis. The dashed line is the frozen threshold \(16.35^\circ\). This figure is **Methods**, not a TEST result. File: `figures/rule_dev_threshold_curve.jpg`.

**Figure 8.** Pseudo-label counts on 80 unlabelled TRAIN windows after the frozen rule: 70 predicted nod, 10 predicted unclear. These are automatic labels, not gold. File: `figures/pseudo_label_distribution.jpg`.

**Figure 9.** Extracted pose traces for gold windows (quality check). File: `figures/gold_visuals/pose_traces_extracted.jpg`.

**Figure 10.** TEST confusion matrix for the frozen pose rule (\(n=15\)): TP 7, FP 4, TN 1, FN 3; F1 0.67. File: `figures/rule_confusion_matrix.jpg`.

**Figure 11.** TEST confusion matrix for the 1D CNN trained on 80 pseudo-labels (\(n=15\)): TP 7, FP 3, TN 2, FN 3; F1 0.70. File: `figures/classifier_confusion_matrix.jpg`.

**Figure 12.** TEST F1 for the frozen rule (0.67) and the 1D CNN (0.70). File: `figures/model_comparison_f1.jpg`.

**Figure 13.** Frozen VideoMAE head training curve: training loss (left axis) and DEV F1 (right axis) by epoch. The star marks the early-stopped best epoch (10, DEV F1 0.90); TEST was scored once at that epoch and threshold. This is a tuning diagnostic — the DEV curve must not be read as generalisation. File: `figures/videomae_training_curve.png`.

**Figure 14.** TEST F1 for the four systems with saved TEST predictions (\(n=15\), scored once): frozen pose rule 0.67, pose 1D CNN (xyz + derivatives) 0.70, frozen VideoMAE head 0.57, fine-tuned VideoMAE 0.82 (highlighted). Error bars are 95% bootstrap CIs (1000 resamples, seed 42). The intervals overlap widely; the differences are not statistically significant — 0.82 is the highest point estimate, not a proven win. File: `figures/model_comparison_f1.png` (four-model version; distinct from the two-model `model_comparison_f1.jpg`).

**Figure 15.** Locked TEST 3 s yaw rule on `gold_028` (shake-only). Yaw trace and per-window amplitude against τ = 4.091°. Orange = annotated shake; green = rule positive. TEST balanced accuracy 0.654 [0.525, 0.794] is the 15-clip score. Pose only. File: `figures/paper/teaser_shake_windowed.jpg`. The old face teaser `teaser_backchannel.jpg` is the superseded 60 s nod protocol.

**Figure 15b.** Listener heads in labelled 3 s TEST windows: shake `gold_023` 15.0 to 18.0 s (watch RIGHT, yaw) and nod `gold_030` 21.0 to 24.0 s (watch RIGHT, pitch). Grey band = annotated gesture; dashed vertical = annotated onset; frames sampled inside the annotated interval. Yaw and Pitch here are EMOCA rotation channels y and x; the anatomical mapping was verified in the methods (Chapter 4), not assumed from the channel names. On the nod panel the annotator marked the trough and the return (the descent begins just before onset), which is why the frozen rule uses return ratio rather than amplitude alone. Shake dashed lines: yaw amplitude threshold 4.091°. Nod bracket: Savitzky-Golay peak-to-peak (amplitude >= 1.49°) with return ratio <= 0.21. Official RealTalk listener boxes; not withdrawn largest-face Haar. Clip ids in this caption only. File: `figures/paper/teaser_windowed_heads.jpg`. Distinct from Figure 15 (pose chart, no faces). RealTalk stills stay in the bound dissertation only.

**Figure 16.** Executed architecture (binary nod only): EMOCA pose → rule + 1D CNN; RGB 16-frame crops → VideoMAE. No BERT/HuBERT/LMF. File: `figures/paper/architecture.png`.

**Figure 17.** TEST P/R (text on bars) and F1 with 95% bootstrap CIs, n=15, scored once. File: `figures/paper/test_metrics.png`.

**Figure 18.** All 15 TEST windows: human gold vs rule / CNN / frozen VideoMAE / fine-tuned n=80 / n=200. File: `figures/paper/test_clip_grid.png`.

**Figure 19.** Qualitative TEST cases on pose axis \(x\): TP `gold_016`, FP `gold_017`, FN `gold_018`, TN `gold_024`, with locked 0/1 predictions. File: `figures/paper/error_cases.png`.

**Figure 21.** Sixteen listener face crops per TEST window — the actual RGB input to VideoMAE. Rows: `gold_016` TP, `gold_017` FP, `gold_018` FN, `gold_024` TN. File: `figures/paper/rgb_frame_strips.png` (generated on otter95 from `features/rgb16`).

---

## Appendix only (not headlines)

- `figures/training_loss.jpg` — TRAIN loss by epoch (feature set C).
- `figures/dev_f1_by_epoch.jpg` — DEV F1 by epoch; used for early stopping only.
- `figures/videomae_training_curve.png` — VideoMAE head TRAIN loss / DEV F1; tuning diagnostic (also usable as Figure 13 next to §5.6, with the caution caption above).

---

## Do not use without a warning caption

- `figures/ablation_f1.jpg` — includes feature set D at F1 0 because training diverged. Either omit D in a table instead, or caption: “Set D diverged (`loss = nan`); the zero F1 is not a valid comparison.”
- Any `pilot_*` or `rule_baseline/` figure from an older synthetic run.
- Any schematic that shows VideoMAE with a score other than the locked TEST values (frozen 0.57, fine-tuned 0.82), or that presents full fine-tuning as completed.

---

## Suggested in-text pointers

| Chapter | Figures |
| --- | --- |
| Intro / data | **15** (pose-only 3 s yaw teaser), **15b** (3 s listener faces: nod + shake), 1, 2, 3, 4 |
| Methods | 5, 6, 7, 8, 9 |
| Results | 10, 11, 12, 14 (+13 beside §5.6 and 15 beside §5.7, with caution captions) |
| Appendix | training loss, DEV F1 by epoch, VideoMAE DEV curves |
