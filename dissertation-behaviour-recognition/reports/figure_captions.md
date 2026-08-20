# Figure captions (copy into Word)

All paths are relative to `dissertation-behaviour-recognition/`. Prefer **JPG** for Word. PNG duplicates exist for some gold plots. Do not invent a pipeline figure that includes VideoMAE scores.

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

---

## Appendix only (not headlines)

- `figures/training_loss.jpg` — TRAIN loss by epoch (feature set C).
- `figures/dev_f1_by_epoch.jpg` — DEV F1 by epoch; used for early stopping only.

---

## Do not use without a warning caption

- `figures/ablation_f1.jpg` — includes feature set D at F1 0 because training diverged. Either omit D in a table instead, or caption: “Set D diverged (`loss = nan`); the zero F1 is not a valid comparison.”
- Any `pilot_*` or `rule_baseline/` figure from an older synthetic run.
- Any schematic that shows VideoMAE as a completed stage with a score.

---

## Suggested in-text pointers

| Chapter | Figures |
| --- | --- |
| Data / annotation | 1, 2, 3, 4 |
| Methods | 5, 6, 7, 8, 9 |
| Results | 10, 11, 12 |
| Appendix | training loss, DEV F1 by epoch |
