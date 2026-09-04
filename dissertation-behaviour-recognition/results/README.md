# Results index

The thesis headlines are the 3 s windowed protocol (balanced accuracy, clip-level
95% CIs). Locked TEST: shake yaw rule **0.654 [0.525, 0.794]**; nod return-ratio
rule **0.634 [0.576, 0.685]**. An earlier 60 s clip study is stored in the same
tree and is not the GitHub headline.

Everything in this directory is a saved artifact. **TEST is locked** (scored once per
model). The 60 s clip comparison remains in `tables/main_results.md` (pose + VideoMAE,
with CIs).

Earlier 60 s pose numbers (also in `final_results.csv`):

| method | split | precision | recall | F1 | confusion |
|---|---|---|---|---|---|
| Pose rule (frozen amplitude) | TEST | 0.64 | 0.70 | **0.67** | TP 7, FP 4, TN 1, FN 3 |
| 1D CNN (feature set C) | TEST | 0.70 | 0.70 | **0.70** | TP 7, FP 3, TN 2, FN 3 |

DEV scores (0.86 rule, 0.89 CNN, VideoMAE DEV F1s) are tuning-split values, not headlines.

## Headline artifacts (locked)

| file | produced by | contains |
|---|---|---|
| `rule_selected_config.json` | `scripts/run_full_experiment.py` | frozen rule parameters (axis x, savgol window 11, threshold 16.35 deg) chosen on DEV |
| `rule_test_predictions.csv` | `scripts/run_full_experiment.py` | per-window TEST predictions for the frozen rule |
| `rule_test_metrics.json` | `scripts/run_full_experiment.py` | TEST P/R/F1/accuracy/confusion for the rule (the 0.67 headline) |
| `training_history.csv` | `src/pose_cnn.py` (via `run_full_experiment.py` or `scripts/train_pose_cnn.py`) | per-epoch train loss, DEV F1, DEV balanced accuracy, DEV probability threshold |
| `classifier_test_predictions.csv` | same | per-window TEST predictions of the best-DEV-epoch CNN |
| `classifier_test_metrics.json` | same | TEST P/R/F1/accuracy/confusion for the CNN (the 0.70 headline) |
| `ablation_results.csv` / `ablation_results.md` | same | feature-set ablations A-D and a 10-epoch budget ablation (DEV-selected, TEST-scored) |

## Supporting artifacts

| file | produced by | contains |
|---|---|---|
| `final_results.csv` | assembled from the saved metrics above | the citable 4-row results table (rule and CNN, DEV and TEST) |
| `final_results_summary.json` / `.md` | `scripts/run_full_experiment.py` | end-of-run summary (paths, split sizes, headline metrics) |
| `experiment_config.json` | `scripts/run_full_experiment.py` | paths, split sizes, feature counts, and status flags for the run |
| `pseudo_labels.csv` | `src/pose_cnn.py` | the 80 rule pseudo-labels (70 nod / 10 unclear) used to train the CNN |
| `gold_dataset_summary.json` / `.csv` | `scripts/run_full_experiment.py` | the 30-window gold dataset (15 DEV / 15 TEST, window = 45 s before to 15 s after the annotated nod) |
| `gold_annotation_summary.json` | generated from `data/gold/annotation_sheet.csv` | annotation counts, side balance, event durations; clip-level protocol note |
| `rule_dev_threshold_search.csv` | `scripts/run_full_experiment.py` | 63-candidate threshold sweep on DEV with full confusion counts per threshold |
| `feature_quality.csv` | `scripts/run_full_experiment.py` | per-clip feature coverage (streams yielding fewer than 2 of 3 windows are excluded) |
| `emoca_stream_status.json` | `scripts/run_full_experiment.py` | EMOCA stream availability per clip |
| `storage_before.json` / `storage_after.json` | `scripts/run_full_experiment.py` | disk-usage snapshots showing the no-download constraint held |
| `model_comparison.csv` | `scripts/run_full_experiment.py` / `scripts/make_figures.py` | rule vs CNN side by side |
| `predicted_vs_annotated.csv` | `scripts/export_predicted_vs_annotated.py` | gold sheet joined with predictions |
| `error_analysis.csv` | `scripts/19_error_analysis.py` | per-clip TEST error categories (shared/rule-only/CNN-only FP and FN) |
| `annotation_efficiency.json` | `scripts/20_annotation_efficiency.py` | manual-annotation effort summary (per-video timing was not recorded) |
| `tables/` | `scripts/make_figures.py` | LaTeX-ready tables for the dissertation |

Figures live in the top-level `figures/` directory (see its own README), produced by
`scripts/make_figures.py` and `scripts/plot_gold_visuals.py`.

## VideoMAE (RGB; same DEV/TEST as pose)

| file | produced by | contains |
|---|---|---|
| `videomae_embeddings_meta.json` | `scripts/extract_videomae_embeddings.py` | frozen 768-D embeddings (provenance) |
| `videomae_frozen_head/` | `scripts/train_videomae_head.py` | TEST F1 **0.57** (TP6 FP5 TN0 FN4) |
| `videomae_finetuned/` | `scripts/finetune_videomae.py` | canonical RGB: TEST F1 **0.82** (TP9 FP3 TN2 FN1); `predictions_test.csv` is the 15-row CI file |
| `videomae_finetuned_n200/` | same script, `--out-dir` | scaling ablation: TEST F1 **0.63** (TP6 FP3 TN2 FN4) |
| `pseudo_labels.csv` | frozen rule | 80 TRAIN labels |
| `pseudo_labels_200.csv` | `scripts/scale_pseudo_pool_200.py` | 200 TRAIN labels (first 80 byte-identical) |
| `tables/bootstrap_ci.csv` | `scripts/bootstrap_f1.py` | 1000 resamples, seed 42, **TEST rows only** |
| `tables/main_results.md` | `scripts/make_main_results.py` | six-row master table |

`best_model.pt` is gitignored (~345 MB). Audio/RGB-audio fusion is a **DEV-only** follow-up (`AUDIO_DEV.md`); it does not rescore this TEST table.

## Audio / HuBERT (GOLD DEV only)

GOLD TEST was **not** scored (`AUDIO_TEST_SCORED = NO`, `FUSION_TEST_SCORED = NO`).

| file | contains |
| --- | --- |
| `tables/multimodal_ablation.md` | MFCC / frozen RGB / concat (DEV-selected thresholds) |
| `hubert_dev/` | frozen HuBERT + 50/50 RGB at threshold 0.5 |
| `../figures/paper/audio_dev_mfcc_f1.{png,pdf}` | Fig. O |
| `../figures/paper/audio_dev_hubert_f1.{png,pdf}` | Fig. P |
| `../figures/paper/audio_dev_confusion.{png,pdf}` | Fig. Q |

## Head-shake DEV-only search

DEV comparison (GOLD TEST not scored): `shake/dev_search/` (`comparison_dev.md`; best non-collapsed run `cnn_40_40`, DEV F1 0.818, next to always-shake DEV F1 0.80). Locked shake TEST remains pose rule F1 0.70 under `shake/` (trivial always-shake TEST ~0.64). Axis issue labelled in `tables/shake_results.md` and `tables/task_framing.md`.

## Three-second sliding-window protocol

A second protocol re-annotates the same thirty gold clips as overlapping 3 s windows
(2 s stride), so each 60 s clip yields 29 windows: 435 DEV and 435 TEST. A window is
positive when it overlaps an annotated nod or shake event. This replaces one label per
clip with a detection task at a timescale closer to the behaviour itself.

Three reporting choices differ from the 60 s protocol above. Balanced accuracy is the
headline, because windows are imbalanced (9 % to 16 % positive) and F1 moves with
prevalence. PR AUC is reported as a threshold-free ranking metric, where chance equals
the prevalence rather than 0.5. Confidence intervals resample whole clips, because
neighbouring windows overlap by one second and are not independent.

The 3 s headline is not that nothing works. Shake from yaw is detectable: the
locked TEST amplitude rule scores **0.654 [0.525, 0.794]**, and the DEV
leave-one-clip-out pose CNN scores **0.606 [0.519, 0.680]**. Both intervals
exclude 0.500. The CNN was not scored on TEST. Nod from pitch is not detectable:
the locked TEST rule is **0.549 [0.480, 0.619]**. Same clips, same window, same
annotator. The earlier 60 s shake rule had used roll. The axis audit placed the
behaviour on yaw, and that corrected axis is what now clears chance.

### Amplitude rule baselines (axis and threshold selected on DEV)

| Task | Axis | Split | Positives | Prevalence | Balanced accuracy | 95 % CI (clip bootstrap) | PR AUC |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: |
| Nod | x (pitch) | DEV | 52 / 435 | 0.120 | 0.580 | — (selection split) | 0.131 |
| Nod | x (pitch) | TEST | 69 / 435 | 0.159 | **0.549** | [0.480, 0.619] | 0.207 |
| Shake | y (yaw) | DEV | 39 / 435 | 0.090 | 0.704 | — (selection split) | 0.186 |
| Shake | y (yaw) | TEST | 40 / 435 | 0.092 | **0.654** | [0.525, 0.794] | 0.164 |

Shake clears chance on locked TEST: the interval excludes 0.500. Nod does not, since
its interval contains chance. The two tasks therefore separate, which makes the nod
result a finding about pitch amplitude at this window length rather than a broken
pipeline. The shake axis is y here, chosen by a DEV sweep over all three axes, which
also resolves the coordinate issue recorded for the 60 s shake result.

Artefacts: `windowed_nod/baselines_bacc/` and `windowed_shake/baselines_bacc/`
(`metrics.json`, `predictions.csv`, `threshold_search_dev.csv`, and for shake
`axis_selection_dev.csv`), from `scripts/evaluate_windowed_{nod,shake}_baselines.py`.

### Pose CNN, leave-one-clip-out on DEV

Each fold trains on 14 DEV clips and scores the held-out clip, so no window is scored
by a model that saw its own clip. Normalisation statistics are fitted per fold on the
training clips only. Feature set C (Euler xyz plus first differences), 15 epochs fixed
in advance.

| Task | Out-of-fold PR AUC | Chance | Balanced accuracy at 0.5 |
| --- | ---: | ---: | ---: |
| Nod | 0.131 | 0.120 | 0.523 |
| Shake | 0.172 | 0.090 | 0.606 |

The nod CNN reaches the rule's PR AUC to three decimals, and a clip bootstrap of
PR AUC minus prevalence spans zero for both the rule (−0.009 to +0.038) and the CNN
(−0.009 to +0.054). At 3 s with 52 positives, a one-parameter threshold and a trained
network are indistinguishable, and neither separates from chance on nod.

Artefacts: `windowed_{nod,shake}/pose_cnn_loco_dev/` from
`scripts/crossval_windowed_pose_cnn_dev.py`. Figure:
`windowed_nod/model_comparison_dev.{png,pdf}` with its values in
`model_comparison_dev.json`, from `scripts/plot_windowed_nod_model_comparison.py`.

### Nod fusion search (DEV only)

`windowed_dev/final_fusion_search/` from
`scripts/evaluate_windowed_final_fusion_search_dev.py` ranks the frozen return-ratio
rule, locked Pose CNN OOF, and 1.5 s VideoMAE OOF (mean-aligned onto 3 s). TEST is
not scored. A two-branch Pose CNN (temporal + amplitude/return-ratio MLP) writes
`windowed_nod/pose_cnn_loco_dev_scalar_branch/` on Otter and is added to the ranking
when that OOF file exists.

### RGB / VideoMAE runs: withdrawn

Four leave-one-clip-out VideoMAE runs (nod and shake, 6 and 12 epochs) were completed
and are **not** interpretable, because the crops they consumed were selected per window
as the largest detected face. Every DEV clip shows more than one person, and the
annotator was instructed to watch one side of the frame. An audit against that
instruction found **222 of 435 DEV windows (51 %) cropped the half of the frame the
annotator was told to ignore**, including 22 of the 52 labelled-positive windows, so for
those windows the label and the pixels describe different people.

An earlier stability test (box far from the clip median) counted 127 affected windows
and understated the problem: a box that sits on the wrong person for a whole clip
produces no outliers at all. `gold_003` is the clear case, wrong on 28 of 29 windows
with a single outlier flagged.

The four numbers are tabulated in `windowed_nod/crop_audit/videomae_withdrawn_dev.md`
(and `.csv`) and are excluded from every comparison figure. Whether RGB carries a nod or
shake signal at 3 s is **unresolved**, not answered. Resolving it requires side-aware
crops, implemented behind `--side-aware` in `scripts/fetch_rgb_windows_nod3s.py`, which
rejects detections outside the annotated half and holds one box per clip.

| file | produced by | contains |
| --- | --- | --- |
| `windowed_nod/crop_audit/crop_boxes_dev_nod.csv` | `scripts/check_windowed_rgb_crop_identity.py` | per-window crop box, detection count, watch side, wrong-half flag |
| `windowed_nod/crop_audit/crop_summary_dev_nod.csv` | same | per-clip counts, including `n_wrong_half` and `n_positive_wrong_half` |
| `windowed_nod/crop_audit/crop_contact_sheet_dev_nod.png` | same | every DEV crop, one row per clip, positives ringed |
| `windowed_nod/crop_audit/rgb_crop_defect_dev.{png,pdf}` | `scripts/plot_windowed_rgb_crop_defect.py` | box position over time for four clips, plus per-clip wrong-side rates |
| `windowed_nod/crop_audit/videomae_withdrawn_dev.{md,csv}` | same script, `--write-table` | the four withdrawn VideoMAE runs with their margins over chance |
| `windowed_{nod,shake}/videomae_loco_dev{,_ep12}/` | `scripts/crossval_windowed_videomae_dev.py` | raw fold metrics and out-of-fold predictions for those runs |

Identity-fixed VideoMAE figures for this protocol live in
`windowed_dev/final_figures/`. A is the nod comparison, B is 3 s vs 1.5 s,
C is confusion counts, D is precision-recall, E is clip timelines, and F is
the identity-fixed contact sheet. F uses identifiable RealTalk stills and
stays in the bound dissertation only.

Annotation inputs and window generation for this protocol are documented in
`../data/windowed_annotations/README.md`.

## Reproducing vs. re-scoring

The saved TEST artifacts above are canonical. Re-running
`scripts/run_full_experiment.py` (or `scripts/train_pose_cnn.py`) reproduces the rule
exactly (it is deterministic) and re-runs the CNN pipeline end-to-end; CPU floating
point and threading mean a fresh CNN *retrain* can land one or two TEST windows on the
other side of the decision threshold. The submitted numbers remain the saved ones; see
`reports/repository_validation.md` for the full determinism audit.
