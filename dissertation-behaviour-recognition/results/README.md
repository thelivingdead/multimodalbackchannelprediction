# Results index

Everything in this directory is a saved artifact. **TEST is locked** (scored once per
model). The citable comparison is `tables/main_results.md` (pose + VideoMAE, with CIs).

Pose headlines (also in `final_results.csv`):

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

## Reproducing vs. re-scoring

The saved TEST artifacts above are canonical. Re-running
`scripts/run_full_experiment.py` (or `scripts/train_pose_cnn.py`) reproduces the rule
exactly (it is deterministic) and re-runs the CNN pipeline end-to-end; CPU floating
point and threading mean a fresh CNN *retrain* can land one or two TEST windows on the
other side of the decision threshold. The submitted numbers remain the saved ones; see
`reports/repository_validation.md` for the full determinism audit.
