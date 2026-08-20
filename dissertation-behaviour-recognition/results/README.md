# Results index

Everything in this directory is a saved artifact of the submitted experiment. The two
headline TEST numbers are **locked** and are not recomputed when this repository is
cleaned or re-documented:

| method | split | precision | recall | F1 | confusion |
|---|---|---|---|---|---|
| Pose rule (frozen amplitude) | TEST | 0.64 | 0.70 | **0.67** | TP 7, FP 4, TN 1, FN 3 |
| 1D CNN (feature set C) | TEST | 0.70 | 0.70 | **0.70** | TP 7, FP 3, TN 2, FN 3 |

`final_results.csv` is the single table to cite; it points back to the per-file sources
below. DEV scores (0.86 rule, 0.89 CNN) are tuning-split values, not headlines.

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

## VideoMAE frozen-head follow-up (20 Aug 2026; not a headline)

A first frozen-VideoMAE run landed after the pose-only submission artifacts were locked.
It is indexed here for completeness; it does **not** change the locked headlines above.

| file | produced by | contains |
|---|---|---|
| `videomae_embeddings_meta.json` | `scripts/extract_videomae_embeddings.py` | frozen `MCG-NJU/videomae-base` embeddings, 768-d, 110 clips (extraction provenance) |
| `videomae_frozen_head/training_history.csv` | `scripts/train_videomae_head.py` | per-epoch loss / DEV F1 / DEV threshold for the MLP head |
| `videomae_frozen_head/predictions.csv` | same | per-window TEST probabilities and predictions |
| `videomae_frozen_head/metrics.json` | same | DEV F1 0.90; TEST F1 0.57 (P 0.55, R 0.60; TP 6, FP 5, TN 0, FN 4) |

Metrics were recomputed from `predictions.csv` during post-cleanup validation and match
`metrics.json` exactly. Fine-tuning and fusion remain not run.

## Reproducing vs. re-scoring

The saved TEST artifacts above are canonical. Re-running
`scripts/run_full_experiment.py` (or `scripts/train_pose_cnn.py`) reproduces the rule
exactly (it is deterministic) and re-runs the CNN pipeline end-to-end; CPU floating
point and threading mean a fresh CNN *retrain* can land one or two TEST windows on the
other side of the decision threshold. The submitted numbers remain the saved ones; see
`reports/repository_validation.md` for the full determinism audit.
