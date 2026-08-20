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
| `rule_selected_config.json` | `scripts/run_full_experiment.py` step "Tune rule" | frozen rule parameters (axis x, savgol window 11, threshold 16.35 deg) chosen on DEV |
| `rule_test_predictions.csv` | `scripts/run_full_experiment.py` step "Score rule on TEST" | per-window TEST predictions for the frozen rule |
| `rule_test_metrics.json` | same | TEST P/R/F1/accuracy/confusion for the rule (the 0.67 headline) |
| `training_history.csv` | `src/pose_cnn.py` (`train_pseudo_cnn`, called from `run_full_experiment.py`) | per-epoch train loss, DEV F1, DEV balanced accuracy, DEV probability threshold |
| `classifier_test_predictions.csv` | same | per-window TEST predictions of the best-DEV-epoch CNN |
| `classifier_test_metrics.json` | same | TEST P/R/F1/accuracy/confusion for the CNN (the 0.70 headline) |
| `ablation_results.csv` / `ablation_results.md` | same | feature-set ablations A-D and a 10-epoch budget ablation (DEV-selected, TEST-scored) |

## Supporting artifacts

| file | produced by | contains |
|---|---|---|
| `final_results.csv` | assembled from the saved metrics above | the citable 4-row results table (rule and CNN, DEV and TEST) |
| `experiment_config.json` | `scripts/run_full_experiment.py` | paths, split sizes, feature counts, and status flags for the run |
| `gold_dataset_summary.json` / `gold_dataset_summary.csv` | `scripts/05_build_gold_windows.py` | the 30-window gold dataset (15 DEV / 15 TEST, window = 45 s before to 15 s after the annotated nod) |
| `gold_annotation_summary.json` | generated from `data/gold/annotation_sheet.csv` | annotation counts, side balance, event durations; clip-level protocol note |
| `rule_dev_threshold_search.csv` | `scripts/run_full_experiment.py` | 63-candidate threshold sweep on DEV with full confusion counts per threshold |
| `feature_quality.csv` | `scripts/04_extract_emoca_features.py` | per-clip feature coverage (streams that yielded fewer than 2 of 3 windows are excluded) |
| `storage_before.json` / `storage_after.json` | run wrapper | disk-usage snapshots proving the no-download constraint held |
| `emoca_stream_status.json` | `scripts/04_extract_emoca_features.py` | EMOCA stream availability per clip |
| `annotation_efficiency.json` | `scripts/20_annotation_efficiency.py` | manual-annotation effort summary (timing fields were not recorded) |
| `figures/` | `scripts/13_compare_and_figures.py` + `scripts/plot_gold_visuals.py` | the six dissertation figures |

## Reproducing vs. re-scoring

The saved TEST artifacts above are canonical. Re-running
`scripts/run_full_experiment.py` (or `scripts/train_pose_cnn.py`) reproduces the rule
exactly (it is deterministic) and reproduces the CNN pipeline end-to-end; CPU floating
point and threading mean a fresh CNN *retrain* can land one or two TEST windows on the
other side of the decision threshold. The submitted numbers remain the saved ones; see
`reports/repository_validation.md` for the full determinism audit.
