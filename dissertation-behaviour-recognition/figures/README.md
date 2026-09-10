# Figures

Thesis plates use **Helvetica Neue Regular** via `src/paper_figure_style.py`
(not Avenir Next, whose first face is Bold). PNG at 300 DPI.

## Current 3 s protocol

| file | role |
| --- | --- |
| `paper/teaser_windowed_heads.png` | GitHub README lead. Listener faces plus Euler on two locked TEST windows |
| `paper/teaser_shake_windowed.png` | Pose-only 3 s yaw chart. Not the GitHub lead |
| `../results/windowed_dev/overleaf_polish/` | Thesis body plates (study design, both gestures, windows, weak supervision, CNN ablations) |
| `../results/windowed_dev/final_figures/figureF_identity_crops.png` | Identity-corrected face crops |

Regenerate body plates with `scripts/plot_overleaf_polish.py`. Regenerate the GitHub lead with `scripts/plot_teaser_windowed_heads.py`. Captions: `paper/CAPTIONS.md`.

The 60 s clip F1 figures below are from an earlier protocol.

Dissertation figures from the executed pipeline. Producers: `scripts/run_full_experiment.py`,
`scripts/plot_gold_visuals.py`, and `scripts/make_figures.py` (which skips gracefully
when a proposal-era pilot input is absent, and creates subfolders on demand).

## Executed-pipeline figures (top level)

| file | source |
| --- | --- |
| `rule_dev_threshold_curve` | DEV threshold sweep for the frozen amplitude rule |
| `example_positive_rotation` / `example_negative_rotation` | rotation-x traces, nod vs unclear window |
| `pseudo_label_distribution` | 80 rule pseudo-labels (70 nod / 10 unclear) |
| `training_loss`, `dev_f1_by_epoch` | 1D CNN training history (epoch/threshold chosen on DEV) |
| `rule_confusion_matrix`, `classifier_confusion_matrix` | TEST confusions (0.67 / 0.70 headlines) |
| `model_comparison_f1`, `ablation_f1` | rule vs CNN; feature-set ablations A-C |
| `gold_label_counts`, `gold_label_distribution`, `gold_split_distribution` | gold-set composition |
| `github_overview` | one-figure pipeline summary |
| `videomae_training_curve` | frozen-head DEV F1 / loss (tuning diagnostic) |
| `videomae_finetuned_training_curve` | fine-tune n=80 DEV F1 / loss (tuning diagnostic) |
| `pipeline_diagram` | offline training + webcam inference schematic |
| `model_comparison_f1` | TEST F1 bars (include CIs in the dissertation caption) |

## Subfolders

| folder | contents |
| --- | --- |
| `gold_visuals/` | annotation visuals from `scripts/plot_gold_visuals.py` (clip overview, label counts, labels by person/split, extracted pose traces) |
| `final_results/` | `pipeline_overview` schematic (no scores) |

Figure captions for the dissertation: `reports/figure_captions.md`.
