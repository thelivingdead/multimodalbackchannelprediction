# Figures

Dissertation figures from the executed pipeline. PNG at 300 DPI for the dissertation,
JPG at 300 DPI for GitHub. Producers: `scripts/run_full_experiment.py`,
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

## Subfolders

| folder | contents |
| --- | --- |
| `gold_visuals/` | annotation visuals from `scripts/plot_gold_visuals.py` (clip overview, label counts, labels by person/split, extracted pose traces) |
| `final_results/` | `pipeline_overview` schematic (no scores) |

Figure captions for the dissertation: `reports/figure_captions.md`.
