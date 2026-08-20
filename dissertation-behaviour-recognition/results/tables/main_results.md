# Main results (TEST, scored once per model under the frozen protocol)

Real values only, read from the saved artefacts listed per row; `N/A` = not run (or no saved predictions.csv for the CI).

| model | input | supervision | train_n | precision | recall | f1 | accuracy | f1_ci_lo | f1_ci_hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rule baseline | EMOCA head rotation | None (DEV-tuned thresholds) | N/A | 0.6364 | 0.7 | 0.6667 | 0.5333 | 0.3529 | 0.8696 |
| Pose CNN raw | EMOCA pose xyz (128-step resampled) | 80 rule pseudo-labels | 80 | 0.6154 | 0.8 | 0.6957 | 0.5333 | N/A | N/A |
| Pose CNN xyz_deriv | EMOCA pose xyz + derivatives | 80 rule pseudo-labels | 80 | 0.7 | 0.7 | 0.7 | 0.6 | 0.4 | 0.8889 |
| Frozen VideoMAE head | RGB 16x224x224 face crops (MCG-NJU/videomae-base) | 80 rule pseudo-labels | 80 | 0.5455 | 0.6 | 0.5714 | 0.4 | 0.2353 | 0.75 |
| Fine-tuned VideoMAE (last 4 blocks) | RGB 16x224x224 face crops (MCG-NJU/videomae-base) | 80 rule pseudo-labels | 80 | 0.75 | 0.9 | 0.8182 | 0.7333 | 0.6 | 0.963 |

Sources: Rule baseline <- `results/rule_test_metrics.json`; Pose CNN raw <- `results/ablation_results.csv` (feature_set `xyz`); Pose CNN xyz_deriv <- `results/classifier_test_metrics.json`; Frozen VideoMAE head <- `results/videomae_frozen_head/metrics.json`; Fine-tuned VideoMAE <- `results/videomae_finetuned/metrics.json`; CIs <- `results/tables/bootstrap_ci.csv` (1000 resamples, seed 42, from saved TEST predictions).
