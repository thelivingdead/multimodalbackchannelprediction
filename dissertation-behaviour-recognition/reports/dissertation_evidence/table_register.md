# Table register

| table | file | purpose |
| --- | --- | --- |
| T-gold | data/gold/events.csv / data/gold_annotations.csv | human 1/0 windows |
| T-rule-test | results/rule_test_metrics.json | **headline** rule TEST P/R/F1 0.64/0.70/0.67 |
| T-cnn-test | results/classifier_test_metrics.json | **headline** CNN TEST P/R/F1 0.70/0.70/0.70 |
| T-rule-cfg | results/rule_selected_config.json | frozen axis x, thr 16.35°, DEV-only |
| T-ablation | results/ablation_results.csv | A–C comparable; **omit D** |
| T-pseudo | results/pseudo_labels.csv | 70/10 automatic TRAIN labels |
| T-compare | results/model_comparison.csv | rule vs CNN TEST (regenerate if CNN row missing) |
| T-time | data/gold/annotation_log.csv | annotation log |
| T-pilot | results/pilot_nod_rule_metrics.json | **do not report** (synthetic / old) |
