# Figure register

| figure | file | purpose | chapter | caption draft | source |
| --- | --- | --- | --- | --- | --- |
| P0 | figures/final_results/pipeline_overview.png | pipeline schematic | 5 | Weak-supervision flow; **no scores** | make_figures.py |
| P0b | figures/github_overview.jpg | README hero | — | RealTalk → pose → weak labels → VideoMAE | make_figures.py |
| P1 | figures/pilot_nod/*_pose_gold.png | pitch/yaw/roll + gold | 4/5 | Listener pose with class-1 nods | headpose CSV + events.csv |
| P2 | figures/rule_baseline/*_pitch_trace.png | pitch + gold + predicted | 6 | Real timestamps | headpose + predictions |
| P3 | figures/rule_baseline/event_metrics.png | P/R/F1 | 7 | From JSON, never typed | pilot_nod_rule_metrics.json |
| P4 | figures/rule_baseline/confusion_matrix.png | frame confusion | 7 | tn/fp/fn/tp from JSON | same |
| P5 | figures/annotations/*_gold_timeline.png | gold intervals | 4 | class-1 nods vs time | events.csv |
| P6 | figures/dataset/split_summary.png | split sizes | 4 | video counts / duration | splits + inventory |
| P7 | figures/annotations/class_counts.png | 1 vs 0 counts | 4 | Only class 1 is a positive | events.csv |
| P8 | figures/rule_baseline/dev_grid_heatmap.png | DEV hyperparameter grid | 6 | F1 at IoU 0.30 | rule_nod_dev_grid.csv |
| P9 | figures/rule_baseline/per_video_f1.png | per-video F1 | 7 | Heterogeneity, not a headline | pilot_nod_per_video.csv |
| P10 | figures/rule_baseline/iou_thresholds.png | IoU 0.10/0.30/0.50 | 7 | Primary metric is 0.30 | pilot_nod_iou_thresholds.csv |
| P11 | figures/rule_baseline/pitch_nod_vs_background.png | pitch in nod vs other | 6 | Feature sanity check | pilot_nod_pitch_by_gold.csv |
| P12 | figures/error_analysis/tp_*.png etc. | pitch error windows | 7 | Qualitative TP/FP/FN | predictions + gold |
| R1 | figures/rule_confusion_matrix.jpg | rule TEST confusion | 7 | TP7 FP4 TN1 FN3; F1 0.67 | rule_test_metrics.json |
| R2 | figures/classifier_confusion_matrix.jpg | CNN TEST confusion | 7 | TP7 FP3 TN2 FN3; F1 0.70 | classifier_test_metrics.json |
| R3 | figures/model_comparison_f1.jpg | rule vs CNN TEST F1 | 7 | Headline comparison | final_results_summary.md |
| R4 | figures/pseudo_label_distribution.jpg | 70/10 pseudo labels | 6 | Weak-label bias | pseudo_labels.csv |
| G1 | figures/gold_visuals/label_counts.jpg | gold 19/11 | 4 | Human labels only | gold sheet |
| G2 | figures/gold_visuals/labels_by_split.jpg | 15/15 | 4 | DEV vs TEST counts | gold sheet |
| G3 | figures/gold_visuals/labels_by_person.jpg | LEFT/RIGHT | 4 | p0/p1 | gold sheet |
| G4 | figures/gold_visuals/clip_overview.jpg | 30 windows | 4 | Watch-window overview | gold sheet |
| G5 | figures/gold_visuals/pose_traces_extracted.jpg | pose sanity | 5 | Features exist; no F1 | npz |
| M1 | figures/rule_dev_threshold_curve.jpg | DEV τ search | 5 | **Not a TEST result** | rule_dev_threshold_search.csv |
| M2 | figures/example_positive_rotation.jpg | gold + pose | 5 | Example nod trace | gold npz |
| M3 | figures/example_negative_rotation.jpg | gold − pose | 5 | Example unclear trace | gold npz |
| X1 | figures/ablation_f1.jpg | A–D TEST F1 | — | **D invalid**; prefer table | ablation_results.csv |
| X2 | figures/training_loss.jpg | TRAIN loss | appendix | Not a headline | training_history.csv |
| X3 | figures/dev_f1_by_epoch.jpg | DEV F1 curve | appendix | Tuning only | training_history.csv |

Do not paste older P3–P12 `pilot_*` figures as RealTalk TEST results. VideoMAE rows in make_figures.py were never produced.
