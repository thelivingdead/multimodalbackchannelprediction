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
| later | videomae/, ablations/, *_frames | skipped until those CSVs/mp4s exist | 7–8 | — | result files |
