# Scope map — planned pipeline vs. what exists here

**Date:** 20 August 2026. **Rule:** this file maps the pasted "exact dissertation pipeline" to the repo as it stands. Nothing here is invented. Source of truth for scores: `results/rule_test_metrics.json`, `results/classifier_test_metrics.json`, `results/ablation_results.csv`, `results/final_results_summary.json`.

## DONE / PARTIAL / NOT DONE

| Pipeline bullet (as pasted) | Status | What actually exists in this repo |
| --- | --- | --- |
| Gold 30 clips, 15 DEV / 15 TEST | **DONE** | 30 human-labelled RealTalk windows, 15/15 split, zero video overlap. 19 clear nod / 11 unclear; LEFT/RIGHT 15/15. `results/gold_dataset_summary.json`, `data/gold/annotation_sheet.csv`, `data/splits/`. |
| Annotation time | **NOT DONE** | `data/gold/annotation_log.csv` has an `annotation_time_s` column, but it is **empty** for all 30 rows (notes say `online_watch`). `scripts/make_figures.py` can plot it but skips when empty. No timing numbers exist. |
| FLAME rules: nod | **DONE** | `src/rules/nod.py` implemented, tuned on DEV (axis x, threshold 16.35°), scored once on TEST: P 0.64, R 0.70, F1 **0.67** (TP7 FP4 TN1 FN3). `results/rule_selected_config.json`, `results/rule_test_metrics.json`. |
| FLAME rules: shake, tilt | **PARTIAL** | `src/rules/head_shake.py` (yaw reuse) and `src/rules/head_tilt.py` (roll) are implemented but **never scored** — their own docstrings say "not scored until gold shake/tilt labels exist". No gold labels, no metrics. |
| FLAME rules: lean, eyebrow | **NOT DONE** | `src/rules/lean.py` and `src/rules/eyebrow_raise.py` are stubs: `supported = False`, `detect()` returns `[]` ("NOT implemented until translation/depth / expression key is VERIFIED"). |
| Pseudo-labels under 25 GB | **DONE** | 80 pseudo TRAIN clips (70 nod / 10 unclear) labelled by the frozen rule. EMOCA streamed, not saved: disk free 22.94 GB before → 22.93 GB after. `results/pseudo_labels.csv`, `results/storage_before.json`, `results/storage_after.json`, `results/emoca_stream_status.json` (81 tar members seen, none saved). |
| Model A — FLAME/pose only | **DONE** | 1D CNN (3× Conv1d, CPU PyTorch) on EMOCA Euler xyz + first differences (feature set C, 6-D, 128 steps), trained on the 80 pseudo-labels, epoch 9 and threshold 0.45 chosen on DEV. TEST P 0.70, R 0.70, F1 **0.70** (TP7 FP3 TN2 FN3). `results/classifier_test_metrics.json`. Implementation: `src/pose_cnn.py` (entry points `scripts/run_full_experiment.py`, `scripts/train_pose_cnn.py`). |
| Model B — VideoMAE only | **NOT DONE** | Never started: otter had ~6.5 GB free on a 25 GB quota after CPU PyTorch; video shards + checkpoint do not fit. `scripts/15_train_videomae.py` / `scripts/16_evaluate_videomae.py` are documented planned-experiment notes (no implementation, nothing downloaded); plan in `reports/videomae_preflight_lab.md`. No score exists. Future work, not a TEST failure. |
| Model C — fusion | **NOT DONE** | `scripts/17_train_fusion.py` is a documented planned-experiment note; `configs/fusion.yaml` (`status: planned_not_run`) was never run. No score exists. |
| Macro F1 (multiclass) | **NOT DONE** | The task is binary (1 = clear nod, 0 = unclear). Metrics are clip-level binary P/R/F1 from the 2×2 on TEST. No 7-class labels, so no multiclass macro-F1 exists. |
| Augmentation | **NOT DONE** | No augmentation anywhere (no jitter/flip/noise/shift in `scripts/run_full_experiment.py` or `src/`). Training used standardisation, class weighting (`pos_weight` 10/70), early stopping, and a DEV threshold sweep only. |
| One-shot TEST | **DONE** | TEST scored once; axis/threshold frozen on DEV; CNN epoch and probability threshold chosen on DEV. DEV F1 0.86 (rule) / 0.89 (CNN) are tuning numbers, not headlines; the frozen axis and threshold are recorded in `results/rule_selected_config.json`. |
| Ablations | **DONE (3 of 4)** | Feature ablations A–D run: A x-only F1 0.70; B xyz F1 0.70 (P 0.62, R 0.80); C xyz+deriv F1 0.70 (reported); **D xyz+deriv+expression diverged** (`loss = nan`, F1 0) — invalid, not a result. `results/ablation_results.csv`. |
| Error-analysis categories | **PARTIAL** | Per-clip TEST error table exists for all 15 clips (`reports/results_chapter_draft.md` §5.5): shared FPs (`gold_017/022/028`), shared FNs (`gold_018/026`), 3 disagreement clips. `reports/discussion_conclusion_draft.md` §8.3 groups FP causes qualitatively (speech-related head movement, tracking jumps, posture shifts) and flags the off-window protocol case. But there is **no formal category taxonomy with per-category counts**; `figures/error_analysis/` is empty. |
| BackchannelAI demo | **NOT DONE** | No demo app, outputs, or screenshots. The placeholder `scripts/22_demo.py` was removed in the 20 Aug cleanup; the proposal-era `web/` prototype at the repo root is unrelated to the submitted experiment. |

**Also true (locked facts):** EMOCA was streamed, never trained; no video frames on otter; metrics are clip-level P/R/F1, not event IoU 0.30 (event F1 is implemented in `src/metrics.py` but was not this protocol); no VideoMAE, no fusion, no 7-class, no macro-F1 multiclass results exist anywhere in the repo.

## Paragraph to paste into the dissertation (scope statement)

> The submitted study is narrower than the original project plan, and the narrowing is deliberate. Proposal-era notes described a seven-class backchannel taxonomy (nod, shake, tilt, lean, eyebrow raise, neutral) and a multimodal architecture combining VideoMAE with FLAME/EMOCA pose. What was executed and evaluated is a binary, pose-only study on 30 human-labelled RealTalk windows: a frozen EMOCA-pose amplitude rule (TEST precision 0.64, recall 0.70, F1 0.67) and a 1D CNN trained on 80 pseudo-labels produced by that rule (TEST precision 0.70, recall 0.70, F1 0.70), with the TEST set scored exactly once. VideoMAE and fusion were not run, because the lab account used for pose extraction has a 25 GB quota and about 6.5 GB remained after a CPU PyTorch install, which cannot hold RealTalk video shards plus a VideoMAE checkpoint; EMOCA itself was streamed from official pickles, not trained. These are scope and resource decisions, not failed experiments: no VideoMAE, fusion, seven-class, or multiclass macro-F1 score is reported, and none is claimed. They are left as future work for a machine with sufficient storage.

---
*Do not commit unless you choose to.*
