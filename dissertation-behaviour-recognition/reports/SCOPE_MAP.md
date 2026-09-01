# Scope map — planned pipeline vs. what exists here

**Living map.** Rows marked DONE include work finished after the original 20 August 2026 snapshot. Preflight notes that still say “VideoMAE not run” are in `archive/reports/`.

**Originally dated:** 20 August 2026. **Rule:** this file maps the pasted "exact dissertation pipeline" to the repo as it stands. Nothing here is invented. Source of truth for scores: `results/rule_test_metrics.json`, `results/classifier_test_metrics.json`, `results/ablation_results.csv`, `results/final_results_summary.json`.

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
| Model B — VideoMAE only | **DONE (frozen + partial FT + n=200)** | Frozen head 20 Aug 2026: TEST F1 **0.57**. Partial fine-tune (last 4/12 blocks, otter95 `/scratch`): TEST F1 **0.82** (`results/videomae_finetuned/`). Scaling TRAIN=200: TEST F1 **0.63** (`results/videomae_finetuned_n200/`). Canonical RGB = n=80. CIs overlap; not significant. Fusion not run. |
| Model C — fusion | **DEV-ONLY SCRIPT** | Locked TEST fusion was not run. New RGB+audio concat is `scripts/train_av_fusion_dev.py` (GOLD DEV only; refuses TEST; needs Step A PASS + frozen embeddings). No TEST F1. |
| Macro F1 (multiclass) | **NOT DONE** | The task is binary (1 = clear nod, 0 = unclear). Metrics are clip-level binary P/R/F1 from the 2×2 on TEST. No 7-class labels, so no multiclass macro-F1 exists. |
| Augmentation | **PARTIAL** | Pose CNN: none. VideoMAE fine-tune: horizontal flip on TRAIN only. |
| One-shot TEST | **DONE** | TEST scored once; axis/threshold frozen on DEV; CNN epoch and probability threshold chosen on DEV. DEV F1 0.86 (rule) / 0.89 (CNN) are tuning numbers, not headlines; the frozen axis and threshold are recorded in `results/rule_selected_config.json`. |
| Ablations | **DONE (3 of 4)** | Feature ablations A–D run: A x-only F1 0.70; B xyz F1 0.70 (P 0.62, R 0.80); C xyz+deriv F1 0.70 (reported); **D xyz+deriv+expression diverged** (`loss = nan`, F1 0) — invalid, not a result. `results/ablation_results.csv`. |
| Error-analysis categories | **DONE (table)** | Per-clip TEST error table for all 15 clips: `results/error_analysis.csv` (from `scripts/19_error_analysis.py`) with categories and counts — both-correct 7, shared FP 3 (`gold_017/022/028`), shared FN 2 (`gold_018/026`), rule-only FP 1, rule-only FN 1, CNN-only FN 1. Narrative version in `reports/results_chapter_draft.md` §5.5; qualitative FP causes in `reports/discussion_conclusion_draft.md` §8.3. (The empty `figures/error_analysis/` placeholder was removed in the 20 Aug cleanup.) |
| BackchannelAI demo | **NOT DONE** | No demo app, outputs, or screenshots. The placeholder `scripts/22_demo.py` was removed in the 20 Aug cleanup; the proposal-era `web/` prototype at the repo root is unrelated to the submitted experiment. |

**Also true (locked facts):** EMOCA was streamed, never trained; RGB windows lived on otter `/scratch`, not in git; metrics are clip-level P/R/F1, not event IoU 0.30; VideoMAE frozen 0.57 / fine-tune n=80 **0.82** / n=200 **0.63**; no GOLD TEST audio/fusion score; no 7-class, no multiclass macro-F1. Pose+RGB are visual representations, not two sensory modalities.

## Paragraph to paste into the dissertation (scope statement)

> The submitted study is narrower than the original project plan, and the narrowing is deliberate. The approved title is **Predicting Backchannel Events from Multimodal Conversational Signals**. The task is **supervised prediction of the backchannel label associated with a conversational window**, not anticipatory forecasting from pre-event context. Pose and RGB are **visual representation experiments** (same camera), not two sensory modalities. What was executed on locked TEST: binary head-nod recognition on 30 human-labelled RealTalk windows (15 DEV / 15 TEST, TEST scored once): a frozen EMOCA-pose amplitude rule (TEST F1 0.67), a pose 1D CNN on 80 rule pseudo-labels (TEST F1 0.70), a frozen VideoMAE head (TEST F1 0.57), a partial VideoMAE fine-tune of the last four encoder blocks (TEST F1 0.82 at 80 labels; 0.63 at 200 labels). EMOCA was streamed, not trained. Audio/RGB-audio fusion, if run, is GOLD DEV only. Text/transcript models and the seven-class taxonomy were not run. At n=15 all 95% CIs overlap, so the TEST ordering is a point-estimate ranking.

---
*Do not commit unless you choose to.*
