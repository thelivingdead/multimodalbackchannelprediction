# Scripts

Canonical nod TEST numbers come from saved artefacts under `results/`. Do **not** re-run locked TEST inference. Do **not** `--force` directories in `LOCKED_OUT_DIRS` (`scripts/check_split_leakage.py`).

New exploratory work should write under `results/dev/`, `results/experiments/`, or a new named folder — never `results/videomae_finetuned/` or `results/joint/videomae_finetuned/`.

## Canonical pipeline

| Script | Purpose | Inputs | Outputs | TRAIN / DEV / TEST |
| --- | --- | --- | --- | --- |
| `run_full_experiment.py` | Stream EMOCA pose, freeze nod rule on DEV, pseudo-label TRAIN, pose CNN | gold CSV, pose npz | `results/rule_*`, `results/classifier_*`, `results/pseudo_labels.csv` | TRAIN pseudo; DEV tune; TEST scored **once** — do not rerun to overwrite |
| `train_pose_cnn.py` | Pose 1D CNN only (same protocol) | frozen rule + features | classifier metrics/predictions | DEV select; TEST once |
| `check_split_leakage.py` | Fail on split leakage; refuse locked out-dirs | gold + pseudo CSVs | stdout PASS/FAIL | no scoring |
| `fetch_rgb_windows.py` | 16-frame RGB crops | RealTalk video (authorised) | `features/rgb16/` (not in git) | all splits for existing gold/pseudo ids |
| `extract_videomae_embeddings.py` | Frozen VideoMAE 768-D | rgb16 | `data/features/videomae/` | no TEST shopping |
| `train_videomae_head.py` | Frozen-encoder MLP head | embeddings + pseudo labels | `results/videomae_frozen_head/` | DEV select; TEST once (**locked**) |
| `finetune_videomae.py` | Fine-tune last 4 blocks | rgb16 | `results/videomae_finetuned/` (n=80) | DEV select; TEST once (**locked**) |
| `scale_pseudo_pool_200.py` | Grow TRAIN pseudo pool 80→200 | frozen rule | `results/pseudo_labels_200.csv` | does not overwrite n=80 TEST dir |
| `bootstrap_f1.py` | 95% CI from **saved** TEST predictions | `predictions.csv` | `results/tables/bootstrap_ci.csv` | TEST rows only; no model forward |
| `make_main_results.py` | Assemble markdown table | locked json | `results/tables/main_results.md` | reads artefacts only |

## Evaluation

| Script | Purpose |
| --- | --- |
| `10_evaluate_rule.py` | Older event-level pilot evaluator (not the 30-window clip protocol) |
| `19_error_analysis.py` | TEST clip error table from saved predictions |
| `20_annotation_efficiency.py` | Annotation-time summary |
| `majority_baseline.py` | Always-positive clip baseline |

## Analysis / visualisation

`make_figures.py`, `make_dissertation_figures.py`, `make_main_results.py`, `plot_paper_style_figures.py`, `plot_videomae_results.py`, `plot_audio_dev_figures.py`, `make_hubert_figures.py`, `plot_spectral_euler.py`, `plot_teaser_figure.py`, `plot_pipeline_diagram.py`, `plot_gold_visuals.py`, `plot_rgb_frame_strips.py`.

These read existing csv/json/npz. They must not rescore GOLD TEST.

## DEV experiments (TEST refused)

| Script | Purpose |
| --- | --- |
| `annotate_nod_events_dev.py` | Local UI: mark nod events inside 60 s DEV clips. Writes `data/windowed_annotations/`. Does not generate 3 s labels, train, or load TEST |
| `import_nod_event_entry.py` | Compile `nod_event_entry.csv` YouTube clocks into `nod_events_windowed.csv`. No 3 s windows |
| `prepare_dev_annotation_clips.py` | Optional yt-dlp cut of 60 s DEV mp4s into `data/windowed_annotations/clips/`. TEST refused |
| `generate_window_labels.py` | After all DEV clips are reviewed: write `data/windowed_annotations/nod_windows_dev.csv`. TEST refused |
| `plot_window_label_logic.py` | Protocol diagram `results/windowed_dev/window_label_logic.png` (not a result) |
| `generate_window_labels_test.py` | After TEST events are reviewed: write `nod_windows_test.csv` |
| `train_windowed_nod_pose_cnn.py` | 3 s pose CNN in `results/windowed_nod/pose_cnn/`. DEV select; TEST once. Does not write locked 60 s dirs |
| `fetch_rgb_windows_nod3s.py` | 16-frame RGB crops per 3 s window → `features/rgb16_windowed/`. Does not write `features/rgb16/` |
| `train_windowed_nod_videomae.py` | 3 s VideoMAE in `results/windowed_nod/videomae_finetuned/`. Run after the fetch. TEST once |
| `plot_annotated_dev_windows.py` | Clear sliding-window figures for the 15 annotated DEV clips |
| `audio_alignment_check.py` | DEV audio/video alignment |
| `train_audio_baseline_dev.py` | MFCC LR on DEV |
| `train_av_fusion_dev.py` | RGB+audio concat on DEV |
| `run_hubert_dev.py` | Frozen HuBERT + 50/50 fusion on DEV |
| `hubert_train_label_permutation.py` | TRAIN-label permutation on DEV |
| `audit_nod_onsets.py` | Onset audit for temporal correspondence |
| `evaluate_temporal_correspondence_dev.py` | Rule vs annotated nod onsets (DEV) |
| `run_shake_dev_search.py` / `*_dev.py` shake wrappers | Shake search with `test_scored: false` |
| `compare_shake_dev_search.py`, `compare_shake_v2_dev.py` | DEV-only shake comparison |

## Utilities

`00_audit_repository.py`, `01_check_storage.py`, `02_audit_data.py`, `04_normalize_annotations.py`, `05_validate_annotations.py`, `06_create_gold_split.py`, `07_extract_features.py`, `import_annotation_sheet.py`, `export_predicted_vs_annotated.py`, `build_video_shard_index.py`, `check_citations.py`.

## Historical / superseded

Numbered VideoMAE stubs `15_train_videomae.py`, `16_evaluate_videomae.py`, `17_train_fusion.py` are **planning notes**, not the executed otter95 runs. Executed VideoMAE is `train_videomae_head.py` / `finetune_videomae.py`.

Shake TEST trainers (`train_shake_cnn.py`, `finetune_videomae_shake.py`, …) wrote locked `results/shake/` dirs. Do not `--force` those dirs.

Earlier seven-class demo scripts live under `archive/`.
