# Predicting Backchannel Events from Multimodal Conversational Signals

MSc dissertation package — Divya Bisht, Institute for People-Centred AI, University of Surrey, 2026.

Listener **head-nod** and **head-shake** recognition on Columbia RealTalk (Geng et al., 2023). The GitHub front page is the repository root [`README.md`](../README.md).

The task is **supervised prediction of the backchannel label associated with a conversational window**, not anticipatory forecasting from pre-event context. **Pose and RGB are visual representation experiments** (two encodings of the same camera), not two sensory modalities. Audio/fusion is **GOLD DEV only**. Text/transcript models are future work.

**Nod** = gold 1/0. **Shake** = `shake_label` on the same 30 windows. **Pose** = EMOCA Euler (used, not trained). **RGB** = 16×224×224 face crops → VideoMAE.

![Nod TEST F1](figures/paper/nod_test_f1.png)

**Nod headline (TEST n = 15, once):** fine-tuned VideoMAE last 4 blocks, 80 TRAIN, **F1 0.82**.

![Shake TEST F1](figures/paper/shake_test_f1.png)

**Shake headline (TEST n = 15, once):** pose rule axis **z**, **F1 0.70** (always-shake **0.64**).

## Labels and protocol

| Value | Meaning |
| --- | --- |
| `1` | Clear nod (only this class is a gold positive) |
| `0` | Unclear / not a nod |

- **30** human-labelled windows: **15 DEV** (tuning) / **15 TEST** (scored **once**, never used to train or pick a threshold).
- TRAIN = frozen-rule **pseudo-labels**, not gold.
- Metric: **clip-level** P/R/F1. Event F1 at IoU 0.30 exists in `src/metrics.py` but is **not** this protocol.
- RealTalk: **p0 = LEFT**, **p1 = RIGHT**, 25 fps.

Gold: `data/gold/annotation_sheet.csv`, `data/gold/shake_annotation_sheet.csv`, `data/gold/events.csv`. Splits: `data/splits/gold_dev.txt`, `data/splits/gold_test.txt`.

## Locked TEST table (n = 15)

Master file: `results/tables/main_results.md` (with bootstrap CIs).

| method | TRAIN | P | R | F1 | 95% CI |
| --- | --- | --- | --- | --- | --- |
| Pose rule (axis x, 16.35°) | — | 0.64 | 0.70 | **0.67** | [0.35, 0.87] |
| Pose 1D CNN | 80 | 0.70 | 0.70 | **0.70** | [0.40, 0.89] |
| Frozen VideoMAE head | 80 | 0.55 | 0.60 | **0.57** | [0.24, 0.75] |
| Fine-tuned VideoMAE (last 4/12 blocks) | 80 | 0.75 | 0.90 | **0.82** | [0.60, 0.96] |
| Fine-tuned VideoMAE (scaling) | 200 | 0.67 | 0.60 | **0.63** | [0.31, 0.84] |

Canonical RGB result: **80-clip fine-tune, F1 0.82**. Pose and RGB in this table are **visual representations of the same video**, not an audio+vision multimodal result. Do not headline DEV F1. Do not claim statistical significance (CIs overlap). **Nod headline: RGB fine-tune F1 0.82.**

## Head-shake TEST (n = 15, scored once)

Same 30 gold videos; `shake_label`; TEST scored **once**; DEV tuning only. TRAIN = frozen-rule **pseudo-labels** (**75/5**), not gold. No shake CIs.

**Shake headline: pose rule F1 0.70.** RGB did not beat pose. VideoMAE F1 **0.60** is below the trivial **always-predict-shake** baseline **0.64**. **AXIS ISSUE:** after `as_euler('xyz')`, **x** = pitch/nod, **y** = yaw/shake, **z** = roll; locked TEST used **z**; new TRAIN ranks **y**. Do not rescore TEST. Table: `results/tables/shake_results.md`.

| method | TRAIN | P | R | F1 |
| --- | --- | --- | --- | --- |
| Pose rule (axis z, τ≈11.15°) | — | 0.54 | 1.00 | **0.70** |
| Pose 1D CNN | 80 (75/5) | 0.47 | 1.00 | **0.64** |
| Always-predict-shake | — | 0.47 | 1.00 | **0.64** |
| Frozen VideoMAE head | 80 (75/5) | 0.46 | 0.86 | **0.60** |
| Frozen VideoMAE head (balanced TRAIN) | 10 | 0.47 | 1.00 | **0.64** |
| Fine-tuned VideoMAE (last 4/12 blocks) | 80 (75/5) | 0.46 | 0.86 | **0.60** |
| Fine-tuned VideoMAE (balanced TRAIN) | 10 | 0.50 | 0.86 | **0.63** |
| Fine-tuned VideoMAE (DEV-threshold protocol) | 80 (75/5) | 0.60 | 0.43 | **0.50** |
| Late fusion pose + RGB | — | 0.54 | 1.00 | **0.70** |

Late fusion F1 **0.70** equals the pose rule. Joint two-head TEST: shake F1 **0.64** (always-shake, TP 7 FP 8); nod F1 **0.70** (does not beat dedicated nod **0.82**). Joint frozen-head TEST: nod F1 **0.53** (P 0.56 R 0.50), shake F1 **0.67** (P 0.50 R 1.00).

Locked json: `results/shake/` and `results/joint/`.

## Shake DEV-only search (TEST not scored)

GOLD TEST was scored **once** and is **locked**. This search used GOLD DEV only (`test_scored: false`). It is **not** a replacement TEST number. Published shake TEST headline remains **pose rule F1 0.70**. DEV CNN 40/40 is a new protocol result (balanced pseudo-labels), reported next to the trivial **always-shake DEV** baseline (F1 **0.80**).

Source: `results/shake/dev_search/comparison_dev.md`. Best config: pose 1D CNN, 40/40, `results/shake/dev_search/cnn_40_40`. DEV n=15 (10 shake+ / 5 shake−). Always-shake on DEV is F1 **0.80**. Frozen VideoMAE 40/40 was conservative (F1 0.462). Fine-tune last-4 40/40 (F1 0.706) did not beat the CNN on DEV.

| method | TRAIN | P | R | F1 | confusion | collapse |
| --- | --- | --- | --- | --- | --- | --- |
| Pose 1D CNN (best DEV) | 40/40 | 0.75 | 0.90 | **0.818** | TP 9, FP 3, TN 2, FN 1 | false |
| Always-predict-shake | — | 0.67 | 1.00 | **0.80** | TP 10, FP 5, TN 0, FN 0 | true |
| Frozen VideoMAE head | 40/40 | 1.00 | 0.30 | **0.462** | TP 3, FP 0, TN 5, FN 7 | false |
| Fine-tuned VideoMAE (last 4 blocks) | 40/40 | 0.86 | 0.60 | **0.706** | TP 6, FP 1, TN 4, FN 4 | false |

After `as_euler('xyz')`: **x** = pitch/nod, **y** = yaw/shake, **z** = roll. Locked TEST rule used **z** (τ≈11.15°); new TRAIN ranks on **y**. Do not retune or rescore locked GOLD TEST. Do not `--force` existing `dev_search/` dirs. Axis audit: `results/shake/dev_search/axis_audit.md` and `figures/shake_axis_audit/`.

## Layout

```
configs/     rule YAML; VideoMAE notes
data/gold/   human labels
data/splits/ DEV / TEST video ids
scripts/     canonical entry points (see below)
src/         events, metrics, nod rule, pose_cnn.py
results/     locked json/csv (see results/README.md)
figures/     dissertation figures
reports/     chapter drafts and audits
tests/       split/label invariants
```

**Canonical scripts (use these):**

| Script | Role |
| --- | --- |
| `scripts/run_full_experiment.py` | Stream EMOCA, frozen rule, pose CNN, ablations |
| `scripts/train_pose_cnn.py` | Pose CNN stage only |
| `scripts/check_split_leakage.py` | Must PASS before VideoMAE train |
| `scripts/fetch_rgb_windows.py` | 16-frame face crops (lab `/scratch`) |
| `scripts/extract_videomae_embeddings.py` | Frozen 768-D embeddings |
| `scripts/train_videomae_head.py` | Frozen-backbone MLP head |
| `scripts/finetune_videomae.py` | Partial fine-tune (last 4 blocks). Default out-dir = n=80; n=200 uses `--out-dir results/videomae_finetuned_n200` |
| `scripts/scale_pseudo_pool_200.py` | 80 → 200 pseudo pool (does not overwrite n=80) |
| `scripts/bootstrap_f1.py` | TEST-only 95% CIs (never full `predictions.csv`) |
| `scripts/make_main_results.py` | Writes `results/tables/main_results.md` |
| `scripts/audio_alignment_check.py` | Step A: 3–5 nod windows, source A/V check (DEV; refuses TEST) |
| `scripts/train_audio_baseline_dev.py` | Step B: audio LR on TRAIN + GOLD DEV (refuses TEST) |
| `scripts/train_av_fusion_dev.py` | Step C: frozen VideoMAE + audio concat, GOLD DEV only |

Otter audio commands: `AUDIO_DEV.md`. Numbered `scripts/15_*.py` / `16_*.py` / `17_*.py` are **planning stubs**, not the executed VideoMAE runs.

Large binaries stay off git: videos, `.pkl`, `best_model.pt`, RGB `.npz`, `.venv`.

## Pipeline (executed)

1. Gold annotation — 30 windows (done)
2. EMOCA pose **streamed** (not trained; `emoca.tar.gz` never saved)
3. Frozen rule on DEV — axis **x**, **16.35°** → TEST F1 **0.67**
4. Pseudo-labels — 80 clips (70 nod / 10 unclear); later 200 for scaling (`results/pseudo_labels_200.csv`, original 80 rows unchanged)
5. Pose 1D CNN — TEST F1 **0.70**
6. Frozen VideoMAE head — TEST F1 **0.57**
7. Fine-tune last 4 encoder blocks (otter95, `/scratch` CUDA) — TEST F1 **0.82** (canonical RGB)
8. Same recipe, TRAIN = 200 — TEST F1 **0.63** (ablation; n=80 artefacts not overwritten)

Do not train or retune on GOLD TEST. Do not pass `--force` on a finished VideoMAE run. Audio/fusion scripts default to DEV-only and refuse TEST scoring.

## Audits

- `reports/annotation_audit.md`
- `reports/split_integrity.md`
- `reports/dissertation_evidence/experiment_log.md`
- `reports/repository_validation.md` (pose 0.67 / 0.70)

Chapter drafts to paste into Word: `reports/abstract_intro_lit_draft.md`, `methods_chapter_draft.md`, `results_chapter_draft.md`, `discussion_conclusion_draft.md`.

## Setup

```bash
cd dissertation-behaviour-recognition
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

GPU VideoMAE used lab `/scratch/db01550/venv` (not this CPU venv).

## Citation

Geng, S., et al. (2023). *RealTalk*. https://realtalk.cs.columbia.edu/
