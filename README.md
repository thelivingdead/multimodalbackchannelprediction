# Predicting Backchannel Events from Multimodal Conversational Signals

MSc dissertation, Divya Bisht  
Centre for Vision, Speech and Signal Processing (CVSSP), University of Surrey, 2026

This repository recognises listener head nods (primary) and head shakes in 60 s conversational windows from Columbia [RealTalk](https://realtalk.cs.columbia.edu/) (Geng et al., 2023). The approved title says prediction. The executed task is clip-level recognition of a behaviour inside an observed window, not anticipatory forecasting from pre-event context.

![Listener backchannel teaser](dissertation-behaviour-recognition/figures/paper/teaser_backchannel.jpg)

Two locked TEST windows, listener in blue and partner in orange. Top: `gold_020`, labelled clear nod. Bottom: `gold_024`, labelled unclear. The pose trace is EMOCA rotation x against the frozen nod-rule threshold 16.35°.

Headline on locked GOLD TEST, n = 15, scored once: fine-tuned VideoMAE, last 4 blocks, 80 pseudo-labelled TRAIN clips, **F1 0.82**. The always-positive baseline scores 0.80 on the same fifteen windows, so read the headline as the highest point estimate rather than a demonstrated win.

![Nod GOLD TEST F1](dissertation-behaviour-recognition/figures/paper/nod_test_f1.png)

Full table: [`results/tables/main_results.md`](dissertation-behaviour-recognition/results/tables/main_results.md).

---

## Motivation

Listener nods and shakes are backchannel cues. Automatic recognition supports conversational agents and the analysis of dyadic talk. The practical constraint here is the small number of gold labels. A pose rule supplies noisy pseudo-labels on TRAIN, models are selected on 15 DEV windows, and TEST is locked and scored once.

## Dataset and annotation

- **Dataset:** Columbia RealTalk. Videos are not redistributed here.
- **Gold set:** 30 human-labelled 60 s listener windows, 15 DEV and 15 TEST, disjoint by `sample_id` and `video_id`.
- **Behaviours:** nod `label` 1/0, shake `shake_label` 1/0 on the same windows.
- **TRAIN:** 80 rule-derived pseudo-labels, later 200 in an ablation.
- **Signals:** EMOCA/FLAME head pose (Euler) and RGB 16-frame face crops. These are two encodings of one camera rather than two independent senses. Audio is the mixed conversation soundtrack and is used on GOLD DEV only.

## Method

```text
Manual gold annotation          30 windows (DEV/TEST)
        ↓
Pose extraction                 EMOCA Euler, used not trained
        ↓
Rule-based pose baseline        DEV-tuned, TEST once
        ↓
Pseudo-labels on TRAIN          frozen rule on unlabelled clips
        ↓
Temporal pose CNN               TRAIN=pseudo, DEV select, TEST once
        ↓
Frozen VideoMAE baseline        RGB crops, frozen encoder
        ↓
Fine-tuned VideoMAE             last 4 blocks, n=80 headline
        ↓
Audio / multimodal DEV          MFCC, HuBERT, fusion (DEV only)
        ↓
Locked TEST evaluation          n=15, scored once per system
```

Gold DEV and TEST carry human labels throughout. TRAIN uses the pose rule as a noisy teacher, and TEST targets are always human.

## Experimental protocol

| Split | n | Role |
| --- | ---: | --- |
| TRAIN | 80 (200 in ablation) | Pseudo-labels from the frozen pose rule |
| DEV | 15 gold | Axis, threshold, epoch, probability threshold |
| TEST | 15 gold | Scored once, never used to choose models |

TEST is small enough that one clip moves F1 by several points. Interval estimates are in `results/tables/bootstrap_ci.csv`, and they overlap across every pair of systems.

## TEST results, nod

| Model | Input | TRAIN | P | R | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Always-positive baseline | n/a | n/a | 0.67 | 1.00 | 0.80 |
| Pose rule (axis x, τ = 16.35°) | EMOCA Euler | n/a | 0.64 | 0.70 | 0.67 |
| Pose 1D CNN | Euler + derivatives | 80 | 0.70 | 0.70 | 0.70 |
| Frozen VideoMAE head | RGB 16-frame crops | 80 | 0.55 | 0.60 | 0.57 |
| Fine-tuned VideoMAE (last 4 blocks) | RGB 16-frame crops | 80 | 0.75 | 0.90 | **0.82** |

The canonical RGB result is n = 80 at F1 0.82. Sources: `results/majority_baseline/metrics.json`, `results/rule_test_metrics.json`, `results/classifier_test_metrics.json`, `results/videomae_frozen_head/metrics.json`, `results/videomae_finetuned/metrics.json`.

### Ablation: 80 against 200 pseudo-labels

The same fine-tune recipe with 200 pseudo-labelled TRAIN clips scored TEST F1 0.63. Increasing the pseudo-labelled set did not monotonically improve performance. One plausible explanation is that extra rule labels also added teacher noise. At n = 15 this is an observed trend rather than a proven cause. Artefacts: `results/videomae_finetuned_n200/`.

## TEST results, head shake

Same thirty windows, same frozen protocol, a different label. The canonical result is the pose rule at F1 0.70, against an always-shake baseline of 0.64. Both VideoMAE variants scored 0.60, so RGB did not beat pose on this task.

![Shake GOLD TEST F1](dissertation-behaviour-recognition/figures/paper/shake_test_f1.png)

Locked TEST used Euler z (roll) at τ ≈ 11.15°. A later coordinate audit found that after `as_euler('xyz')`, geometric left to right shake sits primarily on y (yaw) and nod on x (pitch). TEST had already been scored, so the z-axis result was kept as recorded and was not rescored. Full audit: `results/shake/dev_search/axis_audit.md`, with nod timing caveats in `reports/annotation_audit.md`.

## DEV multimodal experiments

Audio, concat fusion, and frozen HuBERT were run on `gold_001` to `gold_015` only, and the scripts refuse GOLD TEST. These are development diagnostics with no locked TEST score behind them. Tables: `results/tables/multimodal_ablation.md`.

A DEV-only timing check of the nod rule against annotated onsets is in `results/temporal_dev/`. It is a correspondence check, since the annotations mark one gesture per window, so it does not yield an event-detection F1.

## Repository structure

Study package: [`dissertation-behaviour-recognition/`](dissertation-behaviour-recognition/).

```
dissertation-behaviour-recognition/
├── data/gold/          human labels
├── scripts/            see scripts/README.md
├── src/                metrics, pose CNN, audio I/O
├── results/            locked json/csv (do not overwrite TEST dirs)
├── figures/paper/      dissertation figures
├── reports/            audits and chapter drafts
└── tests/              split, lock, and label invariants
archive/                superseded demos and preflight notes
```

## Reproduction

Saved features and prediction CSVs are included where permitted, so the metrics can be recomputed without the videos. Full end-to-end training needs authorised RealTalk access, and the VideoMAE fine-tune needs a GPU environment. Raw-data reproducibility is therefore partial, and the repository does not claim otherwise.

Lightweight validation, with no training and no TEST inference:

```bash
cd dissertation-behaviour-recognition
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python scripts/check_split_leakage.py
```

`reports/repository_validation.md` recomputes clip F1 from the stored TEST prediction CSVs.

VideoMAE is optional and expensive. Do not overwrite the locked result directories:

```bash
pip install -r requirements-video.txt
```

Audio DEV extras: `pip install -r requirements-audio.txt`.

Canonical scripts: [`scripts/README.md`](dissertation-behaviour-recognition/scripts/README.md).

## Validation

- TEST lock: `scripts/check_split_leakage.py`, where `LOCKED_OUT_DIRS` covers the nod, shake, and joint VideoMAE TEST directories.
- Split leakage: `reports/split_integrity.md`.
- Metric consistency: `reports/repository_validation.md`.
- Figure captions: `figures/paper/CAPTIONS.md`.

## Limitations

- TEST n = 15. All bootstrap intervals overlap, so no significance is claimed.
- TRAIN labels come from a rule teacher, and raising the pool to 200 lowered the locked score.
- Pose and RGB share one camera.
- Audio and fusion results exist on DEV only.
- Locked shake TEST used roll (z) where a later audit places the behaviour on yaw (y).
- Two gold nod clocks sit outside the analysed window, recorded in `reports/annotation_audit.md` and kept as annotated rather than silently repaired.

## Ethics, dataset, licensing

Code and gold tables are MIT licensed, see `LICENSE`. Cite this work via `CITATION.cff`.

Columbia RealTalk video, audio, and EMOCA releases are not in this repository. Use of RealTalk follows its own terms: Geng et al. (2023), https://realtalk.cs.columbia.edu/

Some figures include cropped listener faces from RealTalk. Redistribution of those stills is governed by the RealTalk terms and is not licensed from this repository. Where a venue forbids identifiable frames, use the pose traces and aggregate plots instead.

## Citation

See `CITATION.cff`. RealTalk: Geng, S., et al. (2023). *RealTalk*. https://realtalk.cs.columbia.edu/
