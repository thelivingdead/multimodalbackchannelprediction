# Predicting Backchannel Events from Multimodal Conversational Signals

MSc dissertation package — Divya Bisht, Centre for Vision, Speech and Signal Processing (CVSSP), University of Surrey, 2026.

The GitHub front page is the repository root [`README.md`](../README.md).

![Listener backchannel teaser](figures/paper/teaser_backchannel.jpg)

**Task:** clip-level **listener head-nod / head-shake recognition** on Columbia RealTalk, not anticipatory forecasting. Pose and RGB are two encodings of the **same camera**. Audio is **GOLD DEV only**.

**Nod headline (TEST n = 15, once):** fine-tuned VideoMAE last 4 blocks, 80 TRAIN, **F1 0.82**.

![Nod TEST F1](figures/paper/nod_test_f1.png)

| Model | TEST F1 |
| --- | ---: |
| Pose rule | 0.67 |
| Pose CNN | 0.70 |
| Frozen VideoMAE | 0.57 |
| Fine-tuned VideoMAE | **0.82** |

n=200 fine-tune TEST F1 **0.63** is a scaling ablation, not the headline. CIs overlap; no significance claims. Master table: `results/tables/main_results.md`.

**Shake headline (TEST n = 15, once):** pose rule axis **z**, **F1 0.70**. Geometric shake is **y** (yaw); TEST was not rescored after that audit.

Scripts: [`scripts/README.md`](scripts/README.md). Validation: [`reports/repository_validation.md`](reports/repository_validation.md). Captions: [`figures/paper/CAPTIONS.md`](figures/paper/CAPTIONS.md).

## Protocol

- 30 gold windows: 15 DEV / 15 TEST, scored **once**.
- TRAIN = frozen-rule **pseudo-labels**.
- Metric: clip-level P/R/F1.
- RealTalk: p0 = LEFT, p1 = RIGHT, 25 fps.

## Layout

```
configs/     rule YAML
data/gold/   human labels
scripts/     canonical entry points — scripts/README.md
src/         metrics, pose_cnn, audio_io
results/     locked json/csv — do not overwrite TEST dirs
figures/     dissertation figures
reports/     audits and chapter drafts
tests/       invariants and TEST-lock checks
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

VideoMAE: `pip install -r requirements-video.txt`. Audio DEV: `pip install -r requirements-audio.txt`. GPU fine-tune used otter95 `/scratch` CUDA; do not `--force` locked out-dirs.

## Citation

Geng, S., et al. (2023). *RealTalk*. https://realtalk.cs.columbia.edu/
