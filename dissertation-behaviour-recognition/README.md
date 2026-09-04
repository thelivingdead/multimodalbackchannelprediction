# Predicting Backchannel Events from Multimodal Conversational Signals

MSc dissertation package, Divya Bisht, Centre for Vision, Speech and Signal Processing (CVSSP), University of Surrey, 2026.

The GitHub front page is the repository root [`README.md`](../README.md).

This package recognises two listener backchannels on 30 Columbia RealTalk gold clips: **head shake** (yaw) and **head nod** (pitch). The approved title says prediction. The executed task is recognition inside an observed window. An earlier study used one 60 s nod label per clip. The thesis results are the 3 s windowed protocol for both behaviours.

![Listener heads in labelled 3 s TEST windows](figures/paper/teaser_windowed_heads.png)

Two labelled 3 s TEST windows (official RealTalk listener boxes plus Euler). Top: **head shake**, `gold_023`, 15 to 18 s, yaw. Bottom: **head nod**, `gold_030`, 21 to 24 s, pitch. This is a face figure, not a pose-only chart. The TEST scores below are 15-clip balanced accuracies.

The protocol uses 3 s windows, a 2 s stride, 29 windows per clip, and 435 windows per split. DEV is `gold_001` to `gold_015`. TEST is `gold_016` to `gold_030`. The headline metric is balanced accuracy. Chance is 0.500. The 95% intervals are clip-level bootstrap (15 clips, 2000 resamples). An interval that includes 0.500 is not distinguished from chance.

Locked TEST, head shake: the yaw amplitude rule (axis y, τ = 4.091°) scores **0.654 [0.525, 0.794]**. That is the first result that clears chance on locked TEST.

Locked TEST, head nod: the return-ratio rule (amplitude plus return) scores **0.634 [0.576, 0.685]**. Amplitude-only nod TEST is 0.549 [0.480, 0.619] and includes chance.

DEV only, not TEST: shake Pose CNN 0.606 [0.519, 0.680]; nod Pose CNN 0.523 (interval includes chance). Identity-fixed nod VideoMAE, 1.5 s, last two blocks, no horizontal flip, scores 0.571. TEST was not scored for those CNN or VideoMAE runs. Largest-face Haar RGB crops are withdrawn because they showed the wrong person. Later RGB work uses identity-fixed crops only. A DEV nod fusion search did not beat the return-ratio rule. Fusion and the nod CNN/VideoMAE runs that stay at chance were not scored on TEST.

Pose and RGB are two encodings of the same camera.

Scripts: [`scripts/README.md`](scripts/README.md). Captions: [`figures/paper/CAPTIONS.md`](figures/paper/CAPTIONS.md).

## Layout

```
configs/                   rule YAML
data/gold/                 human clip labels
data/windowed_annotations/ 3 s window labels
scripts/                   canonical entry points, see scripts/README.md
src/                       metrics, pose_cnn, audio_io
results/                   locked json/csv; do not overwrite TEST dirs
figures/                   dissertation figures
reports/                   audits and chapter drafts
tests/                     invariants and TEST-lock checks
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

VideoMAE extras: `pip install -r requirements-video.txt`. Audio DEV extras: `pip install -r requirements-audio.txt`. Do not overwrite locked TEST directories.

## Citation

Geng, S., et al. (2023). *RealTalk*. https://realtalk.cs.columbia.edu/
