# Audio / RGB+audio on GOLD DEV only (otter)

**Predicting Backchannel Events from Multimodal Conversational Signals**.



These jobs score **GOLD DEV only**. They refuse `--score-test` and will not write locked
`results/videomae_finetuned/`, locked shake TEST json, or nod TEST tables.
Do not invent F1 if a step is blocked.

Audio is the RealTalk **container soundtrack** for the ~60 s watch window (both
participants). Pose and RGB stay **visual representation experiments** (two
encodings of the camera), not two sensory modalities. The task is **supervised
prediction of the backchannel label associated with a conversational window**,
not anticipatory forecasting from pre-event context. Text/transcript models
are future work. Nod TEST headline remains RGB fine-tune **F1 0.82**. Shake
TEST headline remains pose rule **F1 0.70**.

 `export REALTALK_VIDEO_DIR=/path/to/{video_id}.mp4` if members are cached. `/scratch/db01550/realtalk_videos` does not exist.

## Step A — alignment (must PASS before B/C)

```bash
cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
export OMP_NUM_THREADS=1
PY="$HOME/av_venv/bin/python"
$PY scripts/audio_alignment_check.py
```

Expect `results/audio_alignment_check.md` with **Verdict: PASS** and gitignored
WAVs under `data/audio_alignment_check/`. If FAIL (no audio, HTTP 403, disk),
**stop**. Do not run B or C.

## Step B — audio-only LR, TRAIN + GOLD DEV

```bash
cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
export OMP_NUM_THREADS=1
PY="$HOME/av_venv/bin/python"
$PY scripts/train_audio_baseline_dev.py
```

Writes `results/audio_dev_results.csv` (column `protocol=DEV_ONLY`).
Never scores TEST. ~80 TRAIN videos are fetched one at a time (tens of GB
transfer); existing `data/features/audio/*.npz` are reused.

## Step C — frozen RGB + audio concat, GOLD DEV

```bash
cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
export OMP_NUM_THREADS=1
PY="$HOME/av_venv/bin/python"
$PY scripts/train_av_fusion_dev.py
```

Writes `results/audio_visual_fusion_dev.csv` and
`results/tables/multimodal_ablation.md`. Compare RGB only / audio only /
RGB+audio on DEV. TEST is not scored.

Optional frozen wav2vec2/HuBERT: skip unless GPU/disk are clearly free after A–C.
Do not fine-tune those models. Do not block A–C on them.

## Refuse list

* `--score-test`
* `--split test`
* `--force` on `results/videomae_finetuned/` or locked shake TEST dirs
* committing `best_model.pt`, videos, `.npz` RGB/embeddings, WAVs,
  `scripts/nod_pipeline/`
