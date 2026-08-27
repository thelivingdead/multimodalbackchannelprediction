# Audio / RGB+audio on GOLD DEV only (otter)

Approved dissertation title (unchanged): **Predicting Backchannel Events from Multimodal Conversational Signals**.

**27 Aug 2026 Mac (Cursor agent):** Step A **FAIL** (need ≥3 clips; only `gold_012` extracted). Hugging Face Range reads **work** outside the Cursor sandbox proxy (HTTP 206; `6RDkdbgzeAI` AVI 25 fps + stereo MP3 48 kHz). WAV: `data/audio_alignment_check/gold_012_6RDkdbgzeAI.wav` (60.000 s, 2880000 samples, RMS 0.03762281). Mixed **conversation** audio, not speaker audio. 20-minute stop: clips 2–5 not fetched; **do not train**. Otter SSH works; `/scratch/db01550/venv` and `~/multimodalbackchannelprediction/.venv` are **missing**; frozen VideoMAE npz (110 files, 3.1 MB) exist on otter but were not used. **Do not invent F1.** GOLD TEST was not scored. Full audit: `reports/audio_alignment_audit.md`.

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

Videos are **not** in git. On otter, either:

* set `REALTALK_VIDEO_DIR` to a folder of `{video_id}.mp4`, or
* let the scripts HTTP Range-read members using `results/video_shard_index.json`
  (same mechanism as `scripts/fetch_rgb_windows.py`). Members are deleted after
  the window WAV / features are written.

Frozen VideoMAE embeddings (Step C) are expected at:

`dissertation-behaviour-recognition/data/features/videomae/<sample_id>.npz`

Do **not** retrain VideoMAE. If that folder is missing, skip C and record the blocker.

## Install (once)

```bash
# otter48 (27 Aug 2026): /scratch/db01550/venv and ~/multimodalbackchannelprediction/.venv
# do not exist. System python is /usr/bin/python3. /scratch has ~165G free but no db01550 dir.
ssh otterdiv
cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
export OMP_NUM_THREADS=1
python3 -m venv "$HOME/av_venv"
source "$HOME/av_venv/bin/activate"
python -m pip install librosa imageio-ffmpeg requests numpy scipy scikit-learn
```

Optional: `export REALTALK_VIDEO_DIR=/path/to/{video_id}.mp4` if members are cached. `/scratch/db01550/realtalk_videos` does not exist.

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
