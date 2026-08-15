# Tiny-subset nod-detection pipeline (25 GB safe)

Prove the **entire** nod pipeline on **10 × 1-minute** clips. Do **not** download `emoca.tar.gz` (~23.6 GB).

```
10 small clips → EMOCA pose → pitch plots → candidate nods
  → manual (or demo) ground truth → baseline F1 → visual demo
```

## Install (lab / VS Code Remote)

```bash
cd "/path/to/Msc Dissertation Divya"
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r scripts/nod_pipeline/requirements.txt
```

## Prove it with demo data (no RealTalk download)

```bash
cd "/path/to/Msc Dissertation Divya"
python scripts/nod_pipeline/01_make_tiny_subset.py --mode demo
python scripts/nod_pipeline/02_extract_pose.py
python scripts/nod_pipeline/03_detect_nod_candidates.py
python scripts/nod_pipeline/04_label_ground_truth.py --demo-fill --validate
python scripts/nod_pipeline/05_baseline_experiment.py --model all
python scripts/nod_pipeline/06_visual_demo.py
```

Or: `bash scripts/nod_pipeline/run_all_demo.sh`

**First technical figure:** `data/tiny_subset/demo_00/pitch_p0.png`  
**F1 / metrics:** `outputs/nod_pipeline/metrics.json`  
**Demo plot:** `outputs/nod_pipeline/demo_demo_00_p0.png`

## Real 10 × 1-minute clips (storage-safe)

### A. Files already on the lab (recommended)

Put ~10 `.avi`/`.mp4` and matching `<id>.pkl` EMOCA files in two folders, then:

```bash
python scripts/nod_pipeline/01_make_tiny_subset.py --mode from-local \
  --video-dir /data/realtalk_sample/videos \
  --emoca-dir /data/realtalk_sample/emoca \
  --n 10 --duration 60 --start 30
```

This **trims** each video to 60 s and **slices** the pickle to those frames only.

### B. Hugging Face (will refuse the full EMOCA tar)

```bash
python scripts/nod_pipeline/01_make_tiny_subset.py --mode hf --n 10
```

If the hub only lists `emoca.tar.gz` as one blob, the script **will not download it**. Use `--demo` or copy 10 pkls by hand.

Never:

```bash
# DON'T
huggingface-cli download scottgeng00/realtalk emoca.tar.gz
```

## Manual ground truth (after real candidates)

1. Open `outputs/nod_pipeline/review.html` and/or each 1-minute `clip.mp4`.
2. Edit `outputs/nod_pipeline/labels.csv`:

```
video_id,person,start_time,end_time,label
5hxY5Svr2aM,p0,12.40,13.10,nod
```

Labels: `nod` / `non-nod`  
Or 3-class: `nod` / `other-head-motion` / `neutral` (`--classes 3`)

3. `python scripts/nod_pipeline/04_label_ground_truth.py --validate`

Pose thresholds are **not** ground truth. Your eyes are.

## Pitch axis

EMOCA pose[0:3] is axis-angle. We map Euler **rx → pitch** (nod). If plots look like shakes, re-run:

```bash
python scripts/nod_pipeline/02_extract_pose.py --pitch-axis 1
```

## Outputs

| Path | What |
| --- | --- |
| `data/tiny_subset/<id>/pose.csv` | frame, timestamp, pitch, yaw, roll |
| `data/tiny_subset/<id>/pitch_p*.png` | pitch vs time |
| `outputs/nod_pipeline/candidates.csv` | possible nod intervals |
| `outputs/nod_pipeline/labels.csv` | validated GT |
| `outputs/nod_pipeline/metrics.json` | precision, recall, F1 |
| `outputs/nod_pipeline/demo_*.png` | pitch + GT + predictions |
