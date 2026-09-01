# Lab GPU / VS Code Remote — visualisation quickstart

## Goal

Compare **video frames** with **reported FLAME/EMOCA values** and mark nod/shake detections. Use this for:

- axis-mapping sanity checks (pitch vs true vertical nod)
- fps / frame-index alignment checks
- qualitative QA before trusting auto-labels
- figures for your two-week progress / Data chapter

GPU is **not required** for this script (CPU is enough). Run it on the remote lab machine so you have dataset + storage access.

## Setup (VS Code Remote-SSH → lab)

```bash
cd "/path/to/Msc Dissertation Divya"
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
```

## Demo (no dataset)

```bash
python scripts/visualise_flame_vs_frames.py --demo --out outputs/viz_demo
```

Open:

- `outputs/viz_demo/timeseries_events.png`
- `outputs/viz_demo/contact_sheet_overlay.png`
- `outputs/viz_demo/event_*_strip.png`
- `outputs/viz_demo/detected_events.csv`

## Real RealTalk clip

```bash
python scripts/visualise_flame_vs_frames.py \
  --video /data/realtalk/some_clip.mp4 \
  --flame /data/realtalk/some_clip_flame.npz \
  --fps 25 \
  --min-height-deg 1.0 \
  --every 15 \
  --out outputs/viz_clip01
```

`--min-height-deg 1.0` matches your PDF note (subtle shake below **1°** was missed).

## What to report from one run

1. Number of detected nod/shake events (`summary.json`)
2. Screenshot of one `event_*_nod_strip.png` showing frames + pitch peak
3. Statement: detections look real / false; any axis swap needed
4. Next: annotate continuous spans in VIA for recall measurement

## Hugging Face RealTalk pointer

Dataset card often used: `scottgeng00/realtalk` on Hugging Face. Adapt `--flame` key mapping in `load_flame()` if your NPZ field names differ (aliases already cover common EMOCA/FLAME names).
