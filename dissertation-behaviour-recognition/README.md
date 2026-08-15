# Weakly supervised nod recognition — 1-hour lab run

Storage cap: **24 GB**. Do not download RealTalk or `emoca.tar.gz`.

Annotation is only two numbers:

- **1** = clear nod
- **0** = unclear

## What you will have after one hour

Human labels (class 1 nods) → EMOCA pitch → rule detector tuned on PILOT/DEV → precision / recall / F1 → diagnostic plots.

VideoMAE is **disabled** until that file exists: `results/pilot_nod_rule_metrics.json`.

All current clips are **PILOT/DEV**. TEST stays empty until you add 15 new untouched videos.

---

## On your Mac (push)

```bash
cd "/Users/divyabisht/Downloads/Msc Dissertation Divya"

git add dissertation-behaviour-recognition
git status
# confirm no .mp4 .pkl .venv data/working

git commit -m "Add behaviour-recognition package with 1/0 nod annotation and rule pilot"

git push
```

---

## On the lab (one hour)

### 0. Setup

```bash
cd ~/multimodalbackchannelprediction
source .venv/bin/activate
git pull

cd dissertation-behaviour-recognition
pip install -r requirements.txt
chmod +x scripts/run_hour_pilot.sh scripts/run_hour_pilot_stage_b.sh scripts/lab_disk_report.sh
```

If this folder is missing after `git pull`, you did not push it from the Mac.

### 1. Stage A (~10–15 min, no annotation)

```bash
bash scripts/run_hour_pilot.sh
```

This checks disk, writes 10 × 1-min **synthetic** pilot clips (only if you have no RealTalk files), extracts pitch, inspects EMOCA keys.

If you **already have** real 1-min clips + matching `emoca.pkl` on the lab, copy them instead:

```text
data/working/pilot/<video_id>/clip.mp4
data/working/pilot/<video_id>/emoca.pkl
data/working/pilot/<video_id>/meta.json
```

`meta.json` example:

```json
{"video_id": "5hxY5Svr2aM", "fps": 25, "duration_s": 60, "listener": "p0", "role": "pilot"}
```

Then skip `make_pilot_clips.py` and run extract/inspect yourself.

### 2. YOU annotate (~20–40 min)

```bash
python scripts/annotate_candidates.py
```

For each proposed interval type:

```text
1   clear nod
0   unclear
a   add a missed nod (then type start_s and end_s)
q   quit and save
```

Only **1** counts as a gold positive. **0** is not a nod.

Times are stored in `data/gold/annotation_log.csv`.

### 3. Stage B (~10 min)

```bash
bash scripts/run_hour_pilot_stage_b.sh
```

Read:

```text
results/pilot_nod_rule_metrics.json
reports/pilot_nod_findings.md
figures/pilot_nod/
figures/rule_baseline/
figures/annotations/
```

### 4. Publication figures (anytime)

```bash
python scripts/make_figures.py --all
```

Does **not** invent scores. Missing result files are skipped and printed.
Schematics (`pipeline_overview`, `github_overview`) always generate because they contain no metrics.
Existing files are not overwritten unless you pass `--force`.

After Stage B you should have dissertation-ready PNG+JPG (300 DPI) for:

- gold class counts and timelines
- pitch/yaw/roll with gold intervals
- DEV grid heatmap
- event P/R/F1, IoU 0.10 / 0.30 / 0.50
- per-video F1
- pitch in gold-nod frames vs background
- TP / FP / FN pitch windows

---

## Lab disk — do not wipe the computer

**Do not** clear your whole lab home directory and start from scratch.

Keep the git repo, any real clips/pkls, and any human labels. Delete only quota junk (`emoca.tar.gz`, huge Hugging Face caches, accidental `.venv` in git). Commands: [LAB_CLEANUP.md](LAB_CLEANUP.md).

```bash
bash scripts/lab_disk_report.sh
```

---

## After today (not this hour)

15 DEV (these videos) + 15 new TEST videos → freeze the nod rule on DEV → score TEST once → then pseudo-labels → pose classifier → VideoMAE.

Do not train on TEST. Do not call synthetic clips a RealTalk result if you used `make_pilot_clips.py`.
