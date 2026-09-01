# Lab commands — 30-video nod plan

Run this on **otter48**, not on your Mac.

Do **not** download full RealTalk or `emoca.tar.gz`.

---

## What this is

You are **not** training a future-nod forecaster here.

You are doing the supervisor’s **first annotation / weakly-supervised nod detector** loop:

```
30 × 1-minute clips with gold labels
    ├─ 15 videos DEV   (tune / check)
    └─ 15 videos TEST  (final number only once)

+ extra unlabeled 1-minute clips = TRAIN

1. Rule-based pitch-cycle detector vs gold  →  BASELINE (precision, recall, F1)
2. Same detector on TRAIN                  →  noisy pseudo-labels
3. Train a classifier on those pseudo-labels
4. Pick hyperparameters using DEV gold only
5. Evaluate the classifier on TEST gold    →  SECOND RESULT
```

**80/10/10 is the wrong split for 30 gold videos.** That would leave ~3 test videos. Gold is small, so we keep **all 30 gold videos for DEV+TEST (15/15)** and use a **larger unlabeled TRAIN** set.

---

## Honest label status

`run_30video_plan.sh` uses **synthetic independent gold** (nods injected by the generator, not by the detector).

That proves the **code and split protocol**.

It is **not** a RealTalk dissertation number.

When you have 30 **human** 1-minute labels, put them in `outputs/nod_pipeline/gold_labels.csv` and re-run from step 09.

---

## On your Mac (push code only)

```bash
cd "/Users/divyabisht/Downloads/Msc Dissertation Divya"

git status
git add \
  scripts/nod_pipeline/event_metrics.py \
  scripts/nod_pipeline/08_build_30_video_experiment.py \
  scripts/nod_pipeline/09_analyse_splits.py \
  scripts/nod_pipeline/10_rule_baseline.py \
  scripts/nod_pipeline/11_pseudo_label.py \
  scripts/nod_pipeline/12_train_pseudo_classifier.py \
  scripts/nod_pipeline/13_compare_and_figures.py \
  scripts/nod_pipeline/run_30video_plan.sh \
  scripts/nod_pipeline/README.md \
  scripts/nod_pipeline/LAB_COMMANDS.md

git commit -m "Add 30-video gold/dev/test and pseudo-label training loop"

git push
```

Do **not** add `data/`, `outputs/`, `.venv/`, videos, or pickles.

---

## On the lab (otter48)

Copy **one block at a time**.

### 1. Enter the project

```bash
cd ~/multimodalbackchannelprediction
source .venv/bin/activate
git pull
df -h .
```

If `git pull` says the repo is missing, clone first:

```bash
cd ~
git clone https://github.com/thelivingdead/multimodalbackchannelprediction.git
cd multimodalbackchannelprediction
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r scripts/nod_pipeline/requirements.txt
```

If `joblib` is missing:

```bash
pip install joblib
```

### 2. Run the 30-video plan

```bash
cd ~/multimodalbackchannelprediction
source .venv/bin/activate
mkdir -p outputs/nod_pipeline
chmod +x scripts/nod_pipeline/run_30video_plan.sh
bash scripts/nod_pipeline/run_30video_plan.sh
```

Pose extraction of 80 clips can take **10–20 minutes**. Let it finish.

### 3. Read the results

```bash
echo "===== TIME ====="
cat outputs/nod_pipeline/time_log.txt

echo "===== SPLITS ====="
python - <<'PY'
import json
s=json.load(open("outputs/nod_pipeline/splits.json"))
print("train videos", s["n_train_videos"], "hours", round(s["train_hours"],3))
print("dev videos  ", s["n_dev_videos"], "hours", round(s["dev_hours"],3))
print("test videos ", s["n_test_videos"], "hours", round(s["test_hours"],3))
print("dev gold nods", s["dev_gold_nods"], "test gold nods", s["test_gold_nods"])
PY

echo "===== RULE BASELINE ====="
cat outputs/nod_pipeline/rule_baseline_metrics.csv

echo "===== HYPERPARAM SEARCH (DEV only) ====="
cat outputs/nod_pipeline/hyperparam_search.csv

echo "===== LEARNED MODEL ON TEST ====="
cat outputs/nod_pipeline/learned_test_metrics.json

echo "===== COMPARISON ====="
cat outputs/nod_pipeline/baseline_vs_learned.csv

echo "===== DISK ====="
du -sh data/nod30 outputs/nod_pipeline
df -h .
```

---

## What each number means

| File | Meaning |
| --- | --- |
| `rule_baseline_metrics.csv` | **Baseline.** Rule detector vs gold on DEV and TEST. Event F1, IoU ≥ 0.2. |
| `hyperparam_search.csv` | LogReg / RF / MLP tried; winner chosen on **DEV**, never TEST. |
| `learned_test_metrics.json` | **Second result.** Classifier trained on TRAIN pseudo-labels, scored on TEST gold. |
| `baseline_vs_learned.csv` | Side-by-side TEST F1. |
| `time_log.txt` | Seconds spent in each stage. |

---

## If this works, next (real data)

1. Put 30 real 1-minute clips + your manual `gold_labels.csv` on the lab.
2. Keep the same 15/15 DEV/TEST video split.
3. Add more unlabeled 1-minute clips as TRAIN (stay under ~25 GB; keep ≥5 GB free).
4. Re-run from pose extract → rule baseline → pseudo-label → train → test.

Do not call the synthetic F1 a RealTalk result.
