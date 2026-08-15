# Lab disk: do **not** wipe the machine

**No.** Do not delete your home directory, the git repo, or start the whole lab account from scratch.

You only need to free quota junk and then `git pull` this package. Quota is ~24 GB. Keep at least **5 GB** free.

Run this on **otter48**, not on your Mac.

---

## 1. See what is using space (safe)

```bash
df -h ~
du -sh ~/* ~/.[^.]* 2>/dev/null | sort -h | tail -30
du -sh ~/multimodalbackchannelprediction/* 2>/dev/null | sort -h
```

Or from this folder:

```bash
bash scripts/lab_disk_report.sh
```

---

## 2. Keep

| Keep | Why |
| --- | --- |
| `~/multimodalbackchannelprediction/.git` | your code |
| any **real** `clip.mp4` / `emoca.pkl` you already copied | expensive to get again |
| `data/gold/events.csv` and `annotation_log.csv` if you already labelled | human gold |
| a working `.venv` **on disk** (not committed) | faster than reinstalling |

---

## 3. Delete only if the file exists and you confirm the path

These are the usual quota killers. Check with `ls` first. Do **not** run `rm -rf ~`.

```bash
cd ~/multimodalbackchannelprediction

# Accidental full EMOCA archive (~23 GB) — never keep this
ls -lh emoca.tar.gz 2>/dev/null
rm -f emoca.tar.gz

# Hugging Face hub cache of full shards
du -sh ~/.cache/huggingface 2>/dev/null
# only if that cache is huge and you are not mid-download:
# rm -rf ~/.cache/huggingface/hub

# Torch / pip caches
du -sh ~/.cache/pip ~/.cache/torch 2>/dev/null
# rm -rf ~/.cache/pip

# Old synthetic demo outputs from scripts/nod_pipeline (not gold)
du -sh data/tiny_subset outputs 2>/dev/null
# rm -rf data/tiny_subset outputs

# A .venv that was accidentally committed: remove it from disk after git pull,
# then make a fresh venv. Do not git-commit .venv again.
```

If `git status` still shows thousands of `.venv` files, the repo history may already contain them. Do **not** rewrite git history on the lab. Just:

```bash
git pull
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r dissertation-behaviour-recognition/requirements.txt
```

---

## 4. Then run the real 1-hour path

```bash
cd ~/multimodalbackchannelprediction
source .venv/bin/activate
git pull
cd dissertation-behaviour-recognition
bash scripts/run_hour_pilot.sh
```

If Stage A writes **synthetic** clips, that is only for a pipeline check. Do not put that F1 in the dissertation as a RealTalk result.

---

## 5. What “start from scratch” actually means

| Action | Do it? |
| --- | --- |
| Wipe home directory / reinstall the lab account | **No** |
| Delete the git repo and clone again | Only if the repo is corrupted; usually `git pull` is enough |
| Delete synthetic demo clips and rerun the hour script | Yes, if you have no real clips yet |
| Delete human `events.csv` | **No** |
| Download `emoca.tar.gz` or full RealTalk | **No** |
