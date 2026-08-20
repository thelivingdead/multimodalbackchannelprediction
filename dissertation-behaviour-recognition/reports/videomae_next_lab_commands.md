# VideoMAE — next lab commands (otter48): shard probe and range index ONLY

**Date:** 20 August 2026.
**Machine:** `db01550@otter48`; lab repo root `~/multimodalbackchannelprediction` (`/user/HS400/db01550/multimodalbackchannelprediction`).
**Status:** VideoMAE has **not** been run. No VideoMAE F1, accuracy, or confusion matrix exists anywhere, and none may be written into the dissertation. **Step 1 (§3) has PASSED on otter48.** Exactly one read-only lab step remains authorised — the range index in §4 — and everything else stays forbidden until its output is pasted back. Steps 3–6 (§7–§10) are now written into the repo but are **not authorised**: each is gated on the previous step's pasted-back output, in order.

**Why there is no RGB run yet (locked facts):**

- **No RealTalk RGB exists on otter** (or on the Mac). Video exists only as members of multi-GB `videos/videos_*.tar` shards at `huggingface.co/datasets/scottgeng00/realtalk`.
- otter free disk: **~7.3 GB** of a **~25 GB** home quota. Runtime is **torch 2.13.0+cpu** in `~/multimodalbackchannelprediction/.venv`.
- The lab GPU (RTX A4000) is **not usable for this project**: a CUDA PyTorch stack (~5–8 GB installed) does not fit the quota, so even with the GPU present, execution stays on the existing CPU build.
- Consequence: **frozen CPU VideoMAE embeddings with a tiny trained head are the only feasible variant.** Fine-tuning is a NO-GO on this quota. And **no training of any kind starts until the RGB mapping in §4 is confirmed to exist.**

---

## 1. What is authorised

| Step | What | Writes to otter disk? |
| --- | --- | --- |
| 0 | Read-only local checks (`df`, `ls`, torch version) | No |
| 1 | 64 KB HTTP Range probe of `videos_00.tar` — **PASSED** (206, `accept-ranges: bytes`, shard 00 size 21852938240) | No (response headers only) |
| 2 | Range-walked shard index: `video_id → (shard, offset, size)` — script ready (`scripts/build_video_shard_index.py`) | One JSON, KB-scale |

Everything else — downloading or saving any shard, `hf_hub_download` on shards, decoding video, installing CUDA torch, extracting embeddings, and any training — is **not authorised** and must not be started.

## 2. Step 0 — read-only local checks

```bash
# on otter48 — read-only
df -h ~                                                            # expect ~7.3 GB free; STOP if < 4.5 GB
cd ~/multimodalbackchannelprediction
ls dissertation-behaviour-recognition/features/gold   | wc -l      # expect 30
ls dissertation-behaviour-recognition/features/pseudo | wc -l      # expect 80
source .venv/bin/activate
python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"   # expect 2.13.0+cpu, cuda: False
```

The `ls features/pseudo | wc -l` count (80) is the size of the pseudo pool the frozen pose rule labelled (70 nod / 10 unclear). RGB would ever be fetched only for these same 80 ids plus the 30 gold ids, and only if every later gate passes.

## 3. Step 1 — 64 KB Range probe of `videos_00.tar` — **PASSED**

**Measured on otter48, 20 August 2026 — GO to Step 2:**

- Shard listing returned **14 shards**: `videos/videos_00.tar` … `videos/videos_13.tar`.
- Range probe of `videos_00.tar`: **HTTP 206 Partial Content** with `content-range: bytes 0-65535/21852938240` and `accept-ranges: bytes`. Shard 00 is **21 852 938 240 bytes (~21.9 GB)** and the server honours byte ranges.

The commands and decision table below are kept as the record of what was run; the first table row (206) matched.

First list the exact shard names (API metadata only; nothing is downloaded):

```bash
curl -sS "https://huggingface.co/api/datasets/scottgeng00/realtalk" \
  | python -c "import sys,json; [print(s['rfilename']) for s in json.load(sys.stdin)['siblings'] if s['rfilename'].startswith('videos/')]"
```

Then probe the first shard with a 64 KB range request (headers only; `-L` follows the redirect to the CDN):

```bash
curl -sSIL -H "Range: bytes=0-65535" \
  "https://huggingface.co/datasets/scottgeng00/realtalk/resolve/main/videos/videos_00.tar" \
  | egrep -i "^HTTP/|content-range|content-length|accept-ranges"
```

**Decision (abort criteria):**

| Probe answer | Meaning | Action |
| --- | --- | --- |
| **206 Partial Content** with `content-range: bytes 0-65535/<shard size>` | Server honours byte ranges | **GO to Step 2** |
| **200 OK** (no `content-range`) | Range ignored — any "probe" would stream the whole multi-GB shard | **STOP.** Range walking is impossible on this host without full-shard downloads, which the quota forbids. Paste the headers back; do not retry. |
| **403 Forbidden** | Gated asset / missing authorisation | **STOP.** Paste the headers back; do not attempt token workarounds from this account. |

If the HEAD answer is ambiguous, confirm **once** with a body-discarding GET that cannot fill the disk (`--max-time` bounds the damage if the server answers 200 and starts streaming the shard):

```bash
curl -sS -D - -o /dev/null --max-time 30 -r 0-65535 \
  "https://huggingface.co/datasets/scottgeng00/realtalk/resolve/main/videos/videos_00.tar" \
  | egrep -i "^HTTP/|content-range|content-length"
```

Same decision table.

## 4. Step 2 — range-walked shard index (Step 1 passed; this is GO)

Goal: a JSON map `video_id → {shard, offset, size}` covering the 30 gold ids plus the 80 pseudo ids (110 wanted ids), built by walking tar member headers with 64 KB range reads. **Never** save a shard, a member video, or a decoded frame.

The script is written and in the repo: `dissertation-behaviour-recognition/scripts/build_video_shard_index.py`. It has **not been run anywhere yet** — its first run is this step. After pushing the Mac repo to GitHub, on otter48:

```bash
cd ~/multimodalbackchannelprediction
git fetch origin && git reset --hard origin/main   # pick up the new script
cd dissertation-behaviour-recognition
source ../.venv/bin/activate
python scripts/build_video_shard_index.py
```

What the script does (network + one small JSON; nothing else is written):

- Wanted ids = the 30 gold `video_id`s in `data/gold_annotations.csv` + the 80 `video_id`s embedded in `features/pseudo/*.npz`; it prints the loaded count (expect 110 unique).
- For each of the 14 shards (`videos_00.tar` … `videos_13.tar`) it reads 64 KB windows with `Range` headers and parses the 512-byte tar member headers in memory, keeping only name, offset, and size. Member data is never transferred: the walk jumps straight from each header to the next, so one member costs ~one 64 KB read. `offset`/`size` in the JSON are the member's data bytes inside the shard, so a later step can fetch one video with a single Range GET.
- It stops the instant all wanted ids are found (the current shard is exited and remaining shards are never touched); otherwise a shard is walked to its end-of-archive marker — which wanted ids a shard holds is unknowable until it has been walked, so that is the only safe per-shard stop. Per-shard hits are printed at the end.
- It checks free space on `~` at every shard boundary and **aborts below 5.4 GB** (free space should not move at all — this step is network plus one small JSON).
- It **aborts immediately**, printing the status code and its meaning, if any range read returns non-206.
- It prints a coverage summary (found n/110, per-shard hits, any missing ids) and writes exactly one artefact: `results/video_shard_index.json` (KB-scale).

Paste the coverage output and the JSON (or the abort message) back before anything further is discussed.

## 5. Disk projection — why this stays inside quota

| Item (only if later gates pass) | Projected size |
| --- | --- |
| Steps 1–2 (probe + index JSON) | ~0 (KB) |
| Frozen VideoMAE-Base checkpoint (`MCG-NJU/videomae-base`) | ~0.4 GB |
| `transformers` + `safetensors` + `opencv-python-headless` on top of existing CPU torch | ~0.3–0.5 GB |
| 110 × 16-frame 224×224 face crops (uint8 npz, or re-encoded crops) | ~0.27 GB max |
| Pooled 768-D embeddings (110 clips) | < 1 MB |
| **Projected peak new usage** | **~1.0–1.5 GB** |

Against ~7.3 GB free, **free space never drops below ~5.4 GB** at the projected peak — comfortably above the pipeline's hard floor (`MIN_FREE_GB = 3.0` in `run_full_experiment.py`). If `df -h ~` ever shows less than 5.4 GB during any stage, stop and remove the partial artefact.

**Explicitly infeasible on this quota:** a CUDA PyTorch stack (~5–8 GB) for the RTX A4000; VideoMAE fine-tuning (optimizer states, gradients, checkpoints — GB-scale); saving any `videos_*.tar` shard (multi-GB each); decoding full-length clips to disk (tens of GB). The **only** feasible variant is frozen CPU VideoMAE embeddings with a tiny head — and **no training runs until the §4 RGB mapping is confirmed to exist.**

## 6. What VideoMAE is for, and how any score would be reported

The purpose of the planned VideoMAE experiment is to test whether weak supervision from pose can train an RGB representation: the frozen rule's pseudo-labels (70 nod / 10 unclear) would supervise a small head on frozen VideoMAE embeddings of the same watch windows, answering whether the nod signal survives the move from EMOCA pose coefficients to raw pixels. Any VideoMAE F1 will be reported only if actually measured on the same held-out 15-window TEST set under the existing protocol (DEV for tuning, TEST scored once). Until such a measurement exists, VideoMAE appears in the dissertation only as planned future work, and no VideoMAE number of any kind is reported.

---

## 7. Step 3 — fetch 16-frame RGB face-crop windows (gated on §4 output)

**Gate:** §4 pasted back with all 110 wanted ids found in `results/video_shard_index.json`. If the index is incomplete, stop here — do not fetch.

Script: `scripts/fetch_rgb_windows.py`. For each wanted clip it issues **one** Range GET of exactly that clip's member bytes (`offset`…`offset+size-1` from the index), pipes the bytes through the ffmpeg binary bundled with `imageio-ffmpeg` **in memory** (no shard, no full video, no JPG ever touches disk), decodes the 16 frames uniformly spaced across the clip's window, face-crops to 224×224, and writes `features/rgb16/<sample_id>.npz` (~2.4 MB uint8 each, ~0.27 GB for 110). Free space on `~` is checked before and after every clip and the run aborts below **5.4 GB**; any non-206 range answer aborts. Reruns skip completed clips.

**Face-crop note (measured fact, not assumed):** the committed EMOCA-derived npz files contain only `frames, rotation_xyz, expression, valid_ratio, video_id, person, sample_id` — **no camera/scale/bbox keys** — and the raw EMOCA pickles are not on disk, so a face box cannot be recovered from EMOCA outputs. The script therefore uses the Haar frontal-face cascade shipped inside `opencv-python-headless` on the middle frame (largest detection, box expanded ×1.6 and squared), falling back to a centred square crop (`crop_mode` is recorded per clip). RealTalk shows two people; when several faces are found the largest is used and the clip is flagged `multi_face_clips` in the summary — the p0/p1 label is **not** mapped to image position, and this is a recorded limitation.

```bash
cd ~/multimodalbackchannelprediction
git fetch origin && git reset --hard origin/main   # pick up the new scripts
cd dissertation-behaviour-recognition
source ../.venv/bin/activate
df -h ~                                             # STOP if < 5.4 GB free
pip install opencv-python-headless imageio-ffmpeg   # ~0.2 GB, inside §5 budget
df -h ~                                             # re-check after install
python scripts/fetch_rgb_windows.py --ids gold_001,gold_016   # SMOKE: 2 clips
```

**Smoke gate:** `results/rgb16_fetch_summary.json` must show both clips `ok` (no `short_decode`, no `failed`), `free_gb_end` ≥ 5.4, and the two npz files exist. Paste the summary back. Only then:

```bash
python scripts/fetch_rgb_windows.py                 # FULL: all wanted clips
```

**Full-run gate:** summary shows `n_ok_total` covering the wanted set (110 if the index was complete), failures listed explicitly (failed clips are excluded downstream, never padded), free ≥ 5.4 GB. Paste the summary JSON back.

## 8. Step 4 — frozen VideoMAE embeddings (gated on §7 output)

**Gate:** §7 full-run summary pasted back; all clips needed for TRAIN/DEV/TEST have `features/rgb16/*.npz`.

Script: `scripts/extract_videomae_embeddings.py`. It first issues HTTP HEAD requests to **measure** checkpoint sizes and picks `MCG-NJU/videomae-base` (~0.4 GB, 768-D) if it fits the budget (`free − 5.4 GB`), else `MCG-NJU/videomae-small` (~0.2 GB, 384-D), else exits **BLOCKED** downloading nothing. The encoder is frozen (`eval` + `no_grad`); each clip is mean-pooled over patch tokens to one embedding and saved to `data/features/videomae/<sample_id>.npz` (<1 MB total for 110). Checkpoint name, weight bytes, and `transformers`/`torch` versions are recorded in the commitable `results/videomae_embeddings_meta.json`. HF caches are pinned to the gitignored `.hf_cache/` inside the repo.

```bash
cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
source ../.venv/bin/activate
df -h ~                                             # STOP if < 5.4 GB free
pip install "transformers" "safetensors"            # record: pip show transformers
python scripts/extract_videomae_embeddings.py --limit 2   # SMOKE (downloads checkpoint)
df -h ~                                             # re-check after download
python scripts/extract_videomae_embeddings.py             # FULL
```

**Gate:** `results/videomae_embeddings_meta.json` exists, records the checkpoint actually used, and `n_embeddings_total` equals the rgb16 coverage. If the script prints BLOCKED (checkpoint exceeds budget / unreachable), **stop and paste it back** — no smaller-but-different model may be substituted ad hoc. Paste the meta JSON back.

## 9. Step 5 — split-leakage gate + head training (gated on §8 output)

**Gate:** §8 meta JSON pasted back with complete embeddings.

First the leakage gate — if it prints FAIL, training must not start:

```bash
cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
source ../.venv/bin/activate
python scripts/check_split_leakage.py               # must print PASS
```

It asserts: gold DEV/TEST disjoint by `sample_id` **and** `video_id`; no pseudo id collides with gold; no pseudo video in gold TEST (or DEV). Then the head run (protocol identical in shape to the pose CNN: TRAIN = pseudo clips with embeddings, labels from the frozen rule's `results/pseudo_labels.csv`; DEV = 15 gold DEV for early stopping **and** probability threshold; TEST = 15 gold TEST scored **exactly once**; `BCEWithLogitsLoss` with `pos_weight`; seed 42):

```bash
OMP_NUM_THREADS=1 python scripts/train_videomae_head.py
```

Writes `results/videomae_frozen_head/{metrics.json, predictions.csv, training_history.csv}` (commitable) and `models/videomae_head.pt` (gitignored). The script refuses to rerun while `metrics.json` exists (`--force` required, and any forced rerun must be recorded in `reports/dissertation_evidence/experiment_log.md`) — TEST cannot be silently re-scored.

**Gate:** `metrics.json` exists with real measured values. Paste `metrics.json` back.

## 10. Step 6 — bootstrap CIs + main results table (gated on §9 output)

**Gate:** §9 `metrics.json` pasted back.

```bash
cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
source ../.venv/bin/activate
python scripts/bootstrap_f1.py                      # 1000 resamples, seed 42
python scripts/make_main_results.py
```

`bootstrap_f1.py` computes 95% F1 CIs purely from saved TEST predictions CSVs (rule, pose CNN, and — now — VideoMAE head) into `results/tables/bootstrap_ci.csv`. `make_main_results.py` writes `results/tables/main_results.csv` and `main_results.md` with rows Rule baseline / Pose CNN raw / Pose CNN xyz_deriv / Frozen VideoMAE head and columns `model, input, supervision, train_n, precision, recall, f1, accuracy, f1_ci_lo, f1_ci_hi` — real values only, N/A wherever an artefact was never produced (e.g. the pose-CNN raw ablation saved no predictions CSV, so its CI is N/A by construction).

**Final gate:** paste `results/tables/bootstrap_ci.csv`, `results/tables/main_results.csv` (or `.md`) back. Only then may any VideoMAE number enter the dissertation, quoted exactly as measured.

---

*The §3 probe results were measured by the lab user on otter48 and pasted back. The agent wrote `scripts/build_video_shard_index.py` and the Step 3–6 scripts (`fetch_rgb_windows.py`, `extract_videomae_embeddings.py`, `train_videomae_head.py`, `check_split_leakage.py`, `bootstrap_f1.py`, `make_main_results.py`) but has not run any of them on otter48 — §4 runs first. Derived from the locked facts in `reports/videomae_preflight_lab.md` and `reports/SCOPE_MAP.md`.*
