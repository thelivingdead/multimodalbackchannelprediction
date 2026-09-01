# VideoMAE preflight audit (read-only)

**Date:** 2026-08-20 · **Machine:** this Mac (x86_64, darwin 24.6.0) · **Scope:** audit only — nothing downloaded, installed, trained, or committed. All numbers below were measured on this machine or read from repo files; the only exceptions are externally published facts (RealTalk dataset card, VideoMAE architecture sizes), which are marked **[external, verify]**.

---

## The 11 audit items

### 1. Current pipeline (as it actually ran)

`scripts/run_full_experiment.py` (seed 42, 15 epochs max, `results/experiment_config.json`):

1. Stream `emoca.tar.gz` (~25.3 GB, Hugging Face `scottgeng00/realtalk`) **without saving it**; per `results/emoca_stream_status.json`, 81 archive members were seen.
2. For each of 30 gold windows and 80 unlabeled videos, extract per-frame **EMOCA/FLAME Euler rotation (x,y,z, degrees)** + **20-dim expression** at 25 fps into compressed npz (`features/gold/*.npz`, `features/pseudo/*.npz`).
3. **Rule baseline:** Savitzky–Golay (11,2) smoothing, largest sign-change amplitude within 5–50 frames on one rotation axis; axis (**x**) and threshold (**16.3538°**) frozen on DEV only (`results/rule_selected_config.json`). TEST F1 **0.6667** (`results/rule_test_metrics.json`).
4. **Pseudo-labels:** frozen rule scores the 80 unlabeled clips → **70 nod / 10 unclear** (`results/pseudo_labels.csv`).
5. **1D CNN** (Conv1d 32→64→64 + adaptive avg pool + linear head; BCE with pos_weight; Adam 1e-3; sequences resampled to length 128; normalisation fit on pseudo-train only; epoch and probability threshold chosen on DEV; early stop after 4 stagnant epochs). Ran 13 epochs; best DEV F1 0.8888 at epoch 9–10 (`results/training_history.csv`). TEST F1 **0.70** (`results/classifier_test_metrics.json`).
6. **Ablations A–D** written to `results/ablation_results.csv` (see E2 section).
7. VideoMAE: **not implemented; planned next experiment** — `scripts/15_train_videomae.py` and `scripts/16_evaluate_videomae.py` are documented planned-experiment notes (no implementation, nothing downloaded); `figures/videomae/` contains only an empty `.gitkeep`.

A separate, older **synthetic** pipeline lives at the workspace root (`scripts/nod_pipeline/`, `data/nod30/`, `outputs/nod_pipeline/`): 50 synthetic train + 30 synthetic gold clips with `pose.csv` per clip. It is **not** a RealTalk result path and is not the input to the numbers above.

### 2. Exact paths

Workspace root: `/Users/divyabisht/Downloads/Msc Dissertation Divya` (git repo root)
Project: `/Users/divyabisht/Downloads/Msc Dissertation Divya/dissertation-behaviour-recognition`

| What | Path | State |
| --- | --- | --- |
| Gold annotations (30 rows) | `data/gold_annotations.csv` | present |
| Gold support files | `data/gold/{annotation_sheet,annotation_log,events,watch_list}.csv` | present |
| Split lists | `data/splits/{gold_split.csv,gold_dev.txt,gold_test.txt,planned_dev.txt,planned_test.txt}` | present |
| Gold features | `features/gold/gold_001.npz … gold_030.npz` | 30/30 present |
| Pseudo features | `features/pseudo/pseudo_00001.npz … pseudo_00080.npz` | 80/80 present |
| Metrics | `results/{rule_test_metrics.json, classifier_test_metrics.json, ablation_results.csv, rule_selected_config.json, training_history.csv, final_results_summary.{json,md}, model_comparison.csv, pseudo_labels.csv, …}` | present |
| Scripts | `scripts/00_…23_*.py`, `scripts/run_full_experiment.py` (the one that produced the locked results) | present |
| src | `src/{emoca_loader,data,features,metrics,pose_cnn,rules,…}.py` | present |
| Figures | `figures/` (incl. empty `figures/videomae/`) | present |
| Models / checkpoints / cache | `models/`, `checkpoints/`, `cache/` | **empty** (CNN weights `best_1dcnn.pt` were not kept; `.gitignore` excludes `models/*.pt`) |
| Local "clips" | `../data/nod30/<gold_XX,train_XX>/clip.mp4` | 80 × **13-byte placeholder files** containing the literal text `DEMO_NO_VIDEO` |

### 3. Split sizes

From `data/gold_annotations.csv`, `results/final_results_summary.json`, `results/pseudo_labels.csv`:

| Split | Clips | Labels | Source videos |
| --- | --- | --- | --- |
| Gold DEV | 15 (`gold_001`–`gold_015`) | 9 nod / 6 unclear | 15 unique RealTalk videos, no overlap with TEST |
| Gold TEST | 15 (`gold_016`–`gold_030`) | 10 nod / 5 unclear | 15 unique RealTalk videos |
| Pseudo TRAIN | 80 (`pseudo_00001`–`pseudo_00080`) | 70 nod / 10 unclear (rule-assigned) | 80 unique RealTalk videos, disjoint from gold |

- Every gold window is exactly **1500 frames = 60 s at 25 fps**; persons are `p0` (left) or `p1` (right); annotated windows start up to frame 40550 (~27 min into the source video), so any RGB acquisition must honour the frame offsets, not just grab any minute.
- Feature validity: gold `valid_ratio` 0.868–1.0 (23 rows in `results/feature_quality.csv`; all 30 npz exist); pseudo `valid_ratio` min 0.196 / mean 0.839 / max 1.0.
- Total feature-bearing clips: **110** (30 gold + 80 pseudo).

### 4. Clip/video IDs list

**Gold DEV (15):** `IH6KWbTogT0, P4ul1tuvi9c, xkHwlcDSOjc, RzIxWA-ll8g, D8K1AAxkg0g, niEsUBm1l98, wrhnUxQrx8g, sfUBZaWn2f8, GJtqigeWHV8, jg6y3LABwTs, FzCjvLU7u7Q, 6RDkdbgzeAI, WrWFSBLjWZU, f6aNo5Mod9I, Ak2Bm8mfL3w`

**Gold TEST (15):** `Cusa1_4R_QI, oQNpe8uwSUA, Zrer1sqWzOQ, YDI27aeM2O8, V1tcw5SLwmM, MGXtWqf1_BA, jn_3yDP58Ik, zS-xXIiLrWw, PM3oaJMiDd4, J4XrvnkftL8, G6tLY8FiheE, VSdVKQhnD9s, N-6L1u42cnw, PDd6qEv0_7c, ktR3_bXoxaE`

**Pseudo TRAIN (80, from the `video_id` field inside each npz):** `PyUSAycJHYY, EYq10SpLKb4, cDXtKSnAEQs, fosfLImKexU, T5BWozYHHWI, vWFCAiWvz-g, jVcbcgUWumE, DZ9QCoCRRtY, pSgpP1b0rHo, QzVs-z3A4mc, rwx5n0WfXCo, eZHx_biEECg, O98D2Tlgs-E, H_xMzAWWsNU, tNV8Dq88JFI, GKG2u7T1w2Y, jZdqe7X4IZ4, Q0WUgyqQ-4I, L1jJn9KNjws, xHRe-mZsu8M, MkV7LSXtzkQ, GW-cYEzShmY, qiqRKp3-rxc, IhWJ47-ueiw, 6tswigYQxag, 8PlPyrKM4cs, Fvsyr_nCpAw, zIhDJWOjfCI, p4ZNy6tbiiU, rRCPKtLSef4, epAwpBdUiYU, 0_rZe8PJo-o, UTkuDYOIxKo, WQCx6AS53Wc, MJueiM5zCxs, vI2hfZ3hGDM, mVt6myGMa-0, ayQLGgHx3co, uLE1b45zAN0, 0e0e4tE42oY, ftU60Qmh-pE, NAqVRJuZMD8, Q0u2BSEimEo, mU4XNYhEJNY, DxrVhhxyxlM, sCdn0hyfxYw, qULqdIzvE-o, 3kzx1BBJYDw, fJ2BOxlucBQ, MLz-8ghlWUE, e-XXxjlOOB8, YzPThnV66Pk, 96_F4gCW2eg, 2XWkpFoQn_E, qLfPQzNx-8s, k39nedJ6B5U, eiqEXQrOuIU, RbTQXe1Wjxw, wxp8uTNhvs0, A0HmdHU7Jms, v-qTIR5vxWs, GVsoOyn12Ck, c7NcnrOec5Y, NQEgTcXzwtY, eIEBI2_Y14Q, 9o-81SSaXcI, EgeWk8MSGdU, vAVBZSgXeeo, 87bpWKdM100, lLrPmgnL1AY, qSymgMQuVaM, Ot-d-PmuhOY, jOVEs-4r1jc, -uaFgyxn2kw, N91omgpkthA, p99cIB7nHjw, r-ts8-sJVko, yWUk1fDbAI4, OrOZuWk9FH4, A4jnhOZK8iY`

### 5. Disk used

Measured today with `du -sh`:

| Location | Size |
| --- | ---: |
| Whole workspace (`/Users/divyabisht/Downloads/Msc Dissertation Divya`) | **1.5 GB** |
| `realtalk_nod_forecasting/` (sibling project; 1.4 GB of it is its `.venv`) | 1.4 GB |
| `data/` (workspace root; the `nod30` placeholder clips + pose csv/png/pkl) | 32 MB |
| `.git/` | 15 MB |
| root `.venv/` (Python 3.9.0, near-empty) | 14 MB |
| project `dissertation-behaviour-recognition/` total | ~21 MB |
| — `features/` | 12 MB (gold 3.6 MB + pseudo 8.5 MB) |
| — `figures/` | 8.3 MB |
| — `scripts/` + `src/` | ~0.35 MB |
| — `results/` + `data/` + `logs/` | ~0.16 MB |
| — `models/`, `checkpoints/`, `cache/` | 0 B |

### 6. Disk free

- **Today (df -h):** 233 Gi total, 198 Gi used, **16 Gi available** (93 % full) on `/dev/disk1s2`.
- Project's own logs (`logs/disk_before.txt`, `results/storage_before.json`): 22.94 GB free on 2026-08-15/16 → about 7 GB has been consumed since (largely the 1.4 GB torch venv plus unrelated system use).
- Lab context (from repo notes + your instruction): **otter48 home had ~6.5 GB free after the CPU torch install** against a ~24–25 GB quota (`LAB_NOTES.local.md`: "Storage cap: 24 GB"; `LAB_COMMANDS.md`: "stay under ~25 GB; keep ≥5 GB free"). **Otter home cannot hold RealTalk video shards.**

### 7. GPU / VRAM on this Mac

- **Intel Iris Plus Graphics** (integrated), 1536 MB max dynamic VRAM, Metal supported; x86_64 CPU.
- **No CUDA, no discrete GPU, no Apple-Silicon MPS.** Any torch work here is CPU-only.

### 8. Does raw RGB exist locally?

**No.** A filesystem-wide search of the workspace for `*.mp4/*.avi/*.mov/*.mkv/*.webm` found only:

- 80 × `data/nod30/*/clip.mp4` — all 13-byte `DEMO_NO_VIDEO` placeholders (synthetic pipeline), and
- 2 × `world.mp4` gradio test fixtures inside `realtalk_nod_forecasting/.venv/.../site-packages/` (irrelevant).

The dissertation features contain only EMOCA pose/expression arrays (npz keys: `frames`, `rotation_xyz` (1500×3 float32), `expression` (1500×20 float32), `valid_ratio`, `video_id`, `person`, `sample_id`) — no pixels. Raw RealTalk RGB exists only as `videos/videos_{xx}.tar` shards (50 full-length `.avi` per shard, 25 fps) in the HF dataset; the dataset card reports ~299 GB total for the repo and 25.3 GB for `emoca.tar.gz` **[external, verify per-shard size before any fetch]**. The 110 needed videos are spread across up to 14 shards, so "download one shard" does not solve it.

### 9. Biggest directories (workspace)

1. `realtalk_nod_forecasting/.venv` — 1.4 GB (torch 2.2.2 CPU, torchvision 0.17.2, transformers 4.57.6)
2. `data/nod30` — 32 MB (synthetic placeholders; no video content)
3. project `features/` — 12 MB
4. project `figures/` — 8.3 MB
5. `.git` — 15 MB; root `.venv` — 14 MB

There is no local HF cache (`.hf_cache`), no checkpoints dir content, and no video cache anywhere in the workspace.

### 10. Reusable files for a smallest VideoMAE experiment

| Asset | Reusable as |
| --- | --- |
| `data/gold_annotations.csv` + `data/splits/*` | The 110-clip manifest: video IDs, 25 fps frame offsets, person of interest, frozen DEV/TEST partition |
| `features/gold/*.npz`, `features/pseudo/*.npz` | Existing pose/expression baselines; direct inputs to the E2 delta follow-up (no video needed) |
| `results/rule_selected_config.json` | Frozen rule (axis x, thr 16.3538°) to regenerate identical pseudo-labels |
| `results/pseudo_labels.csv` | Ready-made 70/10 training labels for an RGB-embedding classifier |
| `scripts/run_full_experiment.py` | Harness (`load_npz`, `build_matrix`, `resample_seq`, `CNN`, `run_mode`) — the E2 follow-up is a small patch to this file |
| `../realtalk_nod_forecasting/.venv` | **Python 3.9.13 with torch 2.2.2 + torchvision 0.17.2 + transformers 4.57.6 already installed** (CPU). Transformers ships `VideoMAEModel` + `VideoMAEImageProcessor`, so a frozen-VideoMAE feature extractor needs no new DL install on this Mac |
| `opencv-python-headless` (present in that venv, 5.0.0.93; also pinned in `scripts/nod_pipeline/requirements.txt`) | Video decode + YuNet face detection (YuNet onnx model file is a ~0.3 MB download) |
| `.gitignore` | Already excludes `*.mp4`, big tars, `models/*.pt` — nothing to fix |

Not reusable / missing: trained CNN weights (`models/` empty — the mode-C checkpoint is overwritten by later ablation modes; see `reports/repository_validation.md` §4.3), any RGB. Since this preflight was written, the frozen-head experiment it planned has been executed once via `scripts/extract_videomae_embeddings.py` + `scripts/train_videomae_head.py` (artifacts in `results/videomae_frozen_head/`; TEST F1 0.57 — below the pose CNN, not a headline). The numbered `15_/16_` scripts are documentation pointers; a decord/av fast-reader remains optional, not required.

### 11. Estimated extra storage for the smallest VideoMAE experiment

Frozen VideoMAE, 16-frame face crops, 110 clips. Measured where possible; **[estimate]** flags arithmetic from published sizes, not local measurement.

| Component | Size | Basis |
| --- | ---: | --- |
| VideoMAE-Small checkpoint (ViT-S, ~22 M params) | ~90 MB fp32 | **[estimate]** from published architecture; verify file size at download |
| (Alternative: VideoMAE-Base, ViT-B, ~87 M params) | ~350 MB fp32 | **[estimate]** same caveat |
| Face detector (OpenCV YuNet onnx) | ~0.3 MB | published model size **[estimate]**; opencv already installed |
| Face crops, 110 clips × 16 frames × 224×224×3 uint8 | ~265 MB raw (~10–30 MB as JPEG) | arithmetic: 110×16×150 528 B |
| Embedding cache, clip-level (110 × 768 × fp32) | ~0.34 MB | arithmetic |
| Embedding cache, dense window-level (110 × ~93 windows × 768 × fp32) | ~31 MB | arithmetic (1500 frames ÷ 16-stride) |
| Trimmed 60 s source clips, optional cache (110 × ~5–15 MB mp4/avi) | ~0.5–1.7 GB | **[estimate]** depends on encode settings |
| **Total, no source-video cache (VideoMAE-Small)** | **≲ 0.4 GB** | fits everywhere |
| **Total, with cached trimmed clips + Base checkpoint** | **≲ 2.4 GB** | fits Mac; fits lab quota only outside otter home |

The experiment's own footprint is therefore **under ~1 GB** (under ~2.5 GB even keeping trimmed RGB). The gating item is not experiment storage — it is **getting the 110 source videos at the right frame offsets**, because the HF video shards are multi-GB each and otter48 home (~6.5 GB free) cannot hold even one shard; extraction must stream members to scratch/tmp or to the Mac without landing a whole shard.

---

## GO / NO-GO table — smallest RGB VideoMAE experiment

| Requirement | What it needs | Fits 25 GB lab quota? | Fits this Mac (16 GiB free now, 24 GB policy cap)? |
| --- | --- | --- | --- |
| **RealTalk video access** | 110 specific 60 s windows at known 25 fps offsets; no local RGB today; HF serves only 50-video multi-GB `.tar` shards (~299 GB repo **[external, verify]**) | **NO into otter home** (~6.5 GB free < one shard). Possible only via stream-extract to lab scratch/tmp, or re-fetch from the YouTube source with per-clip frame-offset re-verification | Yes for trimmed clips (~0.5–1.7 GB), but requires a download step the current plan forbids, and YouTube re-encodes must be re-aligned to the annotated frame offsets |
| **VideoMAE checkpoint** | ~90 MB (Small) / ~350 MB (Base), one-time | Yes | Yes |
| **Face-crop dependency** | Per-frame face detect+crop on 16 sampled frames/clip; opencv present; YuNet model ~0.3 MB; no mediapipe installed/needed | Yes | Yes |
| **Embeddings cache** | 0.34 MB clip-level / ~31 MB dense | Yes | Yes |
| **Compute** | Frozen inference, 110 clips: minutes-scale on CPU; head training trivial | Yes (otter CPU torch exists) | Yes (torch 2.2.2 CPU in `realtalk_nod_forecasting/.venv`); no GPU on this Mac (Iris Plus only, no CUDA/MPS) |
| **Protocol / rules** | SUBMISSION_48H.md: "Do not start VideoMAE"; TEST-once discipline means one declared TEST look only | — | — |

**Verdict:** **Conditional GO on resources, NO-GO inside the 2-day submission window.** The smallest experiment (frozen VideoMAE-Small, 16-frame face crops on the 110 known clips, clip-level embedding cache) needs ≲ 0.4 GB of new artifacts plus ~0.5–1.7 GB of trimmed video, so both the 25 GB lab quota and this Mac's 16 GiB free can hold it with several GB to spare — storage and CPU compute are not the blocker. The blocker is video acquisition and protocol: no RGB exists locally, the HF video shards cannot fit in otter48's home (~6.5 GB free after CPU torch), stream-trimming 110 windows out of up to 14 shards (or re-downloading from YouTube and re-verifying frame offsets against `gold_annotations.csv`) is a data-engineering task with real failure modes, and the locked 48-hour submission plan explicitly forbids running VideoMAE and allows only one TEST evaluation. Within 2 days this should stay a future-work paragraph; after submission it is a legitimate, cheap follow-up provided video lands on lab scratch (not home) and the TEST look is declared once.

---

## E2 — does `results/ablation_results.csv` already cover motion?

**Partly, yes — but it cannot isolate motion, and no delta-only row exists.**

The four rows come from `build_matrix` in `scripts/run_full_experiment.py`, where `drot = np.vstack([np.zeros((1,3)), np.diff(rot, axis=0)])` (signed frame-to-frame first difference of Euler xyz, zero-padded at t=0):

| Row | Input | Dims | TEST F1 |
| --- | --- | ---: | ---: |
| `single_axis` | raw x rotation only | 1 | 0.70 |
| `xyz` | raw x,y,z | 3 | 0.6957 |
| `xyz_deriv` | **[raw x,y,z] concatenated with [Δx,Δy,Δz]** | 6 | 0.70 |
| `xyz_deriv_expr` | the above + 20 EMOCA expression dims | 26 | 0.0 (diverged — dev_f1 0.0; do not report) |

- **`xyz_deriv` does encode frame-to-frame motion**, but as **raw + signed delta concatenated** — the raw pose stays in the input, so the row cannot attribute performance to motion itself, and the deltas are signed, not absolute.
- **No dedicated delta-only row and no `abs(delta)` row exists** in `ablation_results.csv` (or anywhere else in `results/`). On the locked TEST (n=15), adding deltas moved F1 by only 0.6957 → 0.70 anyway, so the current table neither proves nor disproves a motion contribution.
- A tiny follow-up needs **no VideoMAE, no video, no new packages**: patch `build_matrix` with two modes — e.g. `deriv_only` (Δx,Δy,Δz; 3 dims) and `abs_deriv` (|Δx|,|Δy|,|Δz|; 3 dims) — and call the existing `run_mode` unchanged: same 110 cached npz features, same 80 pseudo-train clips with the frozen rule's 70/10 labels, same length-128 resampling and pseudo-train normalisation, same 1D CNN, same seed 42, same 15-epoch/DEV-threshold selection, TEST once per variant. Roughly 40 lines, CPU-minutes on the torch venv that already exists. Caveat to state in the report: each extra variant is another look at TEST, so it must be declared as a post-hoc ablation, not folded into the locked results table.
