# VideoMAE pre-flight — LAB (otter48) only

**Date:** 20 August 2026.
**Target machine:** `db01550@otter48` (home `/user/HS400/db01550`). VideoMAE must run there, **not** on the Mac — the lab GPU (if present) is the point.
**Audit basis:** this Mac repo (`/Users/divyabisht/Downloads/Msc Dissertation Divya`, git main) plus locked lab facts from the user's own otter terminal logs earlier this session. Nothing here was downloaded, trained, installed, or committed for this audit. No number in this file is invented; measured figures are quoted with their source, and size figures marked `~` are order-of-magnitude estimates to be confirmed by the verification commands in §8 **before any GO**.

**Locked lab facts (from otter terminal logs):**
- Free disk on otter after the CPU PyTorch install: **6.54 GB** (experiment summary line `Disk free GB: 6.54`).
- Home quota: **~25 GB**.
- `~/multimodalbackchannelprediction/.venv` has **Python 3.12 + torch 2.13.0+cpu** (CPU build).
- GPU: **never verified** — `nvidia-smi` does not appear in any visible log.
- Lab clone: `~/multimodalbackchannelprediction`, `git reset --hard` to GitHub `main`; contains `features/gold` (30 npz) and `features/pseudo` (80 npz).

---

## 1. Current pipeline and exact lab paths

What exists and has already run (source of truth: `results/`, `reports/SCOPE_MAP.md`):

1. **Gold annotation** — 30 × ~60 s RealTalk windows, human-labelled binary (clear nod `1` / unclear `0`), frame windows in `data/gold_annotations.csv`.
2. **Pose feature extraction** — EMOCA/FLAME pickles **streamed** from `huggingface.co/datasets/scottgeng00/realtalk` (`emoca.tar.gz`, opened as `r|gz` over HTTP, never saved) → per-clip npz with `rotation_xyz`, `expression`, `valid_ratio`, `video_id`, `person`, `sample_id` (`scripts/run_full_experiment.py::stream_emoca`).
3. **Rule baseline** — savgol-smoothed single-axis rotation amplitude; axis **x**, threshold **16.35°** frozen on DEV (`results/rule_selected_config.json`); TEST scored once: P 0.64 / R 0.70 / F1 **0.67**.
4. **Pseudo-labels** — frozen rule applied to 80 unlabelled TRAIN clips → 70 nod / 10 unclear (`results/pseudo_labels.csv`).
5. **1D CNN** (3× Conv1d, CPU torch) on xyz + first differences (6-D, 128 steps), epoch 9 and probability threshold chosen on DEV; TEST scored once: P 0.70 / R 0.70 / F1 **0.70** (`results/classifier_test_metrics.json`).
6. **Ablations A–D** in the same script → `results/ablation_results.csv` (see §9).
7. **VideoMAE frozen head — first run 20 Aug 2026 (not a headline).** `scripts/extract_videomae_embeddings.py` (110 clips, frozen `MCG-NJU/videomae-base`, 768-d; `results/videomae_embeddings_meta.json`) + `scripts/train_videomae_head.py` (MLP head on the 80 pseudo-labels, DEV-selected): DEV F1 0.90, **TEST F1 0.57** (P 0.55, R 0.60; TP6 FP5 TN0 FN4) in `results/videomae_frozen_head/` — below both pose models, so the submitted story stays pose-only. **Not run:** VideoMAE fine-tuning and fusion (`configs/videomae_finetune.yaml`, `configs/fusion.yaml` remain `planned_not_run`). The numbered `scripts/15_/16_/17_*.py` are documentation pointers; the earlier `src/videomae_model.py` / `src/fusion_model.py` guard stubs were removed in the 20 Aug cleanup.

Exact lab paths (repo root = `~/multimodalbackchannelprediction` = `/user/HS400/db01550/multimodalbackchannelprediction`):

| What | Lab path |
| --- | --- |
| Package root | `~/multimodalbackchannelprediction/dissertation-behaviour-recognition/` |
| Gold features (30 npz, git-tracked) | `…/dissertation-behaviour-recognition/features/gold/gold_001.npz … gold_030.npz` |
| Pseudo features (80 npz, git-tracked) | `…/dissertation-behaviour-recognition/features/pseudo/pseudo_00001.npz … pseudo_00080.npz` |
| Gold labels + frame windows | `…/dissertation-behaviour-recognition/data/gold_annotations.csv` |
| Split lists | `…/dissertation-behaviour-recognition/data/splits/{gold_dev.txt,gold_test.txt,gold_split.csv,planned_dev.txt,planned_test.txt}` |
| Frozen rule config | `…/dissertation-behaviour-recognition/results/rule_selected_config.json` |
| End-to-end pipeline script | `…/dissertation-behaviour-recognition/scripts/run_full_experiment.py` |
| Venv (torch 2.13.0+cpu, py 3.12) | `~/multimodalbackchannelprediction/.venv` |

Tracked in git main (so the lab clone already has them): gold/pseudo npz (110 files), gold CSVs, split lists, `ablation_results.csv`, rule and classifier TEST metrics, pseudo labels, configs. Mac-only and **not** on the lab unless pushed: untracked `results/*.json` summaries, `reports/` drafts, `figures/`, and all of §5's video discussion.

## 2. Split and file lists

**Gold: 30 human-labelled clips, 15 DEV / 15 TEST, zero source-video overlap** (`results/gold_dataset_summary.json`).

| | DEV | TEST |
| --- | ---: | ---: |
| Windows (sample_ids) | `gold_001`–`gold_015` | `gold_016`–`gold_030` |
| Clear nod / unclear | 9 / 6 | 10 / 5 |
| Source videos | 15 ids in `data/splits/gold_dev.txt` | 15 ids in `data/splits/gold_test.txt` |

DEV videos: `6RDkdbgzeAI, Ak2Bm8mfL3w, D8K1AAxkg0g, FzCjvLU7u7Q, GJtqigeWHV8, IH6KWbTogT0, P4ul1tuvi9c, RzIxWA-ll8g, WrWFSBLjWZU, f6aNo5Mod9I, jg6y3LABwTs, niEsUBm1l98, sfUBZaWn2f8, wrhnUxQrx8g, xkHwlcDSOjc`.
TEST videos: `Cusa1_4R_QI, G6tLY8FiheE, J4XrvnkftL8, MGXtWqf1_BA, N-6L1u42cnw, PDd6qEv0_7c, PM3oaJMiDd4, V1tcw5SLwmM, VSdVKQhnD9s, YDI27aeM2O8, Zrer1sqWzOQ, jn_3yDP58Ik, ktR3_bXoxaE, oQNpe8uwSUA, zS-xXIiLrWw`.
Per-clip frame windows (start/end at 25 fps) and scored person (`p0` = LEFT, `p1` = RIGHT) are in `data/gold_annotations.csv`.

**Pseudo TRAIN: 80 clips** (`pseudo_00001`–`pseudo_00080`), one ~60 s window from each of 80 non-gold videos, labels from the frozen rule: 70 nod / 10 unclear (`results/pseudo_labels.csv`). Feature files: `features/pseudo/pseudo_*.npz`.

## 3. LAB disk — decision numbers

| Figure | Value | Source |
| --- | --- | --- |
| **otter free disk (decision number)** | **6.54 GB** | otter terminal log, experiment summary `Disk free GB: 6.54`, after CPU PyTorch install |
| **otter home quota** | **~25 GB** | user-reported lab account limit; consistent with methods draft §4.1 |
| Mac free disk (**Mac-only, not a lab number**) | 22.94 → 22.93 GB of 233.47 GB | `results/storage_before.json` / `storage_after.json` (measured on the Mac during the pose run; the ~0.02 GB delta shows pose streaming costs ~nothing on disk) |
| Feature npz payload (identical bytes on both machines via git) | 3.6 MB gold + 8.5 MB pseudo | `du` on this repo; git-tracked, so already inside the lab clone's used quota |

Working rule already used by the pipeline: keep **≥3 GB free** at all times (`run_full_experiment.py::stop_if_low_disk`, `MIN_FREE_GB = 3.0`). With 6.54 GB free, the usable budget for *everything new* (checkpoint + deps + RGB crops + embeddings) is **~3.5 GB**.

## 4. GPU — status UNKNOWN

- **No GPU evidence exists.** `nvidia-smi` was never run in any visible otter log. Treat GPU presence as UNKNOWN until §8 command 4 is run.
- **CPU torch is already present**: torch 2.13.0+cpu in `~/multimodalbackchannelprediction/.venv` (Python 3.12). Frozen VideoMAE inference runs on this without any new install of torch itself.
- **Do not plan on CUDA torch even if a GPU appears.** The CUDA wheel set (torch + `nvidia-*` pip libraries) is multi-GB (typically ~5–8 GB installed) and does **not** fit in ~3.5 GB of usable space. A GPU with the existing CPU wheel would still be CPU execution; installing CUDA torch on this quota is a NO-GO (§7).

## 5. Raw RGB on the lab — none

Default answer: **no RealTalk RGB exists on otter, and none exists on the Mac either.** Evidence:

- Repo-wide search finds **no real `.mp4`/`.avi` anywhere**. The only `clip.mp4` files on the Mac are (a) ~32 MB of **synthetic** demo clips under Mac-only `data/nod30/` (generated by `scripts/nod_pipeline/run_30video_plan.sh`; explicitly *not* RealTalk data — `scripts/nod_pipeline/LAB_COMMANDS.md`), and (b) placeholder files containing literal `NO_REAL_VIDEO` / `DEMO_NO_VIDEO` bytes (`scripts/make_pilot_clips.py`, `scripts/nod_pipeline/01_make_tiny_subset.py`).
- `.gitignore` excludes `*.mp4 *.avi *.mov *.tar *.tar.gz *.pkl`; README: "Large binaries stay off git". So the lab clone (reset to git main) cannot contain video.
- `reports/SCOPE_MAP.md` locked fact: "**no video frames on otter**" — only pose npz were copied/committed.
- `scripts/02_audit_data.py` lists `emoca.tar.gz` and "full RealTalk shards" under `do_not_download`.
- RealTalk RGB lives only at the source: **Hugging Face `scottgeng00/realtalk`, `videos/videos_*.tar` shards** (videos as per-video `.avi`/`.mp4` members inside multi-GB tars; `scripts/nod_pipeline/01_make_tiny_subset.py` already contains member-wise extraction logic for exactly this layout). **No per-video RGB file exists locally on either machine.**

## 6. Smallest-VideoMAE requirements

Config intent (`configs/videomae_frozen.yaml`): model `MCG-NJU/videomae-base`, `frames: 16`, `window_s: 2.0`, `frozen: true`.

The smallest defensible variant — **frozen VideoMAE-Base embeddings on the existing 110 windows**, scored by a tiny head trained on DEV-protocol lines:

| Component | Size | Note |
| --- | --- | --- |
| Frozen checkpoint `MCG-NJU/videomae-base` (ViT-B/16, ~87 M params) | ~0.4 GB class (~350–400 MB fp32) | one-time download; verify with `du` after fetch |
| New Python deps on top of existing torch-cpu: `transformers`, `huggingface_hub`, `safetensors`, `opencv-python-headless` (or `decord`/`av`) | ~0.3–0.5 GB installed (estimate) | no CUDA, no `mediapipe`, no `timm` beyond what transformers pulls |
| Face detector | ~0.2–10 MB | OpenCV YuNet ONNX (~0.2 MB) or bundled Haar cascade; must crop the **scored** person (`p0` = LEFT half, `p1` = RIGHT half — RealTalk frames show both participants) |
| Sampled RGB: 110 clips × 16 frames × 224×224×3 uint8 | ~0.27 GB if stored raw as npz; tens of MB if re-encoded as small mp4 crops | only the 16 sampled frames per window are ever kept |
| Embeddings: 110 × 768-D pooled fp32 | ~0.3 MB (MB class) | even unpooled token maps (~1570 tokens × 768) ≈ ~0.5 GB total — fits, but pool anyway |
| Tiny head on embeddings (logreg / 1-layer) | negligible | CPU, seconds |

**The binding constraint is not the model — it is the RGB supply.** Gold windows sit at known frame ranges (`data/gold_annotations.csv`) inside specific source videos, but those videos exist only as members of multi-GB `videos_*.tar` shards that must never touch the lab disk (§3, §7). Therefore:

1. **Stream, never save shards.** Reuse the proven pattern from `run_full_experiment.py::_ProgressStream` (HTTP stream → `tarfile` `r|gz`, per-member processing, stall timeouts, early stop). The 23.6 GB `emoca.tar.gz` was successfully streamed this way on otter already, so multi-GB HF streaming from otter is demonstrated in practice.
2. **Shard→video_id mapping is not known locally.** The HF API lists shard file names, not tar contents. Worst case, shards are streamed until all 30 gold video ids (+80 pseudo ids, if pseudo embeddings are wanted) have been seen — network time, not disk. This lookup is the main **schedule risk** inside the 2-day window.
3. **Decode per member in memory**, seek to the window, sample 16 frames, face-crop the scored side, write only the 16-frame crop, release the member. Never decode full-length clips to disk (110 × ~1500 frames at source resolution = tens of GB — does not fit).
4. Runtime is CPU torch 2.13.0+cpu (§4). 110 forward passes of a frozen ViT-B on 16-frame clips is a small, finite CPU job; no benchmark exists in this repo, so wall-clock time is unverified — smoke-test on 2 clips first.

If extraction stalls, the fallback that still produces a dissertation-usable number is **gold-30-only embeddings** (30 clips instead of 110; DEV/TEST protocol unchanged, rule/CNN numbers unchanged).

## 7. GO/NO-GO for otter (25 GB quota, ~6.5 GB free, ~3.5 GB usable)

| Item | Fits in ~3.5 GB? | Verdict |
| --- | --- | --- |
| VideoMAE-Base frozen checkpoint (~0.4 GB) | yes | **GO** |
| `transformers` + `opencv-python-headless` + tiny ONNX face model (~0.3–0.5 GB) | yes, tight | **GO** |
| 110 × 16-frame 224² face crops stored raw (~0.27 GB) or re-encoded (tens of MB) | yes | **GO** |
| Pooled 768-D embeddings for 110 clips (~0.3 MB) + tiny CPU head | yes | **GO** |
| Frozen embedding extraction on existing CPU torch | yes | **GO** (smoke-test 2 clips first) |
| Any whole `videos_*.tar` shard written to disk (multi-GB each) | **no** | **NO-GO** — stream member-wise only |
| Full-length decoded RGB for 110 × ~60 s clips (tens of GB) | **no** | **NO-GO** — sample 16 frames in memory, discard the rest |
| CUDA PyTorch stack (~5–8 GB installed) | **no** | **NO-GO** on this quota, even if §8 finds a GPU |
| VideoMAE **fine-tuning** (optimizer states + grads + checkpoints, GB-scale) | **no** | **NO-GO** — frozen features + tiny head only |
| Hugging Face cache growth during streaming | risk | **streamed members must not land in `~/.cache/huggingface`**; use raw `requests` streaming (as `run_full_experiment.py` does), not `hf_hub_download`, for shards |
| Deletion plan | — | per-clip: delete nothing needed; keep only 16-frame crops + embeddings; abort if `df -h ~` shows <3 GB free (pipeline rule `MIN_FREE_GB = 3.0`) |

## 8. Verification commands — LAB user on otter48, run these ONLY (nothing else until GO)

```bash
# on otter48 (db01550@otter48) — read-only checks
df -h ~                                        # expect ~6.5 GB free; abort plan if < 4.5 GB
du -sh ~/multimodalbackchannelprediction \
       ~/multimodalbackchannelprediction/.venv \
       ~/multimodalbackchannelprediction/dissertation-behaviour-recognition/features
nvidia-smi                                     # GPU presence UNKNOWN until this runs; record output verbatim
ls ~/multimodalbackchannelprediction/dissertation-behaviour-recognition/features/gold   | wc -l   # expect 30
ls ~/multimodalbackchannelprediction/dissertation-behaviour-recognition/features/pseudo | wc -l   # expect 80
cd ~/multimodalbackchannelprediction && source .venv/bin/activate && \
  python -c "import sys, torch; print(sys.version.split()[0]); print(torch.__version__, 'cuda_available:', torch.cuda.is_available())"
```

Paste the full output back before any install, download, or training. Expected: torch `2.13.0+cpu`, `cuda_available: False` under a CPU build (that is normal and does not by itself prove no GPU exists — only `nvidia-smi` settles hardware presence).

## 9. E2 — does `ablation_results.csv` (row `xyz_deriv`) already cover delta-motion?

**Yes, partially — and it is not delta-only.** Row `xyz_deriv` is feature set C, built in `scripts/run_full_experiment.py::build_matrix` as `concat[rotation_xyz (3), Δrotation (3)]` where `Δ = np.diff` over time (zero-padded first frame) — 6-D, resampled to 128 steps. So first-difference (delta-motion) channels **are** in the reported headline model: TEST F1 0.70, P 0.70, R 0.70, accuracy 0.60, balanced accuracy 0.55 (best epoch 9, DEV F1 0.889). The marginal effect of adding deltas is readable against row `xyz` (B, 3-D): F1 0.696 → 0.700, precision 0.615 → 0.700, recall 0.80 → 0.70, balanced accuracy 0.40 → 0.55.

**What it does not answer:** whether delta-motion *alone* (no absolute pose) carries the signal — there is no Δ-only or |Δ|-only row (A = x-only, B = xyz, C = xyz+Δ, D = xyz+Δ+expression which diverged with `loss = nan`).

A tiny abs(delta-only) follow-up needs, end to end:

1. One new mode in `build_matrix` (e.g. mode `"E"`: `feat = np.abs(drot)` or plain `drot`, 3-D) — two lines; inputs are the existing local npz files (`features/gold` 30 + `features/pseudo` 80 already contain `rotation_xyz`); **no streaming, no new labels, no new downloads**; minutes of CPU.
2. Same CNN, same seed 42, same protocol: epoch and probability threshold chosen on DEV only, then TEST scored once.
3. Append one row to `results/ablation_results.csv` and log the run in `reports/dissertation_evidence/experiment_log.md`.
4. **Protocol caveat to decide before running:** TEST has already been scored for rows A–D. Either declare the Δ-only row part of the same pre-registered ablation family (A–E) in `reports/dissertation_evidence/decisions.md` before looking at its TEST number, or run it DEV-only. Do not silently add a sixth TEST evaluation.

---

*Audit performed from the Mac repo; no lab (otter) commands were run for this audit. Decision numbers are the lab numbers (§3); Mac-only figures are labelled as such.*
