# Final repository audit

Repository cleanup for dissertation submission and public GitHub (1 September 2026).

Checkpoint before this pass: local `d876d13`. Locked TEST inference was not rerun.

## Files changed (this pass)

- Root and package `README.md` (title unchanged: *Predicting Backchannel Events from Multimodal Conversational Signals*; CVSSP line kept)
- `dissertation-behaviour-recognition/scripts/README.md` (new)
- `dissertation-behaviour-recognition/requirements.txt`, `requirements-video.txt`, `requirements-audio.txt`
- `dissertation-behaviour-recognition/.gitignore` (`*.wav`, `data/audio_alignment_check/`)
- `dissertation-behaviour-recognition/scripts/check_split_leakage.py` (lock n=120 and n=200 VideoMAE dirs)
- `dissertation-behaviour-recognition/tests/test_invariants.py` (locked joint dir must be refused)
- `dissertation-behaviour-recognition/scripts/15_train_videomae.py` (pointer to archive preflight)
- `dissertation-behaviour-recognition/reports/repository_validation.md`, `reports/README.md`, `reports/SCOPE_MAP.md`
- `archive/README.md`

## Files archived

Moved to `archive/reports/`:

- `videomae_preflight.md`
- `videomae_preflight_lab.md`
- `videomae_next_lab_commands.md`
- `SUBMISSION_48H.md`
- `WRITING_INVENTORY.md`

Earlier seven-class demo remains under `archive/api`, `archive/web`, `archive/notebooks`, `archive/scripts`.

## README

First screen keeps the dissertation title and CVSSP affiliation. Headline TEST F1 table is pose rule 0.67 / pose CNN 0.70 / frozen VideoMAE 0.57 / fine-tuned VideoMAE **0.82**. DEV audio is separate. Weak supervision, TEST lock, shake z-axis audit, n=80 vs n=200, and RealTalk not-redistributed are stated.

## Dependencies

- `requirements.txt` — CPU pipeline, figures, pytest
- `requirements-video.txt` — recommended torch/transformers (from saved metrics: torch 2.13.0, transformers 5.15.1; not a frozen historic lockfile)
- `requirements-audio.txt` — unchanged extra audio packages

## Tests

`python -m pytest -q` → **22 passed**, 14 matplotlib warnings (1 Sep 2026).

The previous failure `test_videomae_shake_path_isolation` expected writes into `results/joint/videomae_finetuned`. That directory is locked; the test now asserts refusal. The lock was not weakened.

## Metric consistency (from stored TEST predictions, not new inference)

| system | recomputed F1 | matches saved JSON |
| --- | ---: | --- |
| pose rule | 0.6667 | yes |
| pose CNN | 0.7000 | yes |
| frozen VideoMAE | 0.5714 | yes |
| fine-tuned VideoMAE n=80 | 0.8182 | yes |
| fine-tuned VideoMAE n=200 | 0.6316 | yes |

## TEST-lock

`LOCKED_OUT_DIRS` includes nod/shake/joint VideoMAE TEST directories plus n=120 and n=200. Ordinary scripts still must call `assert_unlocked_out_dir` before writing.

## Licensing

RealTalk videos are not in git. Face-crop figures may be identifiable; terms are not independently cleared here. Prefer pose traces for public slides if needed.

## Unresolved / deliberately not changed

- Canonical TEST metrics were not edited.
- Historical shake TEST (axis z) was not rescored.
- Local Mac `main` is still out of sync with `origin/main` (ahead/behind). Do not `git add -A`.
- Untracked leftover paper figures, `.vendor/`, shake-v2 search trees, and chapter-draft WIP were left out of this cleanup commit on purpose.
- `FINAL_REPOSITORY_AUDIT.md` is this file.

Locked TEST model inference was not rerun.

Canonical TEST predictions were not overwritten.

Canonical reported TEST metrics were not changed unless supported by pre-existing saved result artefacts.

Historical shake TEST results were preserved.

No large dataset or model downloads were performed.
