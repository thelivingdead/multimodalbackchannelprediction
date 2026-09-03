# Identity-fixed VideoMAE, DEV only

VideoMAE numbers are not written here until
`scripts/audit_target_person_crops.py` has passed and the two training jobs
have finished on Otter.

## What is already decided

- Previous RGB results stay withdrawn. This directory does not overwrite them.
- TEST window files are not read by the new fetcher or the new trainer.
- Target side is the annotator LEFT/RIGHT instruction, checked against gold
  `person` (`p0` = LEFT, `p1` = RIGHT on all 30 clips).
- New cropper: `src/crop_target_person.py`. If the target half has fewer than
  two consistent detections the window is `unresolved` and is excluded. The
  largest face in the full frame is never used as a fallback.

## After Otter

Fill in from `frozen_encoder/metrics.json` and
`last_blocks_unfrozen/metrics.json`. Then run
`scripts/plot_identity_fixed_comparison.py`.
