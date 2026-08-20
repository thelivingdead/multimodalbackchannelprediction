# Decisions

- Use package `dissertation-behaviour-recognition/` as the dissertation path (do not mix with older `scripts/nod_pipeline/`).
- Two annotation classes only: `1` clear nod, `0` unclear. Only class `1` is a gold positive.
- Primary **reported** metric for the 30-window study: clip-level F1 (not event IoU 0.30).
- 15 DEV for tuning; 15 TEST labelled, scored once, never used for training.
- VideoMAE is **not** run for submission (disk quota). Treat as future work, not a TEST score.
- Synthetic pilot clips are for pipeline checks only and are not reported as RealTalk results.
