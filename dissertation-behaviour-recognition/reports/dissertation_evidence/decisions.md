# Decisions

- Use package `dissertation-behaviour-recognition/` as the dissertation path (do not mix with older `scripts/nod_pipeline/`).
- Two annotation classes only: `1` clear nod, `0` unclear. Only class `1` is a gold positive.
- Primary event metric: F1 at IoU 0.30.
- 15 DEV for tuning; 15 TEST labelled now, scored once, never used for training.
- VideoMAE is not run until real nod-rule metrics exist on RealTalk pose.
- Synthetic pilot clips are for pipeline checks only and are not reported as RealTalk results.
