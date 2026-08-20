# Methodology notes

Submitted experiment (paste `reports/methods_chapter_draft.md`):

- Two annotation classes only: `1` clear nod, `0` unclear. Gold positives = class 1 only.
- **Headline metric:** clip-level precision / recall / F1 on 15 TEST windows. Event F1 at IoU 0.30 was **not** used for this 30-window protocol.
- Tune the nod rule on DEV only. Score GOLD TEST once; do not train on it.
- Scored rule = Savitzky–Golay peak-to-peak amplitude on a DEV-chosen Euler axis (x, 16.35°). Not the unused 1–3 Hz detector in `src/rules/nod.py`.
- EMOCA streamed from Hugging Face; `emoca.tar.gz` not saved. Conversion: `rotvec[:3]` → Euler xyz degrees. Physical pitch not assumed.
- Do not map unknown expression coefficients to eyebrow raise. Ablation D diverged.
- Do not fetch the full RealTalk video set or start VideoMAE on otter (25 GB quota; ~6.5 GB free after CPU PyTorch).
