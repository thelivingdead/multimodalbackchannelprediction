# Writing inventory (19 August 2026)

No main dissertation `.tex` or `.docx` exists in this workspace or in Downloads. There is nothing to extend except markdown notes. Paste the drafts below into Word (or Overleaf) yourself. Do not commit unless you choose to.

## What already exists (usable)

| Item | Path | Status |
| --- | --- | --- |
| Results draft (TEST headlines) | `reports/results_chapter_draft.md` | **Updated** for paste; use this, not DEV F1 |
| Methods draft | `reports/methods_chapter_draft.md` | **New** — matches the pose rule + 1D CNN |
| Discussion, limitations, conclusion | `reports/discussion_conclusion_draft.md` | **New** |
| Abstract, intro, literature | `reports/abstract_intro_lit_draft.md` | **New** — short but complete enough to submit |
| 48-hour paste plan | `reports/SUBMISSION_48H.md` | **Local planning note — Mac-only, uncommitted; the repo does not rely on it** |
| Figure captions | `reports/figure_captions.md` | **New** |
| Harvard references (verified only) | `reports/references_harvard.md` | **New** |
| Gold annotation note | `reports/gold_annotation_results.md` | Counts only; metrics superseded by Results |
| Report outline (submission) | `reports/dissertation_evidence/report_outline.md` | Current 48 h pose-only outline; use this |
| Early methodology / 7-class roadmap | `docs/archive/02_research_methodology_and_roadmap.md` | Proposal-era. Scope was reduced. Use `methods_chapter_draft.md` |
| Data analysis / corpus notes | `docs/archive/01_data_analysis_report.md` | Useful for RealTalk description; **do not claim 7-class results** |
| Literature map | `reports/dissertation_evidence/literature_map.md` | Seed papers only |
| Citation register | `reports/dissertation_evidence/citation_register.md` | Do not add unverified papers |
| Locked JSON metrics | `results/rule_test_metrics.json`, `results/classifier_test_metrics.json` | Source of truth |
| Figures | `dissertation-behaviour-recognition/figures/` | See `figure_captions.md` |

## What does **not** exist

- A compiled dissertation PDF in this repo
- Inter-annotator agreement
- Event-level F1 at IoU 0.30 for the 30-window protocol
- VideoMAE training or scores
- A trained EMOCA model (`emoca.tar.gz` was streamed, not saved)
- Ablation D as a valid result (training diverged)

## Executed experiment (one sentence)

Thirty human-labelled RealTalk windows (15 DEV / 15 TEST) were used to freeze a pose-amplitude rule and then to score a 1D CNN trained on 80 automatic rule labels; TEST was scored once.
