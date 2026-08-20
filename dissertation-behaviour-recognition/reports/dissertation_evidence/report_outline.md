# Report outline (submission, 48 h)

Use this outline, not the older 7-class / VideoMAE experiment list.

1. Abstract (`abstract_intro_lit_draft.md`) — TEST F1 0.67 / 0.70 / 0.57 (frozen VideoMAE) / 0.82 (fine-tuned VideoMAE); CIs overlap
2. Introduction — problem, scope reduction, RQ1–RQ3, contributions (incl. frozen-vs-fine-tuned VideoMAE contrast)
3. Related work — RealTalk, EMOCA/FLAME, backchannels, weak supervision, video transformers (VideoMAE run in two constrained forms; full fine-tuning as future work)
4. Data and annotation — 30 windows, 1/0 protocol, 15/15, ethics
5. Methods — streamed EMOCA, frozen amplitude rule, 80 pseudo-labels, 1D CNN, VideoMAE frozen head + partial fine-tune (GPU on `/scratch`), clip-level P/R/F1
6. Results — locked TEST table (master: `results/tables/main_results.md`); ablations A–C; error list; no D; VideoMAE frozen 0.57 and fine-tuned 0.82 with CI-overlap caveat
7. Discussion
8. Limitations
9. Conclusion and future work
10. References (`references_harvard.md`)
11. Appendix — TRAIN loss / DEV F1-by-epoch if space

Paste files (all under `reports/`):

- `SUBMISSION_48H.md` — hour-by-hour (local planning note, Mac-only and uncommitted; not a committed repo artefact)
- `abstract_intro_lit_draft.md`
- `methods_chapter_draft.md`
- `results_chapter_draft.md`
- `discussion_conclusion_draft.md`
- `figure_captions.md`
- `WRITING_INVENTORY.md`
