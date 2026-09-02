# DEV nod event annotation

The old 60 s clip-level labels in `data/gold_annotations.csv` are unchanged.
This folder holds a new protocol: mark each clear nod *event* inside every DEV clip.
3 s window labels are written to `nod_windows_dev.csv` only after every DEV row has `reviewed=true`.

## Files

- `nod_event_entry.csv` — DEV fill-in sheet
- `nod_event_entry_test.csv` — TEST fill-in sheet (`gold_016`–`gold_030`); do not mix with DEV
- `annotation_status_test.csv` — TEST reviewed flags
- `nod_events_windowed.csv` — compiled DEV events
- `nod_events_windowed_test.csv` — compiled TEST events (`gold_016`–`gold_030`)
- `nod_windows_dev.csv` — 3 s DEV window labels (1 if the slice overlaps a nod)
- `nod_windows_test.csv` — 3 s TEST window labels (same rule; separate file)
- `annotation_status.csv` — `reviewed=true` with `n_nod_events=0` means the clip was watched and had no clear nods. `reviewed=false` means annotation is not finished (not a negative label)
- `clips/` — optional local `{sample_id}.mp4` files (gitignored). If absent, the tool uses YouTube for the RealTalk `video_id`

## Launch (Mac)

```bash
cd dissertation-behaviour-recognition
python scripts/annotate_nod_events_dev.py
```

Open http://127.0.0.1:8765/

Optional local video (if YouTube embedding fails):

```bash
python scripts/prepare_dev_annotation_clips.py
```

Do this on the Mac. Otter is not needed for annotation.

## Window rule (fixed, applied later)

- 60 s source clip, 25 fps
- 3 s window, 2 s stride, 1 s overlap
- windows: 0–3, 2–5, 4–7, … (29 windows; last is 56–59 s)
- label 1 if the window has any non-zero overlap with a nod event, else 0

Do not generate those labels until annotation is finished.

## After annotation (not now)

Window CSV generation is a local file step on this Mac (pose npz already in git). GPU training of 3 s models belongs on otter later. Use class-balanced binary loss when the window set is sparse, e.g. `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)` once the counts exist. Do not look at model scores while annotating.
