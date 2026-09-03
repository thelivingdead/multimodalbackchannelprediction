# Person / side mapping

Established from committed files, not from a guessed convention.

## Sources

- `data/gold_annotations.csv` column `person`: RealTalk speaker index `p0` or `p1`
- `data/gold/watch_list.csv` column `who_to_watch`: annotator instruction
  starting `LEFT` or `RIGHT`
- `data/gold/shake_annotation_sheet.csv` uses the same `who_to_watch` wording

## Check

Joined every gold clip by `video_id`. All 30 clips agree:

| person | watch_side | n clips |
| --- | --- | ---: |
| p0 | LEFT | 15 |
| p1 | RIGHT | 15 |

No clip has `p0` with RIGHT or `p1` with LEFT.

DEV examples: `gold_001` is `p0` / LEFT. `gold_006` is `p1` / RIGHT. That
matches the watch-list sentences "watch the person on the LEFT" and
"watch the person on the RIGHT".

## What the new cropper uses

Target side is the LEFT/RIGHT instruction the annotator followed
(`watch_list.who_to_watch`). Before a clip is fetched, the script asserts
that `person` matches that side under the table above. If they ever disagree
the fetch stops instead of guessing.

`person` is therefore a RealTalk participant index. It is not a free-form
name. In this gold set it is interchangeable with LEFT/RIGHT, and that
interchange is checked, not assumed per clip.
