#!/usr/bin/env python3
"""Step A: verify source-video audio on 3–5 existing nod windows (DEV-only).

Downloads (or reuses) the RealTalk member for each selected gold clip, probes
video/audio streams, extracts the ~60 s watch-window WAV, and checks:

* source video id, listener side (p0=LEFT / p1=RIGHT), timestamps
* duration, native sample rate, non-empty / audible audio
* video–audio streams both present; fps ≈ 25; window fits the file

WAVs go to ``data/audio_alignment_check/`` (gitignored). The verdict is
``results/audio_alignment_check.md`` + ``.json``. GOLD TEST ids are refused.

This script does **not** train a model and does **not** score TEST.

    OMP_NUM_THREADS=1 python scripts/audio_alignment_check.py
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/audio_alignment_check.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_io import (  # noqa: E402
    DEFAULT_ALIGN_IDS,
    WAV_DIR,
    alignment_checks,
    extract_window_wav,
    inventory_clip,
    load_shard_index,
    load_wav_mono,
    probe_media,
    refuse_test_scoring,
    resolve_video_file,
    wav_stats,
)
from src.utils import dump_json  # noqa: E402

OUT_MD = ROOT / "results" / "audio_alignment_check.md"
OUT_JSON = ROOT / "results" / "audio_alignment_check.json"
MIN_PASS = 3


def _md_table(rows: list[dict]) -> str:
    cols = [
        "sample_id",
        "video_id",
        "split",
        "person",
        "label",
        "t0_s",
        "duration_s",
        "src_sr",
        "wav_sr",
        "wav_dur_s",
        "rms",
        "pass",
    ]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        cells = []
        for c in cols:
            val = row.get(c, "")
            if isinstance(val, float):
                cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(payload: dict) -> None:
    status = payload["status"]
    lines = [
        "# Audio alignment check (Step A)",
        "",
        f"**Verdict: {status}.** "
        f"{payload['n_passed']} / {payload['n_checked']} selected nod windows "
        "passed source-video / timestamp / audible-audio checks. "
        "GOLD TEST was not used.",
        "",
        "Approved dissertation title (unchanged): "
        "**Predicting Backchannel Events from Multimodal Conversational Signals**.",
        "",
        "Audio is the RealTalk **container soundtrack** for the watch window "
        "(both participants), not a separated listener channel. Pose and RGB "
        "are visual encodings of the same camera; this check only asks whether "
        "that camera file has usable, time-aligned audio.",
        "",
        f"Pass rule: at least {MIN_PASS} clips must pass, each with a video "
        "stream, an audio stream, fps ≈ 25, non-empty audible WAV whose duration "
        "matches the 1500-frame / 25 fps pose window (±0.5 s), and matching "
        "LEFT/RIGHT listener vs `who_to_watch`.",
        "",
        "## Clips",
        "",
        _md_table(payload["table_rows"]),
        "",
    ]
    if payload.get("blocker"):
        lines += ["## Blocker", "", payload["blocker"], ""]
    for clip in payload["clips"]:
        sid = clip["sample_id"]
        lines += [
            f"## {sid} (`{clip.get('video_id', '')}`)",
            "",
            f"- origin: `{clip.get('origin')}` split **{clip.get('split')}** "
            f"(TEST clips are refused)",
            f"- person: `{clip.get('person')}` → {clip.get('speaker_side')} "
            f"| sheet: {clip.get('who_to_watch')}",
            f"- pose frames: {clip.get('start_frame')}–"
            f"{clip.get('end_frame_inclusive')} inclusive "
            f"({clip.get('n_frames')} frames @ 25 fps → "
            f"{clip.get('duration_s')} s) from t0={clip.get('t0_s')} s",
            f"- watch clock: {clip.get('watch_from')}–{clip.get('watch_until')} "
            f"| marked nod: {clip.get('nod_start')}–{clip.get('nod_end')}",
            f"- youtube: {clip.get('youtube_url')}",
            f"- source: {clip.get('video_source')}",
            f"- probe fps={clip.get('fps')} sr={clip.get('src_sample_rate')} "
            f"duration={clip.get('src_duration_s')} "
            f"has_video={clip.get('has_video')} has_audio={clip.get('has_audio')}",
            f"- WAV: `{clip.get('wav_path')}` duration={clip.get('wav_duration_s')} "
            f"sr={clip.get('wav_sr')} peak={clip.get('peak')} rms={clip.get('rms')}",
            f"- checks: **{'PASS' if clip.get('pass') else 'FAIL'}** "
            f"{clip.get('reasons')}",
            "",
        ]
    lines += [
        "## Next step",
        "",
        "Step B (`scripts/train_audio_baseline_dev.py`) and Step C "
        "(`scripts/train_av_fusion_dev.py`) must **not** run unless this "
        "verdict is PASS. They are DEV-only and refuse GOLD TEST.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ids",
        default=",".join(DEFAULT_ALIGN_IDS),
        help="comma-separated gold sample_ids (DEV only; default: five mixed clips)",
    )
    parser.add_argument(
        "--score-test",
        action="store_true",
        default=False,
        help="refused. GOLD TEST is locked.",
    )
    parser.add_argument(
        "--split",
        default="dev",
        help="must be dev (TEST is refused)",
    )
    parser.add_argument(
        "--keep-video",
        action="store_true",
        help="keep temporary mp4 members (still gitignored). Default: delete.",
    )
    args = parser.parse_args(argv)
    refuse_test_scoring(score_test=bool(args.score_test), split=args.split)

    ids = [s.strip() for s in str(args.ids).split(",") if s.strip()]
    if not ids:
        raise SystemExit("STOP: no sample ids.")
    if len(ids) < 3 or len(ids) > 5:
        print(
            f"NOTE: {len(ids)} ids selected; protocol asks for 3–5. Continuing."
        )

    index = load_shard_index()
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = WAV_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    clips: list[dict] = []
    table_rows: list[dict] = []
    blockers: list[str] = []

    for sample_id in ids:
        meta = inventory_clip(sample_id)
        if str(meta.get("split", "")).upper() == "TEST":
            raise SystemExit(
                f"STOP: {sample_id} is GOLD TEST. Alignment check uses DEV "
                "(or TRAIN) windows only."
            )
        rec = dict(meta)
        rec["pass"] = False
        rec["reasons"] = ["not run"]
        video_path = None
        delete_video = False
        try:
            video_path, prov, delete_video = resolve_video_file(
                meta["video_id"],
                index=index,
                tmp_dir=tmp_dir,
                keep=bool(args.keep_video),
            )
            rec["video_source"] = prov.get("source")
            rec["video_path"] = str(video_path)
            probe = probe_media(video_path)
            rec.update(
                {
                    "fps": probe.get("fps"),
                    "src_sample_rate": probe.get("sample_rate_hz"),
                    "src_duration_s": probe.get("duration_s"),
                    "has_video": probe.get("has_video"),
                    "has_audio": probe.get("has_audio"),
                    "video_line": probe.get("video_line"),
                    "audio_line": probe.get("audio_line"),
                    "width": probe.get("width"),
                    "height": probe.get("height"),
                }
            )
            wav_path = WAV_DIR / f"{sample_id}_{meta['video_id']}.wav"
            extract_window_wav(
                video_path,
                wav_path,
                t0_s=float(meta["t0_s"]),
                duration_s=float(meta["duration_s"]),
            )
            y, sr = load_wav_mono(wav_path)
            stats = wav_stats(y, sr)
            rec["wav_path"] = str(wav_path)
            rec["wav_sr"] = stats["sample_rate_hz"]
            rec["wav_duration_s"] = stats["duration_s"]
            rec["peak"] = stats["peak"]
            rec["rms"] = stats["rms"]
            rec["n_samples"] = stats["n_samples"]
            checks = alignment_checks(meta, probe, stats)
            rec["pass"] = bool(checks["pass"])
            rec["reasons"] = checks["reasons"]
            rec["checks"] = checks
        except SystemExit as exc:
            rec["reasons"] = [str(exc)]
            blockers.append(f"{sample_id}: {exc}")
            print(f"BLOCKED {sample_id}: {exc}")
        except Exception as exc:  # noqa: BLE001 — per-clip failure must not invent pass
            rec["reasons"] = [f"{type(exc).__name__}: {exc}"]
            blockers.append(f"{sample_id}: {exc}")
            print(f"FAIL {sample_id}: {exc}")
        finally:
            if delete_video and video_path is not None and Path(video_path).exists():
                try:
                    Path(video_path).unlink()
                except OSError:
                    pass
        clips.append(rec)
        table_rows.append(
            {
                "sample_id": rec.get("sample_id"),
                "video_id": rec.get("video_id"),
                "split": rec.get("split"),
                "person": rec.get("person"),
                "label": rec.get("label"),
                "t0_s": rec.get("t0_s"),
                "duration_s": rec.get("duration_s"),
                "src_sr": rec.get("src_sample_rate"),
                "wav_sr": rec.get("wav_sr"),
                "wav_dur_s": rec.get("wav_duration_s"),
                "rms": rec.get("rms"),
                "pass": rec.get("pass"),
            }
        )
        print(
            f"{sample_id} {rec.get('video_id')} "
            f"{'PASS' if rec.get('pass') else 'FAIL'} {rec.get('reasons')}"
        )

    n_pass = sum(1 for c in clips if c.get("pass"))
    n_audio = sum(1 for c in clips if c.get("has_audio"))
    status = "PASS" if n_pass >= MIN_PASS and n_audio >= MIN_PASS else "FAIL"
    blocker = None
    if status == "FAIL":
        blocker = (
            f"Step A FAIL: {n_pass}/{len(clips)} clips passed "
            f"(need {MIN_PASS}); audio streams on {n_audio} clips. "
            "Do not run Step B or C. " + (" ".join(blockers) if blockers else "")
        ).strip()
    payload = {
        "status": status,
        "n_checked": len(clips),
        "n_passed": n_pass,
        "min_pass": MIN_PASS,
        "gold_test_scored": False,
        "task": "supervised prediction of the backchannel label associated "
        "with a conversational window (alignment check only)",
        "blocker": blocker,
        "clips": clips,
        "table_rows": table_rows,
        "free_gb_end": shutil.disk_usage(Path.home()).free / 1024**3,
    }
    dump_json(OUT_JSON, payload)
    write_report(payload)
    print(f"\n{status}: wrote {OUT_MD} and {OUT_JSON}")
    if status != "PASS":
        raise SystemExit(blocker or "Step A FAIL")


if __name__ == "__main__":
    main()
