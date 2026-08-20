#!/usr/bin/env python3
"""VideoMAE Step 2 (otter48): range-walked index of the RealTalk video shards.

Builds ``results/video_shard_index.json``::

    video_id -> {"shard": "videos_XX.tar", "offset": int, "size": int}

covering exactly the wanted clips: the 30 gold ``video_id``s from
``data/gold_annotations.csv`` plus the 80 ``video_id``s embedded in
``features/pseudo/*.npz`` (110 unique expected). ``offset``/``size`` locate the
member's *data* bytes inside the shard, so a later step can pull one video with
a single ``Range: bytes=offset-(offset+size-1)`` request.

Nothing but that one KB-scale JSON is ever written: no shard, no member video,
no frame. Member data is never even transferred — each 64 KB Range read is
positioned at a tar member's 512-byte header, and the walk then jumps directly
to the next header at ``data_offset + ceil(size/512)*512``.

Stopping rules
--------------
* Global: the instant every wanted id is found the walk stops — the current
  shard is exited immediately and all remaining shards are skipped.
* Per shard: a shard is walked to its tar end-of-archive marker. There is no
  safe earlier per-shard exit, because which wanted ids a shard holds is
  unknown until it has been walked. Per-shard hit counts print at the end.

Hard aborts (SystemExit; do not retry, paste the message back)
--------------------------------------------------------------
* Any Range read answered with a status other than HTTP 206 (200 = the server
  ignored ``Range`` and would stream a multi-GB shard; 403 = gated asset).
* Free space on ``~`` below ``MIN_FREE_GB`` at a shard boundary.

Lab invocation (existing venv; requests + numpy only — no installs)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    source ../.venv/bin/activate
    python scripts/build_video_shard_index.py
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import requests

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
GOLD_CSV = PACKAGE_ROOT / "data" / "gold_annotations.csv"
PSEUDO_DIR = PACKAGE_ROOT / "features" / "pseudo"
OUT_JSON = PACKAGE_ROOT / "results" / "video_shard_index.json"

SHARD_URL = (
    "https://huggingface.co/datasets/scottgeng00/realtalk"
    "/resolve/main/videos/videos_{:02d}.tar"
)
SHARD_COUNT = 14  # videos_00.tar .. videos_13.tar, confirmed by the Step 1 probe

WINDOW = 64 * 1024  # bytes per Range read
TAR_BLOCK = 512
MIN_FREE_GB = 5.4  # command-sheet rule for this step (pipeline floor is 3.0 GB)
TIMEOUT_S = 60
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}
EXPECTED_WANTED = 110  # 30 gold + 80 pseudo; a mismatch is printed, not fatal

_STATS = {"requests": 0}


def free_gb() -> float:
    return shutil.disk_usage(Path.home()).free / 1024**3


def check_disk() -> None:
    free = free_gb()
    if free < MIN_FREE_GB:
        raise SystemExit(
            f"STOP: free disk on ~ is {free:.2f} GB < {MIN_FREE_GB} GB. "
            "This step is network plus one small JSON; find what is filling "
            "the disk before rerunning."
        )


def load_wanted_ids() -> tuple[set[str], set[str]]:
    with GOLD_CSV.open(newline="") as fh:
        gold = {row["video_id"].strip() for row in csv.DictReader(fh)}
    pseudo = set()
    for npz_path in sorted(PSEUDO_DIR.glob("*.npz")):
        with np.load(npz_path) as z:
            vid = z["video_id"]
        if hasattr(vid, "item"):
            vid = vid.item()
        if isinstance(vid, bytes):
            vid = vid.decode()
        pseudo.add(str(vid).strip())
    return gold, pseudo


def parse_tar_size(field: bytes) -> int:
    if field[0] & 0x80:  # base-256 large-file encoding
        return int.from_bytes(bytes([field[0] & 0x7F]) + field[1:], "big")
    return int(field.split(b"\0")[0].strip() or b"0", 8)


def fetch_range(url: str, lo: int, hi: int) -> tuple[bytes, int | None]:
    resp = None
    for attempt in (1, 2):
        try:
            resp = requests.get(
                url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=TIMEOUT_S
            )
            break
        except requests.RequestException as exc:
            if attempt == 2:
                raise SystemExit(
                    f"STOP: network error on {url} bytes {lo}-{hi}: {exc}"
                ) from exc
            time.sleep(5)
    _STATS["requests"] += 1
    if resp.status_code != 206:
        meaning = {
            200: "server ignored the Range header and answered the whole shard",
            403: "gated asset / missing authorisation",
        }.get(resp.status_code, "unexpected status")
        raise SystemExit(
            f"STOP: range read {url} bytes {lo}-{hi} returned "
            f"HTTP {resp.status_code} ({meaning}). Do not retry; paste this back."
        )
    total = None
    content_range = resp.headers.get("Content-Range", "")
    if "/" in content_range:
        try:
            total = int(content_range.rsplit("/", 1)[1])
        except ValueError:
            pass
    return resp.content, total


def walk_shard(shard_idx: int, wanted: set[str], found: dict) -> list[str]:
    shard_name = f"videos_{shard_idx:02d}.tar"
    url = SHARD_URL.format(shard_idx)
    hits: list[str] = []
    pending_longname: str | None = None
    members = 0
    pos = 0  # absolute offset of the next tar header
    while True:
        window, total = fetch_range(url, pos, pos + WINDOW - 1)
        if members == 0 and total:
            print(f"  {shard_name}: shard size {total / 1e9:.2f} GB")
        if len(window) < TAR_BLOCK:
            break  # short/empty answer at EOF: no further header can start here
        inner = 0
        end_of_archive = False
        while inner + TAR_BLOCK <= len(window):
            block = window[inner : inner + TAR_BLOCK]
            if not any(block):
                # first zero block of the end-of-archive marker
                end_of_archive = True
                break
            header_abs = pos + inner
            name = block[0:100].split(b"\0")[0].decode("utf-8", "replace")
            prefix = block[345:500].split(b"\0")[0].decode("utf-8", "replace")
            if prefix:
                name = f"{prefix}/{name}"
            if pending_longname is not None:
                name, pending_longname = pending_longname, None
            size = parse_tar_size(block[124:136])
            typeflag = block[156:157]
            data_abs = header_abs + TAR_BLOCK
            if typeflag == b"L":
                # GNU longname entry: the next member's real name is this data
                raw = window[inner + TAR_BLOCK : inner + TAR_BLOCK + size]
                pending_longname = raw.split(b"\0")[0].decode("utf-8", "replace")
            elif typeflag in (b"0", b"\0"):
                member = Path(name)
                vid = member.stem
                if (
                    member.suffix.lower() in VIDEO_EXTS
                    and vid in wanted
                    and vid not in found
                ):
                    found[vid] = {
                        "shard": shard_name,
                        "offset": data_abs,
                        "size": size,
                    }
                    hits.append(vid)
                    print(
                        f"  hit {len(found)}/{len(wanted)}: {vid} in "
                        f"{shard_name} @ {data_abs} ({size / 1e6:.1f} MB)"
                    )
                    if wanted <= found.keys():
                        return hits
            members += 1
            if members % 100 == 0:
                pct = f" ({100 * header_abs / total:.0f}%)" if total else ""
                print(
                    f"  {shard_name}: {members} members walked, {len(hits)} "
                    f"wanted here, offset {header_abs / 1e9:.2f} GB{pct}"
                )
            next_header = data_abs + ((size + TAR_BLOCK - 1) // TAR_BLOCK) * TAR_BLOCK
            if next_header - pos + TAR_BLOCK <= len(window):
                inner = next_header - pos  # another small member in this window
            else:
                pos = next_header  # jump: member data is never transferred
                break
        if end_of_archive:
            break
    return hits


def main() -> None:
    gold, pseudo = load_wanted_ids()
    wanted = gold | pseudo
    print(
        f"wanted ids: {len(gold)} gold + {len(pseudo)} pseudo = "
        f"{len(wanted)} unique (gold/pseudo overlap: {len(gold & pseudo)})"
    )
    if len(wanted) != EXPECTED_WANTED:
        print(
            f"NOTE: expected {EXPECTED_WANTED} unique wanted ids, got "
            f"{len(wanted)} — continuing; check the coverage printout."
        )

    found: dict[str, dict] = {}
    per_shard: dict[str, list[str]] = {}
    stopped_early = False
    for shard_idx in range(SHARD_COUNT):
        check_disk()
        print(
            f"shard {shard_idx + 1}/{SHARD_COUNT} (videos_{shard_idx:02d}.tar), "
            f"{free_gb():.2f} GB free on ~"
        )
        per_shard[f"videos_{shard_idx:02d}.tar"] = walk_shard(
            shard_idx, wanted, found
        )
        if wanted <= found.keys():
            stopped_early = True
            print("all wanted ids found — stopping; remaining shards skipped")
            break
    check_disk()

    missing = sorted(wanted - found.keys())
    print("\n==== coverage ====")
    print(f"found {len(found)}/{len(wanted)} wanted ids")
    for shard_name, ids in per_shard.items():
        line = f"  {shard_name}: {len(ids)} wanted"
        if ids:
            line += f" — {', '.join(sorted(ids))}"
        print(line)
    skipped = SHARD_COUNT - len(per_shard)
    if stopped_early and skipped:
        print(f"  ({skipped} shard(s) never touched)")
    if missing:
        print(f"missing {len(missing)}: {', '.join(missing)}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    ordered = {vid: found[vid] for vid in sorted(found)}
    OUT_JSON.write_text(json.dumps(ordered, indent=2) + "\n")
    transferred_mb = _STATS["requests"] * WINDOW / 1e6
    print(
        f"\nwrote {OUT_JSON} ({OUT_JSON.stat().st_size} bytes); "
        f"{_STATS['requests']} range reads, ~{transferred_mb:.0f} MB "
        "transferred, 0 bytes of video kept"
    )
    if missing:
        raise SystemExit(
            f"INCOMPLETE: {len(missing)} wanted ids were not seen in any "
            "shard. The JSON holds only the found ids; paste the coverage back."
        )


if __name__ == "__main__":
    main()
