#!/usr/bin/env python3
"""VideoMAE Step 3 (otter48): fetch 16-frame RGB face-crop windows per clip.

For every wanted sample (30 gold ``features/gold/*.npz`` + 80 pseudo
``features/pseudo/*.npz``) this pulls **only that clip's member bytes** out of
its Hugging Face ``videos_*.tar`` shard — one ``Range: bytes=offset-(offset+size-1)``
request per clip, using ``results/video_shard_index.json`` (Step 2 output) —
decodes 16 uniformly spaced frames **in memory** with the ffmpeg binary bundled
with ``imageio-ffmpeg`` (bytes are piped to ffmpeg's stdin; nothing touches
disk), face-crops, and writes one compact uint8 npz per clip::

    features/rgb16/<sample_id>.npz
        rgb            uint8 (16, 224, 224, 3)   RGB, face-cropped
        frame_indices  int32 (16,)               absolute source-video frames
        sample_id / video_id / person            scalars (from the pose npz)
        crop_box       int32 (4,)                x, y, w, h in native pixels
        crop_mode      str                       'haar' | 'centre'
        n_faces        int                       faces found on the middle frame

No shard, no full-length video, and no JPG frame is ever written to disk. The
member bytes live only in RAM for the duration of one clip's decode.

Face-crop provenance (documented limitation)
--------------------------------------------
The EMOCA-derived pose npz files contain only ``frames, rotation_xyz,
expression, valid_ratio, video_id, person, sample_id`` — **no camera, scale,
or bounding-box keys** (verified on the committed files), and the raw EMOCA
pickles are not on disk. A face box therefore cannot be recovered from EMOCA
outputs. This script instead runs the Haar frontal-face cascade shipped inside
``opencv-python-headless`` on the middle decoded frame and takes the **largest**
detection, expanded by ``CROP_SCALE`` and squared. If no face is found it falls
back to a centred square crop (``crop_mode='centre'``). When several faces are
found (RealTalk videos show two people) the largest is used and the sample is
flagged ``n_faces > 1`` in the summary — the ``person`` field (p0/p1) cannot be
mapped to image position without manual verification, so no left/right
assignment is invented.

Hard rules
----------
* Free space on ``~`` must stay above ``MIN_FREE_GB`` (5.4 GB); checked before
  and after every clip. Abort below it.
* Any Range read answered with a status other than HTTP 206 aborts the run.
* Frames are selected by absolute decode index (0-based), matching the
  convention of the npz ``frames`` arrays (FPS = 25). If the video decodes
  fewer frames than requested, the sample is marked ``short_decode`` and
  **skipped** — frames are never padded or fabricated.
* Existing outputs are skipped, so reruns resume.

Lab invocation (existing venv + ``pip install opencv-python-headless imageio-ffmpeg``)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    source ../.venv/bin/activate
    python scripts/fetch_rgb_windows.py --ids gold_001,gold_016   # smoke: 2 clips
    python scripts/fetch_rgb_windows.py                            # full: all wanted
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import subprocess
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import requests

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = PACKAGE_ROOT / "features" / "gold"
PSEUDO_DIR = PACKAGE_ROOT / "features" / "pseudo"
INDEX_JSON = PACKAGE_ROOT / "results" / "video_shard_index.json"
OUT_DIR = PACKAGE_ROOT / "features" / "rgb16"
SUMMARY_JSON = PACKAGE_ROOT / "results" / "rgb16_fetch_summary.json"

SHARD_URL = (
    "https://huggingface.co/datasets/scottgeng00/realtalk"
    "/resolve/main/videos/{}"
)

FPS = 25.0  # project convention (run_full_experiment.py); frames are absolute
N_FRAMES = 16
CROP_SIZE = 224
CROP_SCALE = 1.6  # face box linear expansion, EMOCA-style head crop
MIN_FREE_GB = 5.4  # command-sheet rule for this step
TIMEOUT_S = 180  # one member can be tens of MB
WRITE_CHUNK = 1 << 20
STDERR_TAIL = 4096

_STATS = {"requests": 0, "bytes": 0}


def free_gb() -> float:
    return shutil.disk_usage(Path.home()).free / 1024**3


def check_disk(where: str = "") -> None:
    free = free_gb()
    if free < MIN_FREE_GB:
        raise SystemExit(
            f"STOP: free disk on ~ is {free:.2f} GB < {MIN_FREE_GB} GB"
            f"{' at ' + where if where else ''}. Remove partial artefacts "
            "before rerunning; completed clips are skipped on resume."
        )


def npz_scalar(z: zipfile.ZipFile, name: str) -> str:
    arr = np.load(io.BytesIO(z.read(name)), allow_pickle=True)
    if hasattr(arr, "item"):
        arr = arr.item()
    if isinstance(arr, bytes):
        arr = arr.decode()
    return str(arr)


def load_samples() -> list[dict]:
    """Wanted clips from the committed pose npz files (window = frames array)."""
    samples = []
    for split_dir, origin in ((GOLD_DIR, "gold"), (PSEUDO_DIR, "pseudo")):
        for npz_path in sorted(split_dir.glob("*.npz")):
            with zipfile.ZipFile(npz_path) as z:
                frames = np.load(io.BytesIO(z.read("frames.npy")))
                sample = {
                    "sample_id": npz_scalar(z, "sample_id.npy"),
                    "video_id": npz_scalar(z, "video_id.npy"),
                    "person": npz_scalar(z, "person.npy"),
                    "origin": origin,
                    "frame_lo": int(frames.min()),
                    "frame_hi": int(frames.max()),
                }
            if sample["sample_id"] != npz_path.stem:
                raise SystemExit(
                    f"STOP: {npz_path.name} embeds sample_id "
                    f"{sample['sample_id']!r} — filename mismatch; investigate "
                    "before fetching."
                )
            samples.append(sample)
    return samples


def uniform_indices(lo: int, hi: int, n: int) -> np.ndarray:
    idx = np.linspace(lo, hi, n).round().astype(np.int64)
    return np.unique(idx)  # dedupe guard; length checked downstream


def fetch_member(url: str, offset: int, size: int) -> bytes:
    resp = None
    for attempt in (1, 2):
        try:
            resp = requests.get(
                url,
                headers={"Range": f"bytes={offset}-{offset + size - 1}"},
                timeout=TIMEOUT_S,
            )
            break
        except requests.RequestException as exc:
            if attempt == 2:
                raise SystemExit(
                    f"STOP: network error on {url} bytes {offset}-"
                    f"{offset + size - 1}: {exc}"
                ) from exc
            time.sleep(5)
    _STATS["requests"] += 1
    if resp.status_code != 206:
        meaning = {
            200: "server ignored the Range header and answered the whole shard",
            403: "gated asset / missing authorisation",
        }.get(resp.status_code, "unexpected status")
        raise SystemExit(
            f"STOP: range read {url} bytes {offset}-{offset + size - 1} "
            f"returned HTTP {resp.status_code} ({meaning}). Do not retry; "
            "paste this back."
        )
    data = resp.content
    _STATS["bytes"] += len(data)
    if len(data) != size:
        raise SystemExit(
            f"STOP: range read returned {len(data)} bytes, index says {size}. "
            "The shard index does not match the served bytes; rebuild it."
        )
    return data


def _drain(stream, sink: bytearray, cap: int | None = None) -> None:
    while True:
        chunk = stream.read(1 << 16)
        if not chunk:
            break
        if cap is None:
            sink += chunk
        else:  # keep only the tail (stderr log)
            sink += chunk
            if len(sink) > cap:
                del sink[: len(sink) - cap]


def decode_frames(video_bytes: bytes, frame_indices: np.ndarray) -> np.ndarray:
    """Decode exactly ``frame_indices`` (absolute, 0-based) via an ffmpeg pipe.

    Returns uint8 array (n, H, W, 3) with n == len(frame_indices); raises
    RuntimeError otherwise. Requires imageio-ffmpeg (bundled ffmpeg binary).
    """
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise SystemExit(
            "STOP: imageio-ffmpeg is not installed. Run "
            "`pip install imageio-ffmpeg opencv-python-headless` in the venv."
        ) from exc
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    expr = "+".join(f"eq(n,{i})" for i in frame_indices)
    cmd = [
        exe, "-hide_banner", "-loglevel", "info",
        "-an", "-sn", "-dn",
        "-i", "pipe:0",
        "-vf", f"select='{expr}'",
        "-vsync", "0",  # passthrough: select must not be padded to CFR
        "-frames:v", str(len(frame_indices)),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out_buf, err_buf = bytearray(), bytearray()
    t_out = threading.Thread(target=_drain, args=(proc.stdout, out_buf, None))
    t_err = threading.Thread(
        target=_drain, args=(proc.stderr, err_buf, STDERR_TAIL)
    )
    t_out.start()
    t_err.start()
    try:
        for pos in range(0, len(video_bytes), WRITE_CHUNK):
            proc.stdin.write(video_bytes[pos : pos + WRITE_CHUNK])
    except (BrokenPipeError, OSError):
        pass  # ffmpeg exited early; stderr tail below explains
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
    try:
        proc.wait(timeout=600)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        proc.wait()
        raise RuntimeError("ffmpeg decode timed out after 600 s") from exc
    t_out.join()
    t_err.join()
    err_tail = bytes(err_buf).decode("utf-8", "replace")
    match = re.search(rb"Video: .{0,200}?(\d{2,6})x(\d{2,6})", bytes(err_buf))
    if not match:
        raise RuntimeError(
            "no video stream dimensions in ffmpeg output: " + err_tail[-300:]
        )
    width, height = int(match.group(1)), int(match.group(2))
    frame_bytes = width * height * 3
    n_out = len(out_buf) // frame_bytes
    if proc.returncode not in (0, None) and n_out == 0:
        raise RuntimeError(
            f"ffmpeg exited {proc.returncode}: {err_tail[-300:]}"
        )
    if n_out < len(frame_indices):
        raise RuntimeError(
            f"short_decode: wanted {len(frame_indices)} frames up to index "
            f"{int(frame_indices[-1])}, video yielded {n_out}"
        )
    frames = np.frombuffer(bytes(out_buf[: n_out * frame_bytes]), np.uint8)
    return frames.reshape(n_out, height, width, 3)


def crop_window(frames: np.ndarray) -> tuple[np.ndarray, tuple, str, int]:
    """Face-crop all frames with one box taken on the middle frame.

    Returns (crops (n, 224, 224, 3) uint8 RGB, (x, y, w, h), mode, n_faces).
    Haar cascade from opencv-python-headless; centred-square fallback.
    """
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "STOP: opencv-python-headless is not installed. Run "
            "`pip install imageio-ffmpeg opencv-python-headless` in the venv."
        ) from exc
    height, width = frames.shape[1:3]
    mid = frames[len(frames) // 2]
    gray = cv2.cvtColor(mid, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )
    n_faces = len(faces)
    if n_faces:
        x, y, w, h = max(faces, key=lambda b: int(b[2]) * int(b[3]))
        side = int(round(max(w, h) * CROP_SCALE))
        cx, cy = x + w // 2, y + h // 2
        side = min(side, width, height)
        x0 = int(min(max(cx - side // 2, 0), width - side))
        y0 = int(min(max(cy - side // 2, 0), height - side))
        mode = "haar"
    else:
        side = min(width, height)
        x0, y0 = (width - side) // 2, (height - side) // 2
        mode = "centre"
    box = (x0, y0, side, side)
    crops = np.stack(
        [
            cv2.resize(
                f[y0 : y0 + side, x0 : x0 + side],
                (CROP_SIZE, CROP_SIZE),
                interpolation=cv2.INTER_AREA,
            )
            for f in frames
        ]
    )
    return crops, box, mode, n_faces


def save_clip(out_path: Path, crops: np.ndarray, sample: dict,
              frame_indices: np.ndarray, box: tuple, mode: str,
              n_faces: int) -> None:
    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez(
        tmp_path,
        rgb=crops.astype(np.uint8),
        frame_indices=frame_indices.astype(np.int32),
        sample_id=sample["sample_id"],
        video_id=sample["video_id"],
        person=sample["person"],
        crop_box=np.asarray(box, dtype=np.int32),
        crop_mode=mode,
        n_faces=np.int64(n_faces),
    )
    tmp_path.rename(out_path)  # atomic: no partial npz on abort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N clips (after --ids filter)")
    parser.add_argument("--ids", type=str, default=None,
                        help="comma-separated sample_ids, e.g. gold_001,gold_016")
    args = parser.parse_args()

    if not INDEX_JSON.exists():
        raise SystemExit(
            f"STOP: {INDEX_JSON} does not exist. Run "
            "scripts/build_video_shard_index.py (Step 2) first and paste its "
            "coverage output back."
        )
    index = json.loads(INDEX_JSON.read_text())
    samples = load_samples()
    id_filter = None
    if args.ids:
        id_filter = {s.strip() for s in args.ids.split(",") if s.strip()}
        known = {s["sample_id"] for s in samples}
        unknown = sorted(id_filter - known)
        if unknown:
            raise SystemExit(
                f"STOP: unknown --ids {unknown}; valid ids are the stems of "
                "features/gold/*.npz and features/pseudo/*.npz"
            )
    if id_filter:
        samples = [s for s in samples if s["sample_id"] in id_filter]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("STOP: no samples selected.")

    no_index = [s["sample_id"] for s in samples if s["video_id"] not in index]
    print(
        f"{len(samples)} clips selected "
        f"({sum(s['origin'] == 'gold' for s in samples)} gold / "
        f"{sum(s['origin'] == 'pseudo' for s in samples)} pseudo); "
        f"{free_gb():.2f} GB free on ~"
    )
    if no_index:
        raise SystemExit(
            f"STOP: {len(no_index)} selected clips have no shard-index entry: "
            f"{no_index}. Rebuild results/video_shard_index.json."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    check_disk("start")
    records: dict[str, dict] = {}
    if SUMMARY_JSON.exists():
        records = json.loads(SUMMARY_JSON.read_text()).get("clips", {})

    for i, sample in enumerate(samples, 1):
        sid = sample["sample_id"]
        out_path = OUT_DIR / f"{sid}.npz"
        if out_path.exists():
            print(f"[{i}/{len(samples)}] {sid}: exists — skipped")
            continue
        check_disk(sid)
        entry = index[sample["video_id"]]
        url = SHARD_URL.format(entry["shard"])
        frame_indices = uniform_indices(
            sample["frame_lo"], sample["frame_hi"], N_FRAMES
        )
        t0 = time.time()
        try:
            blob = fetch_member(url, int(entry["offset"]), int(entry["size"]))
            frames = decode_frames(blob, frame_indices)
            crops, box, mode, n_faces = crop_window(frames)
            save_clip(out_path, crops, sample, frame_indices, box, mode, n_faces)
        except RuntimeError as exc:
            records[sid] = {
                "sample_id": sid,
                "video_id": sample["video_id"],
                "origin": sample["origin"],
                "status": "failed",
                "reason": str(exc)[:300],
            }
            print(f"[{i}/{len(samples)}] {sid}: FAILED — {exc}")
            continue
        check_disk(sid)
        records[sid] = {
            "sample_id": sid,
            "video_id": sample["video_id"],
            "origin": sample["origin"],
            "status": "ok",
            "crop_mode": mode,
            "n_faces": n_faces,
            "member_bytes": int(entry["size"]),
            "frame_first": int(frame_indices[0]),
            "frame_last": int(frame_indices[-1]),
            "elapsed_s": round(time.time() - t0, 1),
        }
        print(
            f"[{i}/{len(samples)}] {sid} ({sample['video_id']}): ok, "
            f"crop={mode} faces={n_faces}, "
            f"{entry['size'] / 1e6:.1f} MB ranged, "
            f"{records[sid]['elapsed_s']} s, {free_gb():.2f} GB free"
        )

    ok = sum(r.get("status") == "ok" for r in records.values())
    failed = {s: r.get("reason", "") for s, r in records.items()
              if r.get("status") == "failed"}
    summary = {
        "script": Path(__file__).name,
        "index": str(INDEX_JSON.relative_to(PACKAGE_ROOT)),
        "n_requested": len(samples),
        "n_ok_total": ok,
        "n_failed_total": len(failed),
        "failed": failed,
        "crop_modes": {
            m: sum(r.get("crop_mode") == m for r in records.values())
            for m in ("haar", "centre")
        },
        "multi_face_clips": sorted(
            s for s, r in records.items() if (r.get("n_faces") or 0) > 1
        ),
        "http_requests": _STATS["requests"],
        "bytes_ranged": _STATS["bytes"],
        "free_gb_end": round(free_gb(), 2),
        "clips": records,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"\nwrote {SUMMARY_JSON}: {ok} ok / {len(failed)} failed across all "
        f"runs; this run: {_STATS['requests']} range requests, "
        f"{_STATS['bytes'] / 1e6:.0f} MB transferred, 0 bytes of video kept"
    )
    if failed:
        raise SystemExit(
            f"INCOMPLETE: {len(failed)} clips failed (see summary JSON). "
            "Embeddings/training must use only the ok clips."
        )


if __name__ == "__main__":
    main()
