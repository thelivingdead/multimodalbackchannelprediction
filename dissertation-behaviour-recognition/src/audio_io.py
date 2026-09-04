"""Audio window I/O for nod DEV-only experiments (no GOLD TEST scoring).

RealTalk members are pulled from Hugging Face tar shards with HTTP Range
reads (same index as ``scripts/fetch_rgb_windows.py``). Temporary videos
are deleted after the watch-window WAV / features are written.

Audio is the **container soundtrack** (both participants), not a
listener-only track. Pose and RGB remain visual encodings of the same
camera stream; this module is the separate auditory stream.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_VENDOR = Path(__file__).resolve().parents[1] / ".vendor"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

import numpy as np
import pandas as pd
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy import signal

from .paths import ROOT
from .utils import parse_clock

FPS = 25.0
FEATURE_DIM = 30  # 13 MFCC mean + 13 std + RMS mean/std + centroid mean/std
N_MFCC = 13
TARGET_SR = 16000
SHARD_URL = (
    "https://huggingface.co/datasets/scottgeng00/realtalk"
    "/resolve/main/videos/{}"
)
INDEX_JSON = ROOT / "results" / "video_shard_index.json"
GOLD_CSV = ROOT / "data" / "gold_annotations.csv"
GOLD_SHEET = ROOT / "data" / "gold" / "annotation_sheet.csv"
PSEUDO_LABELS = ROOT / "results" / "pseudo_labels.csv"
GOLD_NPZ = ROOT / "features" / "gold"
PSEUDO_NPZ = ROOT / "features" / "pseudo"
EMB_DIR = ROOT / "data" / "features" / "videomae"
WAV_DIR = ROOT / "data" / "audio_alignment_check"
AUDIO_FEAT_DIR = ROOT / "data" / "features" / "audio"
LOCKED_RESULT_DIRS = (
    ROOT / "results" / "videomae_finetuned",
    ROOT / "results" / "videomae_finetuned_n200",
    ROOT / "results" / "videomae_finetuned_n120",
    ROOT / "results" / "videomae_frozen_head",
    ROOT / "results" / "shake",
    ROOT / "results" / "joint",
)
DEFAULT_ALIGN_IDS = (
    "gold_012",
    "gold_013",
    "gold_010",
    "gold_014",
    "gold_009",
)
SILENCE_RMS = 1e-4
SILENCE_PEAK = 1e-3
DUR_TOL_S = 0.50
AV_DUR_TOL_S = 0.50
FPS_TOL = 0.15
FETCH_TIMEOUT_S = 900
WRITE_CHUNK = 1 << 20

_DURATION_RE = re.compile(
    r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", re.IGNORECASE
)
_START_RE = re.compile(r"start:\s*([0-9.]+)", re.IGNORECASE)
_HZ_RE = re.compile(r"(\d+)\s*Hz", re.IGNORECASE)
_FPS_RE = re.compile(r"([0-9.]+)\s*fps", re.IGNORECASE)
_TBR_RE = re.compile(r"([0-9.]+)\s*tbr", re.IGNORECASE)
_WH_RE = re.compile(r"(\d{2,5})x(\d{2,5})")
_CHAN_RE = re.compile(r"\b(mono|stereo|5\.1|2\s*channels|1\s*channels)\b", re.I)


def refuse_test_scoring(*, score_test: bool = False, split: str | None = None) -> None:
    """Abort if the caller asked to score GOLD TEST."""
    if score_test:
        raise SystemExit(
            "STOP: DEV-only audio script refuses --score-test. "
            "GOLD TEST is locked and will not be scored for audio or fusion."
        )
    if split is None:
        return
    token = str(split).strip().lower().replace("-", "_")
    if token in {"test", "gold_test", "holdout", "hold_out"}:
        raise SystemExit(
            f"STOP: split={split!r} is GOLD TEST. Audio/fusion scripts "
            "are DEV-only and will not load or score TEST windows."
        )


def assert_not_locked_out_dir(out_dir: Path) -> Path:
    out_dir = Path(out_dir).resolve()
    for blocked in LOCKED_RESULT_DIRS:
        blocked_r = blocked.resolve()
        try:
            out_dir.relative_to(blocked_r)
        except ValueError:
            continue
        raise SystemExit(
            f"STOP: refusing to write locked dir {blocked_r}. "
            "Do not --force locked VideoMAE / shake TEST artefacts. "
            "Use results/audio_dev/ or another new folder."
        )
    return out_dir


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise SystemExit(
            "STOP: imageio-ffmpeg is not installed. "
            "pip install imageio-ffmpeg  (or use the otter AUDIO_DEV.md venv)."
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_shard_index() -> dict[str, dict]:
    if not INDEX_JSON.exists():
        raise SystemExit(
            f"STOP: missing {INDEX_JSON}. Need scripts/build_video_shard_index.py."
        )
    return json.loads(INDEX_JSON.read_text())


def _scalar(value: Any) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
    else:
        item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode()
    return str(item)


def pose_meta(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as z:
        frames = np.asarray(z["frames"])
        start = int(frames.min())
        end_incl = int(frames.max())
        n = int(frames.size)
        return {
            "sample_id": _scalar(z["sample_id"]),
            "video_id": _scalar(z["video_id"]),
            "person": _scalar(z["person"]),
            "start_frame": start,
            "end_frame_inclusive": end_incl,
            "n_frames": n,
            "t0_s": start / FPS,
            "t1_s": (end_incl + 1) / FPS,
            "duration_s": n / FPS,
        }


def _sheet_by_video() -> dict[str, dict[str, Any]]:
    if not GOLD_SHEET.exists():
        return {}
    rows = {}
    for rec in pd.read_csv(GOLD_SHEET).to_dict(orient="records"):
        rows[str(rec["video_id"])] = rec
    return rows


def inventory_clip(sample_id: str) -> dict[str, Any]:
    """Gold or pseudo window metadata from committed pose npz + gold CSV."""
    sample_id = str(sample_id)
    gold_path = GOLD_NPZ / f"{sample_id}.npz"
    pseudo_path = PSEUDO_NPZ / f"{sample_id}.npz"
    if gold_path.exists():
        meta = pose_meta(gold_path)
        meta["origin"] = "gold"
        gold = pd.read_csv(GOLD_CSV)
        row = gold.loc[gold["sample_id"].astype(str) == sample_id]
        if row.empty:
            raise SystemExit(f"STOP: {sample_id} has pose npz but no gold CSV row.")
        rec = row.iloc[0]
        meta["split"] = str(rec["split"]).upper()
        meta["label"] = int(rec["label"])
        sheet = _sheet_by_video().get(meta["video_id"], {})
        meta["who_to_watch"] = str(sheet.get("who_to_watch", rec.get("person", "")))
        meta["watch_from"] = sheet.get("watch_from", "")
        meta["watch_until"] = sheet.get("watch_until", "")
        meta["nod_start"] = sheet.get("nod_start", "")
        meta["nod_end"] = sheet.get("nod_end", "")
        meta["youtube_url"] = sheet.get("youtube_url", "")
        side = "LEFT" if meta["person"] == "p0" else "RIGHT"
        meta["speaker_side"] = side
        who = str(meta["who_to_watch"]).upper()
        meta["side_matches_sheet"] = side in who if who else None
        wf = parse_clock(meta["watch_from"])
        wu = parse_clock(meta["watch_until"])
        meta["watch_from_s"] = wf
        meta["watch_until_s"] = wu
        meta["watch_matches_frames"] = None
        if wf is not None and wu is not None:
            meta["watch_matches_frames"] = (
                abs(wf - meta["t0_s"]) <= 1.0 and abs(wu - meta["t1_s"]) <= 1.0
            )
        return meta
    if pseudo_path.exists():
        meta = pose_meta(pseudo_path)
        meta["origin"] = "pseudo"
        meta["split"] = "TRAIN"
        pl = pd.read_csv(PSEUDO_LABELS)
        hit = pl.loc[pl["sample_id"].astype(str) == sample_id]
        if hit.empty:
            raise SystemExit(f"STOP: {sample_id} missing from {PSEUDO_LABELS}.")
        meta["label"] = int(hit.iloc[0]["pseudo_label"])
        meta["label_kind"] = "pseudo"
        return meta
    raise SystemExit(
        f"STOP: no pose npz for {sample_id} under features/gold or features/pseudo."
    )


def nod_train_dev_ids() -> tuple[list[str], list[str]]:
    """TRAIN = 80 pseudo; DEV = 15 gold DEV. TEST ids are never returned."""
    pseudo = pd.read_csv(PSEUDO_LABELS)
    train_ids = [str(s) for s in pseudo["sample_id"].tolist()]
    gold = pd.read_csv(GOLD_CSV)
    gold["split"] = gold["split"].astype(str).str.upper()
    tes = gold.loc[gold["split"] == "TEST", "sample_id"].astype(str)
    if tes.empty:
        raise SystemExit("STOP: gold CSV has no TEST rows to exclude; check the sheet.")
    dev = gold.loc[gold["split"] == "DEV"].sort_values("sample_id")
    dev_ids = [str(s) for s in dev["sample_id"].tolist()]
    overlap = set(train_ids) & set(dev_ids)
    if overlap:
        raise SystemExit(f"STOP: TRAIN/DEV id overlap {sorted(overlap)}")
    test_ids = set(tes.tolist())
    leaked = [s for s in train_ids + dev_ids if s in test_ids]
    if leaked:
        raise SystemExit(f"STOP: TEST ids leaked into TRAIN/DEV list: {leaked}")
    return train_ids, dev_ids


def local_video_path(video_id: str) -> Path | None:
    roots: list[Path] = []
    env = os.environ.get("REALTALK_VIDEO_DIR", "").strip()
    if env:
        roots.append(Path(env))
    roots.extend(
        [
            ROOT / "data" / "working" / "videos",
            Path("/scratch/db01550/realtalk_videos"),
            Path.home() / "realtalk_videos",
        ]
    )
    names = (f"{video_id}.mp4", f"{video_id}.mkv", f"{video_id}.webm", f"{video_id}.avi")
    for root in roots:
        if not root.is_dir():
            continue
        for name in names:
            path = root / name
            if path.is_file() and path.stat().st_size > 0:
                return path
    return None


def _hf_headers() -> dict[str, str]:
    headers = {"User-Agent": "dissertation-behaviour-recognition-audio-check"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"
    return headers


def fetch_member_to_file(video_id: str, dest: Path, index: dict[str, dict]) -> dict[str, Any]:
    """Download one tar member (the source video) via HTTP Range."""
    if video_id not in index:
        raise SystemExit(f"STOP: {video_id} missing from video_shard_index.json.")
    info = index[video_id]
    shard = info["shard"]
    offset = int(info["offset"])
    size = int(info["size"])
    url = SHARD_URL.format(shard)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = _hf_headers()
    headers["Range"] = f"bytes={offset}-{offset + size - 1}"
    try:
        import requests
    except ImportError as exc:
        raise SystemExit("STOP: pip install requests") from exc

    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            with requests.get(
                url, headers=headers, timeout=FETCH_TIMEOUT_S, stream=True
            ) as resp:
                if resp.status_code != 206:
                    meaning = {
                        200: "server ignored Range (would pull the whole shard)",
                        401: "auth required",
                        403: "gated / forbidden",
                    }.get(resp.status_code, "unexpected")
                    raise SystemExit(
                        f"STOP: range read {url} {offset}-{offset + size - 1} "
                        f"HTTP {resp.status_code} ({meaning})."
                    )
                written = 0
                with tmp.open("wb") as handle:
                    for chunk in resp.iter_content(WRITE_CHUNK):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        written += len(chunk)
            if written != size:
                raise SystemExit(
                    f"STOP: {video_id} range read wrote {written} bytes, index says {size}."
                )
            tmp.replace(dest)
            return {
                "source": "huggingface_range",
                "url": url,
                "shard": shard,
                "offset": offset,
                "size": size,
                "path": str(dest),
            }
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 — retry, then curl fallback
            last_exc = exc
            if tmp.exists():
                tmp.unlink()
            if attempt == 2:
                break
            time.sleep(5)
    return _curl_range(url, dest, tmp, offset, size, video_id, shard, last_exc)


def _curl_range(
    url: str,
    dest: Path,
    tmp: Path,
    offset: int,
    size: int,
    video_id: str,
    shard: str,
    last_exc: Exception | None,
) -> dict[str, Any]:
    """Fallback when Python requests is proxied but curl is not."""
    curl = shutil.which("curl")
    if not curl:
        raise SystemExit(
            f"STOP: network error fetching {video_id} from {url}: {last_exc}. "
            "curl is also missing."
        )
    cmd = [
        curl, "-fL", "--retry", "2", "--retry-delay", "5",
        "-H", f"Range: bytes={offset}-{offset + size - 1}",
        "-o", str(tmp),
        url,
    ]
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        cmd[4:4] = ["-H", f"Authorization: Bearer {token.strip()}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not tmp.exists():
        raise SystemExit(
            f"STOP: curl range read failed for {video_id} ({url}): "
            f"exit {proc.returncode} {(proc.stderr or '')[-400:]}. "
            f"requests error was: {last_exc}"
        )
    written = tmp.stat().st_size
    if written != size:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"STOP: curl wrote {written} bytes for {video_id}, index says {size}."
        )
    tmp.replace(dest)
    return {
        "source": "huggingface_range_curl",
        "url": url,
        "shard": shard,
        "offset": offset,
        "size": size,
        "path": str(dest),
    }


def resolve_video_file(
    video_id: str,
    *,
    index: dict[str, dict],
    tmp_dir: Path,
    keep: bool = False,
) -> tuple[Path, dict[str, Any], bool]:
    """Return (path, provenance, delete_when_done)."""
    local = local_video_path(video_id)
    if local is not None:
        return local, {"source": "local", "path": str(local)}, False
    size = int(index[video_id]["size"])
    free = shutil.disk_usage(Path.home()).free
    need = size + 512 * 1024 * 1024
    if free < need:
        raise SystemExit(
            f"BLOCKED: {video_id} member is {size / 1024**2:.0f} MB but free "
            f"disk is {free / 1024**3:.2f} GB (need member + 0.5 GB). "
            "Run on otter with REALTALK_VIDEO_DIR or more free space. "
            "No F1 invented."
        )
    dest = tmp_dir / f"{video_id}.mp4"
    prov = fetch_member_to_file(video_id, dest, index)
    return dest, prov, not keep


def probe_media(path: Path) -> dict[str, Any]:
    exe = ffmpeg_exe()
    proc = subprocess.run(
        [exe, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    duration_s = None
    start_s = 0.0
    dur_m = _DURATION_RE.search(text)
    if dur_m:
        duration_s = (
            int(dur_m.group(1)) * 3600
            + int(dur_m.group(2)) * 60
            + float(dur_m.group(3))
        )
    st_m = _START_RE.search(text)
    if st_m:
        start_s = float(st_m.group(1))
    video_line = next(
        (ln.strip() for ln in text.splitlines() if "Video:" in ln and "Stream" in ln),
        "",
    )
    audio_line = next(
        (ln.strip() for ln in text.splitlines() if "Audio:" in ln and "Stream" in ln),
        "",
    )
    fps = None
    if video_line:
        fm = _FPS_RE.search(video_line) or _TBR_RE.search(video_line)
        if fm:
            fps = float(fm.group(1))
    sr = None
    channels = None
    if audio_line:
        hm = _HZ_RE.search(audio_line)
        if hm:
            sr = int(hm.group(1))
        cm = _CHAN_RE.search(audio_line)
        if cm:
            token = cm.group(1).lower()
            channels = 1 if "mono" in token or token.startswith("1") else 2
    wh = _WH_RE.search(video_line) if video_line else None
    return {
        "ffmpeg_ok": bool(video_line or audio_line),
        "raw_stderr_tail": text[-2500:],
        "duration_s": duration_s,
        "container_start_s": start_s,
        "has_video": bool(video_line),
        "has_audio": bool(audio_line),
        "video_line": video_line,
        "audio_line": audio_line,
        "fps": fps,
        "sample_rate_hz": sr,
        "channels": channels,
        "width": int(wh.group(1)) if wh else None,
        "height": int(wh.group(2)) if wh else None,
    }


def extract_window_wav(
    video_path: Path,
    wav_path: Path,
    *,
    t0_s: float,
    duration_s: float,
) -> dict[str, Any]:
    exe = ffmpeg_exe()
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = wav_path.with_suffix(".part.wav")
    cmd = [
        exe, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{t0_s:.6f}",
        "-t", f"{duration_s:.6f}",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-c:a", "pcm_s16le",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 64:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"ffmpeg wav extract failed (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '')[-400:]}"
        )
    tmp.replace(wav_path)
    return {"wav_path": str(wav_path), "ffmpeg_cmd": cmd}


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    sr, y = wavfile.read(path)
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float32) / np.float32(np.iinfo(y.dtype).max)
    else:
        y = y.astype(np.float32)
    return y, int(sr)


def resample_mono(y: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if sr == target_sr:
        return y.astype(np.float32)
    n_out = int(round(len(y) * target_sr / sr))
    if n_out < 16:
        raise RuntimeError(f"resample would produce {n_out} samples")
    x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, y.astype(np.float64)).astype(np.float32)


def wav_stats(y: np.ndarray, sr: int) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
    return {
        "n_samples": int(y.size),
        "sample_rate_hz": int(sr),
        "duration_s": float(y.size / sr) if sr else 0.0,
        "peak": peak,
        "rms": rms,
        "non_empty": bool(y.size > 0 and peak > 0.0),
        "audible": bool(y.size > 0 and peak >= SILENCE_PEAK and rms >= SILENCE_RMS),
    }


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 26) -> np.ndarray:
    def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    fmax = sr / 2.0
    mels = np.linspace(float(hz_to_mel(0.0)), float(hz_to_mel(fmax)), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, centre, right = bins[m - 1], bins[m], bins[m + 1]
        if centre > left:
            fb[m - 1, left:centre] = (np.arange(left, centre) - left) / (centre - left)
        if right > centre:
            fb[m - 1, centre:right] = (right - np.arange(centre, right)) / (right - centre)
    return fb


def _features_scipy(y: np.ndarray, sr: int) -> tuple[np.ndarray, str]:
    win = max(int(0.025 * sr), 16)
    hop = max(int(0.010 * sr), 8)
    n_fft = 512
    freqs, _, zxx = signal.stft(
        y.astype(np.float64),
        fs=sr,
        nperseg=win,
        noverlap=win - hop,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    mag = np.abs(zxx)
    if mag.size == 0:
        raise RuntimeError("STFT produced no frames")
    fb = _mel_filterbank(sr, n_fft)
    n_freq = min(fb.shape[1], mag.shape[0])
    mag_n = mag[:n_freq, :]
    fb_n = fb[:, :n_freq]
    # Explicit mix: a BLAS gemm SIGFPEs in some Mac sandboxes.
    mel = np.empty((fb_n.shape[0], mag_n.shape[1]), dtype=np.float64)
    for i in range(fb_n.shape[0]):
        mel[i] = np.sum(fb_n[i, :, None] * mag_n, axis=0)
    logmel = np.log(np.maximum(mel, 1e-10))
    mfcc = dct(logmel, type=2, axis=0, norm="ortho")[:N_MFCC]
    centroid = np.sum(freqs[:n_freq, None] * mag[:n_freq, :], axis=0) / np.maximum(
        mag[:n_freq, :].sum(axis=0), 1e-10
    )
    rms_frames = []
    for start in range(0, max(len(y) - win, 0) + 1, hop):
        frame = y[start : start + win]
        if frame.size < win:
            break
        rms_frames.append(float(np.sqrt(np.mean(np.square(frame)))))
    rms_arr = np.asarray(rms_frames, dtype=np.float64)
    if rms_arr.size == 0:
        rms_arr = np.asarray([float(np.sqrt(np.mean(np.square(y))))], dtype=np.float64)
    feat = np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            np.array([rms_arr.mean(), rms_arr.std(ddof=0)]),
            np.array([centroid.mean(), centroid.std(ddof=0)]),
        ]
    ).astype(np.float32)
    if feat.shape != (FEATURE_DIM,):
        raise RuntimeError(f"expected {FEATURE_DIM}-D features, got {feat.shape}")
    return feat, "scipy_stft_mfcc"


def _features_librosa(y: np.ndarray, sr: int) -> tuple[np.ndarray, str]:
    import librosa

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    rms = librosa.feature.rms(y=y)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    feat = np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            np.array([float(rms.mean()), float(rms.std())]),
            np.array([float(cent.mean()), float(cent.std())]),
        ]
    ).astype(np.float32)
    if feat.shape != (FEATURE_DIM,):
        raise RuntimeError(f"librosa features {feat.shape}, expected {FEATURE_DIM}")
    return feat, "librosa"


def extract_audio_features(
    y: np.ndarray, sr: int, *, prefer_librosa: bool = True
) -> tuple[np.ndarray, dict[str, Any]]:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size < sr // 4:
        raise RuntimeError(f"audio too short: {y.size} samples at {sr} Hz")
    backend = "scipy_stft_mfcc"
    if prefer_librosa:
        try:
            feat, backend = _features_librosa(y, sr)
        except Exception:
            feat, backend = _features_scipy(y, sr)
    else:
        feat, backend = _features_scipy(y, sr)
    stats = wav_stats(y, sr)
    stats["backend"] = backend
    stats["feature_dim"] = int(feat.size)
    return feat, stats


def feature_path(sample_id: str) -> Path:
    return AUDIO_FEAT_DIR / f"{sample_id}.npz"


def save_features(sample_id: str, feat: np.ndarray, extra: dict[str, Any]) -> Path:
    AUDIO_FEAT_DIR.mkdir(parents=True, exist_ok=True)
    path = feature_path(sample_id)
    payload = {"features": np.asarray(feat, dtype=np.float32), "sample_id": sample_id}
    for key, value in extra.items():
        if isinstance(value, (int, float, np.floating, np.integer)):
            payload[key] = np.asarray(value)
        elif isinstance(value, str):
            payload[key] = np.asarray(value)
    np.savez(path, **payload)
    return path


def load_features(sample_id: str) -> np.ndarray | None:
    path = feature_path(sample_id)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        return np.asarray(z["features"], dtype=np.float32).reshape(-1)


def load_videomae_embedding(sample_id: str) -> np.ndarray | None:
    path = EMB_DIR / f"{sample_id}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        key = "embedding" if "embedding" in z.files else z.files[0]
        return np.asarray(z[key], dtype=np.float32).reshape(-1)


def alignment_checks(meta: dict[str, Any], probe: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    fps = probe.get("fps")
    fps_ok = fps is not None and abs(float(fps) - FPS) <= FPS_TOL
    if not probe.get("has_video"):
        reasons.append("no video stream")
    if not probe.get("has_audio"):
        reasons.append("no audio stream")
    if not fps_ok:
        reasons.append(f"fps={fps} (expected ~{FPS})")
    if probe.get("sample_rate_hz") in (None, 0):
        reasons.append("missing sample rate")
    if not stats.get("non_empty"):
        reasons.append("empty wav")
    if not stats.get("audible"):
        reasons.append(
            f"not audible (peak={stats.get('peak')} rms={stats.get('rms')})"
        )
    expected = float(meta["duration_s"])
    got = float(stats.get("duration_s") or 0.0)
    dur_ok = abs(got - expected) <= DUR_TOL_S
    if not dur_ok:
        reasons.append(f"wav duration {got:.3f}s vs window {expected:.3f}s")
    src_dur = probe.get("duration_s")
    window_fits = True
    if src_dur is not None:
        window_fits = float(src_dur) + 0.25 >= float(meta["t1_s"])
        if not window_fits:
            reasons.append(
                f"source duration {src_dur:.2f}s < window end {meta['t1_s']:.2f}s"
            )
    side_ok = meta.get("side_matches_sheet")
    if side_ok is False:
        reasons.append("person p0/p1 does not match who_to_watch")
    watch_ok = meta.get("watch_matches_frames")
    if watch_ok is False:
        reasons.append("watch_from/until does not match pose frame window")
    av_ok = bool(probe.get("has_audio") and probe.get("has_video"))
    return {
        "pass": not reasons,
        "reasons": reasons,
        "fps_ok": bool(fps_ok),
        "duration_ok": bool(dur_ok),
        "window_fits_source": bool(window_fits),
        "av_streams_present": av_ok,
        "side_ok": side_ok,
        "watch_ok": watch_ok,
    }
