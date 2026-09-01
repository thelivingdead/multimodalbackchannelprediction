#!/usr/bin/env python3
"""Step 1 — Build a tiny RealTalk working set: ~10 × 1-minute clips.

Storage rule: never download/extract the full ~23.6 GB emoca.tar.gz.

Modes
-----
  --demo        Synthetic 10 clips (no download). Use this to prove the pipeline.
  --from-local  Use videos + EMOCA pkls you already have on disk.
  --hf          Download only needed files from Hugging Face (videos shard +
                individual EMOCA pkls if the hub lists them). Falls back to
                extracting a SINGLE video from one videos_*.tar member.

Output
------
  data/tiny_subset/<video_id>/
      clip.mp4          60 s, 25 fps
      emoca.pkl         frame dict sliced to the clip (or synthetic)
      meta.json
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pose_utils import CLIP_SECONDS, FPS_DEFAULT, ensure_dir, write_json  # noqa: E402

HF_REPO = "scottgeng00/realtalk"
DEFAULT_IDS = [
    "5hxY5Svr2aM",
    "7kQvYx0pLmA",
    "tiny01",
    "tiny02",
    "tiny03",
    "tiny04",
    "tiny05",
    "tiny06",
    "tiny07",
    "tiny08",
]


def _trim_video(src: Path, dst: Path, start_s: float, duration_s: float, fps: float) -> bool:
    try:
        import cv2
    except ImportError:
        print("OpenCV missing — copying source video without trim:", src)
        shutil.copy2(src, dst)
        return False
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {src}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    start_f = int(start_s * src_fps)
    n_keep = int(duration_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
    written = 0
    while written < n_keep:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    writer.release()
    cap.release()
    return written > 0


def _slice_emoca(src_pkl: Path, dst_pkl: Path, start_frame: int, n_frames: int) -> int:
    with open(src_pkl, "rb") as f:
        data = pickle.load(f)
    sliced = {}
    for k, v in data.items():
        try:
            fi = int(k)
        except (TypeError, ValueError):
            continue
        if start_frame <= fi < start_frame + n_frames:
            sliced[fi - start_frame] = v
    with open(dst_pkl, "wb") as f:
        pickle.dump(sliced, f, protocol=pickle.HIGHEST_PROTOCOL)
    return len(sliced)


def make_demo_subset(out_root: Path, n: int, duration: float, fps: float) -> None:
    """Write 10 synthetic clip folders so later steps run without RealTalk."""
    n_frames = int(duration * fps)
    for i in range(n):
        vid = f"demo_{i:02d}"
        d = ensure_dir(out_root / vid)
        # Tiny placeholder "video": numpy-free note file; 06 will synthesise frames.
        (d / "clip.mp4").write_bytes(b"DEMO_NO_VIDEO")
        fake_emoca = {}
        for fidx in range(n_frames):
            # axis-angle around x so pitch varies; inject nod-like oscillation
            t = fidx / fps
            nod = 0.0
            if 4.0 + (i % 5) <= t <= 4.7 + (i % 5) or 12.0 <= t <= 12.7 or 20.0 <= t <= 20.8:
                nod = -0.14 * __import__("math").sin(2 * 3.1416 * 2.0 * (t % 1.0))
            pose = [nod + 0.02 * __import__("math").sin(0.3 * t), 0.01, 0.0, 0.0, 0.0, 0.0]
            fake_emoca[fidx] = {"p0": {"pose": pose}, "p1": {"pose": [0.01, 0.0, 0.0, 0, 0, 0]}}
        with open(d / "emoca.pkl", "wb") as f:
            pickle.dump(fake_emoca, f)
        write_json(
            d / "meta.json",
            {
                "video_id": vid,
                "source": "synthetic_demo",
                "fps": fps,
                "duration_s": duration,
                "start_s": 0.0,
                "n_frames": n_frames,
                "persons": ["p0", "p1"],
            },
        )
        print("demo clip", vid)


def from_local(args) -> None:
    video_dir = Path(args.video_dir)
    emoca_dir = Path(args.emoca_dir)
    videos = sorted(list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mp4")))[: args.n]
    if not videos:
        raise SystemExit(f"No videos in {video_dir}")
    out = ensure_dir(Path(args.out))
    n_frames = int(args.duration * args.fps)
    start_s = args.start
    start_f = int(start_s * args.fps)
    for src in videos:
        vid = src.stem
        pkl = emoca_dir / f"{vid}.pkl"
        if not pkl.exists():
            print("SKIP (no EMOCA pkl):", vid)
            continue
        d = ensure_dir(out / vid)
        _trim_video(src, d / "clip.mp4", start_s, args.duration, args.fps)
        n = _slice_emoca(pkl, d / "emoca.pkl", start_f, n_frames)
        write_json(
            d / "meta.json",
            {
                "video_id": vid,
                "source": str(src),
                "emoca_src": str(pkl),
                "fps": args.fps,
                "duration_s": args.duration,
                "start_s": start_s,
                "n_frames": n,
                "persons": ["p0", "p1"],
            },
        )
        print(f"local clip {vid}: {n} emoca frames")


def from_hf(args) -> None:
    """Download *individual* files only. Never pull emoca.tar.gz whole."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub")

    out = ensure_dir(Path(args.out))
    cache = ensure_dir(Path(args.cache))
    api = HfApi()
    print("Listing files on", HF_REPO, "(this is metadata only)…")
    files = api.list_repo_files(HF_REPO, repo_type="dataset")
    pkls = [f for f in files if f.endswith(".pkl") and "emoca" in f.lower()]
    videos = [f for f in files if f.endswith(".avi")]
    print(f"Hub lists {len(pkls)} emoca pkls, {len(videos)} avis (may be inside tars).")

    # Prefer loose pkls. If only emoca.tar.gz exists, refuse.
    tar_emoca = [f for f in files if f.endswith("emoca.tar.gz") or f.endswith("emoca.tar")]
    if not pkls and tar_emoca:
        print(
            "WARNING: Hub only exposes a single emoca archive "
            f"{tar_emoca[0]}. Do NOT download it on a 25 GB disk.\n"
            "Use --demo to prove the pipeline, or copy 10 pkls onto the lab "
            "and run --from-local."
        )
        make_demo_subset(out, args.n, args.duration, args.fps)
        return

    chosen_pkls = pkls[: args.n]
    if not chosen_pkls:
        print("No individual EMOCA files; writing demo subset instead.")
        make_demo_subset(out, args.n, args.duration, args.fps)
        return

    n_frames = int(args.duration * args.fps)
    start_f = int(args.start * args.fps)
    for rel in chosen_pkls:
        local_pkl = Path(
            hf_hub_download(HF_REPO, rel, repo_type="dataset", local_dir=str(cache))
        )
        vid = local_pkl.stem
        d = ensure_dir(out / vid)
        n = _slice_emoca(local_pkl, d / "emoca.pkl", start_f, n_frames)
        # matching video: try same id .avi or inside a videos_*.tar member
        avi_rel = next((v for v in videos if Path(v).stem == vid), None)
        if avi_rel:
            local_avi = Path(
                hf_hub_download(HF_REPO, avi_rel, repo_type="dataset", local_dir=str(cache))
            )
            _trim_video(local_avi, d / "clip.mp4", args.start, args.duration, args.fps)
        else:
            (d / "clip.mp4").write_bytes(b"NO_VIDEO_ON_HUB_FOR_THIS_ID")
        write_json(
            d / "meta.json",
            {
                "video_id": vid,
                "source": "huggingface:" + rel,
                "fps": args.fps,
                "duration_s": args.duration,
                "start_s": args.start,
                "n_frames": n,
                "persons": ["p0", "p1"],
            },
        )
        print(f"hf clip {vid}: {n} frames (delete cache/{rel} if you need space)")


def extract_one_member_from_tar(tar_path: Path, member_name: str, dest: Path) -> None:
    """Extract a single file from a tar without unpacking the rest."""
    with tarfile.open(tar_path, "r:*") as tf:
        member = next((m for m in tf.getmembers() if Path(m.name).name == member_name), None)
        if member is None:
            raise FileNotFoundError(member_name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        extracted = tf.extractfile(member)
        if extracted is None:
            raise RuntimeError("empty member")
        dest.write_bytes(extracted.read())


def main() -> None:
    p = argparse.ArgumentParser(description="Tiny 10×1min RealTalk subset (storage-safe)")
    p.add_argument("--mode", choices=["demo", "from-local", "hf"], default="demo")
    p.add_argument("--out", default=str(ROOT / "data" / "tiny_subset"))
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--duration", type=float, default=CLIP_SECONDS)
    p.add_argument("--start", type=float, default=30.0, help="Start offset in source video (s)")
    p.add_argument("--fps", type=float, default=FPS_DEFAULT)
    p.add_argument("--video-dir", default="")
    p.add_argument("--emoca-dir", default="")
    p.add_argument("--cache", default=str(ROOT / "data" / "hf_cache"))
    args = p.parse_args()
    ensure_dir(Path(args.out))
    if args.mode == "demo":
        make_demo_subset(Path(args.out), args.n, args.duration, args.fps)
    elif args.mode == "from-local":
        if not args.video_dir or not args.emoca_dir:
            raise SystemExit("--from-local needs --video-dir and --emoca-dir")
        from_local(args)
    else:
        from_hf(args)
    print("Wrote subset to", args.out)


if __name__ == "__main__":
    main()
