#!/usr/bin/env python3
"""VideoMAE Step 4 (otter48): frozen VideoMAE embeddings of the RGB windows.

Loads ``features/rgb16/<sample_id>.npz`` (Step 3 output, uint8
(16, 224, 224, 3) face crops) and runs a **frozen** ``MCG-NJU/videomae-*``
encoder on CPU — no fine-tuning, no gradients. Each clip is pooled to one
embedding (mean over patch tokens of ``last_hidden_state``) and saved to
``data/features/videomae/<sample_id>.npz``; run provenance (checkpoint,
versions, per-clip status) goes to the commitable
``results/videomae_embeddings_meta.json``.

Checkpoint choice is measured, not assumed
------------------------------------------
``--checkpoint auto`` (default) issues HTTP HEAD requests against the
checkpoint files and compares their sizes with the disk budget
(``free - MIN_FREE_GB``):

* ``MCG-NJU/videomae-base`` (~0.4 GB, embed dim 768) is used if it fits.
* Otherwise ``MCG-NJU/videomae-small`` (~0.2 GB, embed dim 384) is tried.
* If neither fits inside the budget, or the files cannot be reached, the
  script exits with a clear ``BLOCKED`` message and downloads nothing.

Hard rules
----------
* Free space on ``~`` must stay above ``MIN_FREE_GB`` (5.4 GB), checked before
  and after the checkpoint download and after every clip.
* The backbone is never trained: ``torch.no_grad()``, ``model.eval()``.
* HF caches are pinned to ``.hf_cache/`` inside the repo (gitignored) so
  nothing lands outside the quota-visible tree.

Lab invocation (after Step 3 outputs exist)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    source ../.venv/bin/activate
    pip install "transformers" "safetensors"
    python scripts/extract_videomae_embeddings.py --limit 2   # smoke
    python scripts/extract_videomae_embeddings.py             # full
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import requests

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RGB16_DIR = PACKAGE_ROOT / "features" / "rgb16"
OUT_DIR = PACKAGE_ROOT / "data" / "features" / "videomae"
META_JSON = PACKAGE_ROOT / "results" / "videomae_embeddings_meta.json"

# Pin HF caches inside the repo BEFORE transformers is imported (lazy import
# happens in main()); .hf_cache/ is gitignored.
os.environ.setdefault("HF_HOME", str(PACKAGE_ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

CANDIDATES = {
    "MCG-NJU/videomae-base": {"embed_dim": 768},
    "MCG-NJU/videomae-small": {"embed_dim": 384},
}
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")  # transformers' order
MIN_FREE_GB = 5.4
TIMEOUT_S = 60


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


def probe_weight_size(repo_id: str) -> tuple[str, int] | None:
    """(filename, size_bytes) of the checkpoint file transformers would
    download, via HTTP HEAD. None if no known weight file is reachable."""
    for fname in WEIGHT_FILES:
        url = f"https://huggingface.co/{repo_id}/resolve/main/{fname}"
        try:
            resp = requests.head(url, allow_redirects=True, timeout=TIMEOUT_S)
        except requests.RequestException:
            return None
        if resp.status_code == 200:
            size = int(resp.headers.get("Content-Length", "0"))
            return (fname, size) if size else None
    return None


def choose_checkpoint(requested: str) -> tuple[str, int, dict]:
    """Return (repo_id, weight_bytes, probe_report) or SystemExit BLOCKED."""
    names = (
        [requested]
        if requested != "auto"
        else list(CANDIDATES)  # base first, small fallback
    )
    probe: dict[str, dict] = {}
    budget_gb = free_gb() - MIN_FREE_GB
    for name in names:
        if name not in CANDIDATES:
            raise SystemExit(
                f"STOP: unknown --checkpoint {name!r}; choices: auto, "
                + ", ".join(CANDIDATES)
            )
        found = probe_weight_size(name)
        if found is None:
            probe[name] = {"status": "unreachable"}
            print(f"  {name}: weight file not reachable via HEAD")
            continue
        fname, size = found
        probe[name] = {
            "status": "ok", "weight_file": fname,
            "weight_bytes": size, "weight_gb": round(size / 1024**3, 3),
        }
        fits = size / 1024**3 < budget_gb
        print(
            f"  {name}: {fname} is {size / 1024**3:.2f} GB; budget after "
            f"{MIN_FREE_GB} GB floor is {budget_gb:.2f} GB — "
            f"{'fits' if fits else 'TOO BIG'}"
        )
        if fits:
            return name, size, probe
    raise SystemExit(
        "BLOCKED: no VideoMAE checkpoint fits the disk budget "
        f"(free {free_gb():.2f} GB, floor {MIN_FREE_GB} GB) or none is "
        f"reachable. Probe report: {json.dumps(probe)}. Nothing was "
        "downloaded; paste this back."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", type=str, default=None)
    parser.add_argument("--checkpoint", default="auto",
                        help="auto | MCG-NJU/videomae-base | MCG-NJU/videomae-small")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    npz_paths = sorted(RGB16_DIR.glob("*.npz"))
    if args.ids:
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        unknown = sorted(wanted - {p.stem for p in npz_paths})
        if unknown:
            raise SystemExit(
                f"STOP: --ids {unknown} have no features/rgb16 npz. Run "
                "scripts/fetch_rgb_windows.py for them first."
            )
        npz_paths = [p for p in npz_paths if p.stem in wanted]
    if args.limit is not None:
        npz_paths = npz_paths[: args.limit]
    if not npz_paths:
        raise SystemExit(
            "STOP: no features/rgb16/*.npz selected. Run Step 3 "
            "(scripts/fetch_rgb_windows.py) first."
        )

    check_disk("start")
    print(f"checkpoint probe ({free_gb():.2f} GB free):")
    checkpoint, weight_bytes, probe = choose_checkpoint(args.checkpoint)
    embed_dim = CANDIDATES[checkpoint]["embed_dim"]

    try:
        import torch
        import transformers
        from transformers import VideoMAEImageProcessor, VideoMAEModel
    except ImportError as exc:
        raise SystemExit(
            "BLOCKED: transformers/torch import failed "
            f"({exc}). Install with `pip install transformers safetensors` "
            "in the existing CPU-torch venv; do not install a CUDA stack."
        ) from exc

    t0 = time.time()
    print(f"downloading/loading {checkpoint} (~{weight_bytes / 1024**3:.2f} GB)…")
    try:
        processor = VideoMAEImageProcessor.from_pretrained(checkpoint)
        model = VideoMAEModel.from_pretrained(checkpoint)
    except Exception as exc:
        raise SystemExit(
            f"BLOCKED: checkpoint {checkpoint} could not be downloaded/"
            f"loaded within quota ({exc}). Partial cache may exist under "
            ".hf_cache/ — remove it before retrying. Paste this back."
        ) from exc
    check_disk("post-download")
    model.eval()
    torch.manual_seed(42)  # eval-only; keeps any dropout-free path deterministic
    print(
        f"loaded in {time.time() - t0:.0f} s; transformers "
        f"{transformers.__version__}, torch {torch.__version__}, "
        f"{free_gb():.2f} GB free"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    if META_JSON.exists():
        records = json.loads(META_JSON.read_text()).get("clips", {})

    todo = [p for p in npz_paths if not (OUT_DIR / f"{p.stem}.npz").exists()]
    print(f"{len(npz_paths)} selected, {len(todo)} to embed (rest exist)")

    def embed_batch(paths: list[Path]) -> np.ndarray:
        clips = []
        for p in paths:
            with np.load(p, allow_pickle=True) as z:
                rgb = z["rgb"]
            if rgb.shape != (16, 224, 224, 3) or rgb.dtype != np.uint8:
                raise SystemExit(
                    f"STOP: {p.name} has rgb shape {rgb.shape} dtype "
                    f"{rgb.dtype}, expected (16, 224, 224, 3) uint8."
                )
            clips.append([rgb[i] for i in range(rgb.shape[0])])
        inputs = processor(clips, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs).last_hidden_state  # (B, tokens, dim)
        return out.mean(dim=1).cpu().numpy().astype(np.float32)

    n_done = 0
    for start in range(0, len(todo), args.batch_size):
        batch = todo[start : start + args.batch_size]
        embs = embed_batch(batch)
        for p, emb in zip(batch, embs):
            tmp = OUT_DIR / f"{p.stem}.tmp.npz"
            np.savez(
                tmp,
                embedding=emb,
                sample_id=p.stem,
                checkpoint=checkpoint,
                pooling="mean_over_patch_tokens",
            )
            tmp.rename(OUT_DIR / f"{p.stem}.npz")
            records[p.stem] = {
                "sample_id": p.stem, "status": "ok",
                "embed_dim": int(emb.shape[0]),
            }
        n_done += len(batch)
        check_disk(f"clip {n_done}/{len(todo)}")
        print(f"  embedded {n_done}/{len(todo)} ({free_gb():.2f} GB free)")

    n_ok = sum(r.get("status") == "ok" for r in records.values())
    meta = {
        "script": Path(__file__).name,
        "checkpoint": checkpoint,
        "embed_dim": embed_dim,
        "pooling": "mean over patch tokens of last_hidden_state (frozen encoder)",
        "backbone_frozen": True,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "weight_bytes": weight_bytes,
        "checkpoint_probe": probe,
        "input": "features/rgb16 uint8 (16, 224, 224, 3) face crops",
        "n_requested": len(npz_paths),
        "n_embeddings_total": n_ok,
        "free_gb_end": round(free_gb(), 2),
        "clips": records,
    }
    META_JSON.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwrote {META_JSON}: {n_ok} embeddings of dim {embed_dim} "
          f"from {checkpoint}")
    if n_ok < len(npz_paths):
        raise SystemExit(
            f"INCOMPLETE: {n_ok}/{len(npz_paths)} embeddings exist. "
            "Training must use only clips with embeddings."
        )


if __name__ == "__main__":
    main()
