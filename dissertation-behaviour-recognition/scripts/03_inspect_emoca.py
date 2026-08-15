#!/usr/bin/env python3
"""03 — Inspect a real EMOCA pickle. Do not guess keys if a file exists."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import default_pilot_dir, list_clip_dirs  # noqa: E402
from src.emoca_loader import load_pickle, summarize_structure  # noqa: E402
from src.utils import dump_json  # noqa: E402


def classify_fields(summary: dict) -> dict[str, str]:
    """pose is LIKELY from EMOCA docs; others UNCLEAR until seen in a real file."""
    return {
        "pose": "LIKELY — EMOCA typically stores 6D axis-angle (global 3 + jaw 3). Confirm in this pickle.",
        "exp / expression": "UNCLEAR until key is observed",
        "shape": "UNCLEAR until key is observed",
        "cam / translation": "UNCLEAR until key is observed",
        "confidence": "UNCLEAR until key is observed",
        "pitch/yaw/roll": "DERIVED — rotvec[:3] → scipy Rotation.as_euler('xyz', degrees=True)",
    }


def main() -> None:
    clips = list_clip_dirs(default_pilot_dir())
    pkl = None
    for c in clips:
        cand = c / "emoca.pkl"
        if cand.exists() and cand.stat().st_size > 10:
            pkl = cand
            break
    schema: dict = {
        "file": str(pkl) if pkl else None,
        "status": "inspected" if pkl else "NO_PKL_FOUND",
        "fields": classify_fields({}),
        "structure": None,
        "note": "Do not design eyebrow/lean rules until expression/translation keys are VERIFIED here.",
    }
    if pkl:
        obj = load_pickle(pkl)
        schema["python_type"] = type(obj).__name__
        schema["structure"] = summarize_structure(obj)
        if isinstance(obj, dict) and obj:
            k0 = next(iter(obj))
            rec = obj[k0]
            schema["example_frame_key"] = str(k0)
            schema["example_frame_type"] = type(rec).__name__
            if isinstance(rec, dict):
                schema["people_keys"] = [str(x) for x in rec.keys()]
                person = rec.get("p0", next(iter(rec.values())))
                if isinstance(person, dict):
                    schema["person_keys"] = [str(x) for x in person.keys()]
                    for name in person:
                        if "pose" in str(name).lower():
                            schema["fields"]["pose"] = f"VERIFIED key={name!s}"
    dump_json(ROOT / "reports" / "emoca_schema.json", schema)
    print("Wrote reports/emoca_schema.json")
    print("file", schema["file"], "status", schema["status"])


if __name__ == "__main__":
    main()
