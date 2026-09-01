"""Fail if committed result artefacts leak /user/ or /home/ prefixes."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".npz", ".pt", ".wav", ".mp4"}


def test_results_have_no_user_or_home_prefixes() -> None:
    assert RESULTS.is_dir(), f"missing {RESULTS}"
    leaks: list[str] = []
    for path in RESULTS.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(ROOT).as_posix()
        for i, line in enumerate(text.splitlines(), 1):
            if "/user/" in line or "/home/" in line:
                leaks.append(f"{rel}:{i}: {line.strip()[:160]}")
    assert not leaks, "absolute /user/ or /home/ path in results/\n" + "\n".join(leaks)
