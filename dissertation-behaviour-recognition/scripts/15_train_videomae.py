#!/usr/bin/env python3
"""Planned experiment, not run: VideoMAE on the 30 gold windows.

Status (20 August 2026): no video pixels exist in this repository and no
VideoMAE training or evaluation was performed. The submitted study is
pose-only; the verified TEST numbers come from the pose rule and the 1D CNN
(results/rule_test_metrics.json, results/classifier_test_metrics.json).

Why it was not run: the lab account used for pose extraction has a ~25 GB
quota with ~6.5 GB free after a CPU PyTorch install, which cannot hold
RealTalk video shards plus a VideoMAE checkpoint. EMOCA pose was streamed
from the official archive without saving it.

Plan when storage allows (details and verification commands:
reports/videomae_preflight_lab.md):
  - model: MCG-NJU/videomae-base, 16-frame frozen encoder (configs/videomae_frozen.yaml),
  - input: 2 s RGB windows around the annotated gold windows,
  - protocol unchanged: tune on the 15 DEV windows, score the 15 TEST windows once.

This script intentionally performs no computation and downloads nothing.
"""
from __future__ import annotations

PREFLIGHT = "reports/videomae_preflight_lab.md"


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
