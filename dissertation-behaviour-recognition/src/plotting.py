"""Publication figures: 300 DPI PNG + JPG, no silent overwrite, no invented numbers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

DPI = 300
JPEG_QUALITY = 95

# Readable, colourblind-safer defaults (no decorative 3D).
mpl.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.constrained_layout.use": True,
    }
)


@dataclass
class FigureLog:
    generated: list[tuple[str, str]] = field(default_factory=list)  # stem, source
    skipped: list[tuple[str, str]] = field(default_factory=list)  # name, reason

    def skip(self, name: str, reason: str) -> None:
        self.skipped.append((name, reason))
        print(f"SKIP  {name}\n      {reason}")

    def ok(self, stem: Path, source: str) -> None:
        self.generated.append((str(stem), source))
        print(f"SAVE  {stem}.png + .jpg")


def save_publication_figure(
    fig: Figure,
    output_stem: Path,
    log: FigureLog | None = None,
    source: str = "",
    force: bool = False,
) -> bool:
    """Write <stem>.png and <stem>.jpg at 300 DPI. Skip if both exist unless force=True."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    jpg = output_stem.with_suffix(".jpg")
    if png.exists() and jpg.exists() and not force:
        if log:
            log.skip(str(output_stem), "already exists (pass --force to overwrite)")
        else:
            print(f"SKIP  {output_stem} (exists)")
        plt.close(fig)
        return False
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(
        jpg,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": JPEG_QUALITY},
    )
    plt.close(fig)
    if log:
        log.ok(output_stem, source)
    return True


def require_files(log: FigureLog, name: str, *paths: Path) -> bool:
    missing = [str(p) for p in paths if not p.exists() or p.stat().st_size < 8]
    if missing:
        log.skip(name, "missing " + ", ".join(missing))
        return False
    return True
