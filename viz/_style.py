"""Shared plotting style, colour palette and figure-registry helpers.

Every figure module imports from here so that all generated images share one
visual language: same fonts, same colours meaning the same thing, same
title/subtitle block. Consistency is what turns a pile of plots into a
teaching sequence.

Colour contract (kept identical across every figure):
    INK      dark navy   -> observed data / ground truth
    MODEL    crimson     -> the thing the model predicts
    ERROR    red         -> residuals, error, "what is left over"
    ALT      steel blue  -> a competing / comparison model
    ACCENT   amber       -> the highlighted answer, the point to look at
    GOOD     teal        -> the correct or recommended choice
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# palette
# --------------------------------------------------------------------------
INK = "#22314E"
MODEL = "#C1272D"
ERROR = "#E4572E"
ALT = "#3E7CB1"
ACCENT = "#E9A13B"
GOOD = "#2A9D8F"
MUTED = "#9AA5B1"
GRID = "#E3E7EC"
PANEL = "#FBFCFD"

SEQ = [INK, MODEL, ALT, GOOD, ACCENT, "#8E6C99", "#7F8C8D"]

# --------------------------------------------------------------------------
# global rcParams
# --------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#C7CDD4",
    "axes.linewidth": 1.0,
    "axes.labelcolor": "#3A4654",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlecolor": "#22314E",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.9,
    "xtick.color": "#5A6672",
    "ytick.color": "#5A6672",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#D4D9DF",
    "legend.fontsize": 10,
    "font.size": 11,
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def annotate_box(ax, x, y, text, *, color=INK, fontsize=10.5, ha="left", va="top", alpha=0.94):
    """Text in a soft rounded box - used for the 'read this' callouts."""
    return ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va,
        fontsize=fontsize, color=color, linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.55", facecolor=PANEL,
                  edgecolor="#D4D9DF", alpha=alpha),
    )


def titled(fig, title: str, subtitle: str = "", *, top: float = 0.9) -> None:
    """Attach a left-aligned title + grey explanatory subtitle to a figure.

    Call this LAST: it runs tight_layout with a reserved strip at the top, then
    writes the heading into that strip.
    """
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.text(0.012, 0.995, title, ha="left", va="top",
             fontsize=15.5, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.012, 0.952, subtitle, ha="left", va="top",
                 fontsize=10.8, color="#63707E")


# --------------------------------------------------------------------------
# figure registry
# --------------------------------------------------------------------------
@dataclass
class Figure:
    number: str
    slug: str
    title: str
    caption: str
    section: str
    func: Callable


@dataclass
class Registry:
    figures: list[Figure] = field(default_factory=list)

    def register(self, number: str, slug: str, title: str, caption: str, section: str):
        def wrap(func: Callable) -> Callable:
            self.figures.append(Figure(number, slug, title, caption, section, func))
            return func
        return wrap

    def run(self, out_dir: Path, only: str | None = None) -> list[dict]:
        """Execute every registered figure function and write its PNG."""
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[dict] = []
        for spec in sorted(self.figures, key=lambda f: f.number):
            if only and only not in (spec.number, spec.slug):
                continue
            name = f"{spec.number}_{spec.slug}.png"
            path = out_dir / name
            fig = spec.func()
            fig.savefig(path)
            plt.close(fig)
            print(f"  [{spec.number}] {name}")
            written.append({
                "number": spec.number, "slug": spec.slug, "file": name,
                "title": spec.title, "caption": spec.caption, "section": spec.section,
            })
        return written


def write_manifest(path: Path, entries: list[dict]) -> None:
    """Merge new entries into the manifest that index.html is built from."""
    existing: dict[str, dict] = {}
    if path.exists():
        for item in json.loads(path.read_text()):
            existing[item["file"]] = item
    for item in entries:
        existing[item["file"]] = item
    ordered = sorted(existing.values(), key=lambda e: (e["section"], e["number"]))
    path.write_text(json.dumps(ordered, indent=2))
