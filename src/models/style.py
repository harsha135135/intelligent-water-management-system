"""Shared plotting identity — palette, type and page constants.

Every figure in this repository is drawn from here, so a model keeps its colour across the whole
study and a reader who has learned one chart can read the next. Extracted from the figure modules
it used to live in, which is why several of them could be retired.

Colour follows the validated categorical palette (blue / orange / aqua), assigned in fixed order
by *entity*, never by rank:

    Chronos-2  #2a78d6      NPTS  #eb6834      SeasonalNaive-24  #1baf7a

Validated with the palette checker at all-pairs on a light surface: worst CVD deltaE 9.2, worst
normal-vision deltaE 24.0, all gates pass. Aqua sits below 3:1 contrast on the light surface, so
the relief rule applies — every figure carries a legend or direct labels, and each has an
accompanying CSV.
"""

from __future__ import annotations

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"

CHRONOS = "Chronos2-ZS"
INCUMBENT = "NPTS"
REFERENCE = "SeasonalNaive"

HORIZONS = [6, 12, 24, 48, 72, 168]
HLABEL = {6: "6h", 12: "12h", 24: "24h", 48: "48h", 72: "72h", 168: "168h"}

COLOR = {CHRONOS: "#2a78d6", INCUMBENT: "#eb6834", REFERENCE: "#1baf7a"}
LABEL = {CHRONOS: "Chronos-2 (zero-shot)", INCUMBENT: "NPTS (incumbent)",
         REFERENCE: "SeasonalNaive-24"}

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_br", ["#d03b3b", "#e8a0a0", "#f0efec", "#86b6ef", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def caption(fig, text: str, width: int = 108, y: float = -0.04) -> None:
    """A wrapped note under a figure. Uses figure coordinates so ``savefig(bbox_inches="tight")``
    widens the whole figure rather than clipping the text."""
    fig.text(0.01, y, textwrap.fill(text, width), ha="left", va="top",
             fontsize=8.5, color=INK2, linespacing=1.5)


def save(fig, plots_dir, name: str) -> None:
    """Write both raster and vector. PNG for embedding, SVG so a figure can be re-typeset."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(plots_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  [plot] {name}.png / .svg")
