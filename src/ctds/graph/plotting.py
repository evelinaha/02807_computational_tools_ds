"""Shared matplotlib style for the report figures (one consistent look, colour-blind-safe order)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical palette in fixed order (validated for colour-vision deficiency on a light surface).
# Colour follows the entity: e.g. "raw" is always slot 1, "jaccard" slot 2, "ppmi" slot 3.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQUENTIAL = "Blues"

WEIGHTING_COLOURS = {"raw": PALETTE[0], "jaccard": PALETTE[1], "ppmi": PALETTE[2]}
GROUP_COLOURS = {"head": PALETTE[0], "mid": PALETTE[1], "tail": PALETTE[2]}


def set_style() -> None:
    """Apply the report style globally."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2, "lines.markersize": 5,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight", "font.size": 9,
    })


def save(fig: plt.Figure, folder: Path, name: str) -> Path:
    """Save a figure as PNG in ``folder`` and return the path (for the results document)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    return path
