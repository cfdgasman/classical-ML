"""Consistent matplotlib style + a ``savefig`` that writes into ``./images``.

Notebooks are executed with the chapter folder as the working directory, so
``savefig("foo")`` lands in ``<chapter>/images/foo.png`` where the chapter
README can embed it.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Colour-blind-safe categorical palette (Okabe-Ito order, black dropped).
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]


def set_style() -> None:
    """Apply the course-wide plot style. Call once at the top of a notebook."""
    mpl.rcParams.update({
        "figure.figsize": (7, 4.2),
        "figure.dpi": 100,
        "savefig.dpi": 120,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "image.cmap": "viridis",
    })


def savefig(name: str, fig=None, folder: str = "images") -> Path:
    """Save ``fig`` (default: current figure) as ``images/<name>.png`` and return the path."""
    fig = fig or plt.gcf()
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path)
    return path
