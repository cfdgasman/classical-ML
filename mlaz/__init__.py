"""mlaz -- tiny shared helpers for the ML A-to-Z (classical) notebooks.

Kept deliberately small: the *learning* code lives in the notebooks.
This package only handles boring plumbing (paths, plot style, saving figures)
so every chapter looks the same and runs from any working directory.
"""
from .datasets import DATA_DIR, REPO_ROOT, load_concrete, load_mpg, load_penguins, load_titanic
from .plotting import PALETTE, savefig, set_style

__all__ = [
    "DATA_DIR", "REPO_ROOT", "PALETTE", "savefig", "set_style",
    "load_concrete", "load_mpg", "load_penguins", "load_titanic",
]
