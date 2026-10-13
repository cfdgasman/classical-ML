"""Loaders for the raw CSV files vendored in ``data/raw``.

Every loader returns the file *exactly as shipped* (no cleaning) so that the
cleaning itself can be taught in the notebooks.
"""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw"


def _read(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, **kwargs)


def load_titanic() -> pd.DataFrame:
    """891 Titanic passengers (seaborn-data). Classic binary target: ``survived``."""
    return _read("titanic.csv")


def load_mpg() -> pd.DataFrame:
    """398 cars, 1970-82 (UCI Auto MPG via seaborn-data). Regression target: ``mpg``."""
    return _read("mpg.csv")


def load_penguins() -> pd.DataFrame:
    """344 Palmer penguins (seaborn-data). Multiclass target: ``species``."""
    return _read("penguins.csv")


def load_concrete() -> pd.DataFrame:
    """1030 concrete mixes (Yeh 1998, UCI). Regression target: ``strength`` in MPa.

    Mix components are kg per m^3 of concrete, ``age`` is in days.
    """
    return _read("concrete.csv")
