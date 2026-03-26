"""
data_loader.py — Load and merge Phase 1 regime + Phase 2 sentiment features.

Paths to upstream CSV files are read directly from config.yaml as relative
paths from the repo root (deps/ submodules).  No environment variables are
used.  If a file is missing the likely cause is an uninitialised submodule, and
the error message tells the user exactly how to fix it.

Merge recipe follows CONTEXT.md §3:
  - Inner join on Date (drops ~38 US holiday rows present in sentiment but
    absent from the NYSE trading-day regime calendar).
  - Result: ~5180 rows × 17 columns, 0 NaN.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config from *path*."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_path(path: str) -> str:
    """
    Verify that *path* exists and return it unchanged.

    Raises
    ------
    FileNotFoundError
        If the path does not point to an existing file.  The error message
        includes a submodule hint so the user knows how to fix it.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            f"Run: git submodule update --init --recursive"
        )
    return path


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_merged_features(config: dict) -> pd.DataFrame:
    """
    Load Phase 2 sentiment features and Phase 1 regime probabilities, merge
    them on Date (inner join), and return the combined DataFrame.

    Parameters
    ----------
    config : dict
        Loaded config.yaml.  ``config["data"]["sentiment_path"]`` and
        ``config["data"]["regime_path"]`` are paths relative to the repo root
        (populated from the deps/ submodules by default).

    Returns
    -------
    pd.DataFrame
        ~5180 rows × 17 columns.  All numeric columns are NaN-free (the inner
        join ensures this by design — see CONTEXT.md §3).

    Raises
    ------
    FileNotFoundError
        If either upstream file is missing.  Run
        ``git submodule update --init --recursive`` to populate deps/.
    """
    sentiment_path = _resolve_path(config["data"]["sentiment_path"])
    regime_path    = _resolve_path(config["data"]["regime_path"])

    sentiment = pd.read_csv(sentiment_path)
    regime    = pd.read_csv(regime_path)

    sentiment["Date"] = pd.to_datetime(sentiment["Date"])
    regime["Date"]    = pd.to_datetime(regime["Date"])

    # Inner merge — drops ~38 US holiday dates that appear in sentiment
    # (generated with freq="B" which includes bank holidays) but are absent
    # from the NYSE trading-day calendar used by Phase 1.
    merged = sentiment.merge(regime, on="Date", how="inner")

    return merged


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into non-overlapping train / val / test sets by date range.

    Boundaries (from config.yaml):
      Train : 2022-01-03 → 2023-12-31  (~504 trading days)
      Val   : 2024-01-01 → 2024-12-31  (~252 trading days)
      Test  : 2025-01-01 → 2025-12-31  (~252 trading days, out-of-sample)

    Parameters
    ----------
    df : pd.DataFrame
        Merged features DataFrame returned by ``load_merged_features``.
    config : dict
        Loaded config.yaml.

    Returns
    -------
    train_df, val_df, test_df : tuple of pd.DataFrame
    """
    dates = df["Date"]

    splits = config["splits"]
    train_mask = (dates >= splits["train_start"]) & (dates <= splits["train_end"])
    val_mask   = (dates >= splits["val_start"])   & (dates <= splits["val_end"])
    test_mask  = (dates >= splits["test_start"])  & (dates <= splits["test_end"])

    return (
        df[train_mask].reset_index(drop=True),
        df[val_mask].reset_index(drop=True),
        df[test_mask].reset_index(drop=True),
    )
