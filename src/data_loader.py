"""
data_loader.py — Load and merge Phase 1 regime + Phase 2 sentiment features.

Paths to upstream CSV files are resolved exclusively from environment variables
(SENTIMENT_FEATURES_PATH, REGIME_PROBS_PATH).  If either variable is unset or
the resolved path does not exist, a FileNotFoundError is raised immediately at
startup — no silent fallback.

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

def _resolve_path(template: str, var_name: str) -> str:
    """
    Expand environment variables in *template* and verify the file exists.

    Raises
    ------
    FileNotFoundError
        If the expanded path does not point to an existing file.  The error
        message names the expected environment variable so the user knows
        exactly what to set.
    """
    path = os.path.expandvars(template)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found at: {path}\n"
            f"Set the {var_name} environment variable to a valid file path.\n"
            f"See .env.example at the repo root for reference."
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
        ``config["data"]["regime_path"]`` must contain ``${...}`` env-var
        references.

    Returns
    -------
    pd.DataFrame
        ~5180 rows × 17 columns.  All numeric columns are NaN-free (the inner
        join ensures this by design — see CONTEXT.md §3).

    Raises
    ------
    FileNotFoundError
        If either upstream file is missing (see ``_resolve_path``).
    """
    sentiment_path = _resolve_path(
        config["data"]["sentiment_path"], "SENTIMENT_FEATURES_PATH"
    )
    regime_path = _resolve_path(
        config["data"]["regime_path"], "REGIME_PROBS_PATH"
    )

    sentiment = pd.read_csv(sentiment_path)
    regime = pd.read_csv(regime_path)

    sentiment["Date"] = pd.to_datetime(sentiment["Date"])
    regime["Date"] = pd.to_datetime(regime["Date"])

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
