"""
tests/test_data_loader.py — Unit tests for src/data_loader.py.

All tests are fully self-contained: they create synthetic CSV fixtures in
tmp_path and monkeypatch environment variables.  No upstream data files or
network access are required.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data_loader import (
    _resolve_path,
    load_config,
    load_merged_features,
    split_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Columns from CONTEXT.md §2 / §3
SENTIMENT_COLS = [
    "Date", "Ticker", "tone_score", "risk_score", "growth_outlook",
    "x_sentiment", "hype_velocity", "trending_score", "sentiment_velocity",
    "risk_momentum", "data_source", "days_since_filing", "has_sentiment_data",
]
REGIME_COLS = ["Date", "prob_low", "prob_med", "prob_high", "prob_crash", "regime_label"]

EXPECTED_MERGED_COLS = [
    "Date", "Ticker", "tone_score", "risk_score", "growth_outlook",
    "x_sentiment", "hype_velocity", "trending_score", "sentiment_velocity",
    "risk_momentum", "data_source", "days_since_filing", "has_sentiment_data",
    "prob_low", "prob_med", "prob_high", "prob_crash",
    # regime_label comes from Phase 1 CSV; allowed but not used as a feature
]


def _make_sentiment_df(dates: list) -> pd.DataFrame:
    """Return a minimal sentiment DataFrame matching the Phase 2 schema."""
    rows = []
    for d in dates:
        for t in TICKERS:
            rows.append({
                "Date": d,
                "Ticker": t,
                "tone_score": 0.1,
                "risk_score": 0.3,
                "growth_outlook": 0.2,
                "x_sentiment": 0.5,
                "hype_velocity": 0.4,
                "trending_score": 0.6,
                "sentiment_velocity": 0.0,
                "risk_momentum": 0.0,
                "data_source": "claude",
                "days_since_filing": 10,
                "has_sentiment_data": True,
            })
    return pd.DataFrame(rows)


def _make_regime_df(dates: list) -> pd.DataFrame:
    """Return a minimal regime DataFrame matching the Phase 1 schema."""
    rows = []
    for d in dates:
        rows.append({
            "Date": d,
            "prob_low": 0.25,
            "prob_med": 0.50,
            "prob_high": 0.20,
            "prob_crash": 0.05,
            "regime_label": "med",
        })
    return pd.DataFrame(rows)


def _write_csvs(tmp_path, sentiment_dates, regime_dates):
    """Write synthetic CSV files and return (sentiment_path, regime_path)."""
    s_path = tmp_path / "sentiment_features.csv"
    r_path = tmp_path / "regime_probs.csv"
    _make_sentiment_df(sentiment_dates).to_csv(s_path, index=False)
    _make_regime_df(regime_dates).to_csv(r_path, index=False)
    return str(s_path), str(r_path)


def _make_config(s_path: str, r_path: str) -> dict:
    """Return a minimal config dict with literal (already-expanded) paths."""
    return {
        "data": {
            "sentiment_path": s_path,
            "regime_path": r_path,
        },
        "splits": {
            "train_start": "2022-01-03",
            "train_end":   "2023-12-31",
            "val_start":   "2024-01-01",
            "val_end":     "2024-12-31",
            "test_start":  "2025-01-01",
            "test_end":    "2025-12-31",
        },
    }


# ---------------------------------------------------------------------------
# _resolve_path tests
# ---------------------------------------------------------------------------

class TestResolvePath:
    def test_existing_file_returns_path(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col\n1\n")
        result = _resolve_path(str(f), "SOME_VAR")
        assert result == str(f)

    def test_missing_file_raises(self, tmp_path):
        missing = str(tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError, match="SOME_VAR"):
            _resolve_path(missing, "SOME_VAR")

    def test_unset_env_var_raises(self, monkeypatch):
        monkeypatch.delenv("SENTIMENT_FEATURES_PATH", raising=False)
        with pytest.raises(FileNotFoundError, match="SENTIMENT_FEATURES_PATH"):
            _resolve_path("${SENTIMENT_FEATURES_PATH}", "SENTIMENT_FEATURES_PATH")

    def test_env_var_set_but_file_missing_raises(self, monkeypatch, tmp_path):
        missing = str(tmp_path / "gone.csv")
        monkeypatch.setenv("SENTIMENT_FEATURES_PATH", missing)
        with pytest.raises(FileNotFoundError, match="SENTIMENT_FEATURES_PATH"):
            _resolve_path("${SENTIMENT_FEATURES_PATH}", "SENTIMENT_FEATURES_PATH")

    def test_env_var_set_and_file_exists(self, monkeypatch, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col\n1\n")
        monkeypatch.setenv("SENTIMENT_FEATURES_PATH", str(f))
        result = _resolve_path("${SENTIMENT_FEATURES_PATH}", "SENTIMENT_FEATURES_PATH")
        assert result == str(f)


# ---------------------------------------------------------------------------
# load_config tests
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_yaml(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("tickers:\n  - AAPL\n  - MSFT\n")
        cfg = load_config(str(cfg_path))
        assert cfg["tickers"] == ["AAPL", "MSFT"]

    def test_real_config_has_required_keys(self):
        cfg_path = Path(__file__).parent.parent / "config.yaml"
        cfg = load_config(str(cfg_path))
        assert "tickers" in cfg
        assert "data" in cfg
        assert "splits" in cfg
        assert "env" in cfg
        assert "ppo" in cfg

    def test_real_config_obs_dim(self):
        cfg_path = Path(__file__).parent.parent / "config.yaml"
        cfg = load_config(str(cfg_path))
        n_tickers = cfg["env"]["n_tickers"]
        n_per_ticker = cfg["env"]["n_per_ticker_features"]
        n_shared = cfg["env"]["n_shared_features"]
        assert cfg["env"]["obs_dim"] == n_tickers * n_per_ticker + n_shared

    def test_real_config_split_non_overlap(self):
        cfg_path = Path(__file__).parent.parent / "config.yaml"
        cfg = load_config(str(cfg_path))
        s = cfg["splits"]
        assert s["train_end"] < s["val_start"]
        assert s["val_end"] < s["test_start"]


# ---------------------------------------------------------------------------
# load_merged_features tests
# ---------------------------------------------------------------------------

class TestLoadMergedFeatures:
    def test_merge_returns_dataframe(self, tmp_path):
        dates = ["2022-01-03", "2022-01-04", "2022-01-05"]
        s_path, r_path = _write_csvs(tmp_path, dates, dates)
        cfg = _make_config(s_path, r_path)
        df = load_merged_features(cfg)
        assert isinstance(df, pd.DataFrame)

    def test_merge_has_correct_row_count(self, tmp_path):
        dates = ["2022-01-03", "2022-01-04", "2022-01-05"]
        s_path, r_path = _write_csvs(tmp_path, dates, dates)
        cfg = _make_config(s_path, r_path)
        df = load_merged_features(cfg)
        # 3 dates × 5 tickers = 15 rows
        assert len(df) == 15

    def test_inner_join_drops_holiday_rows(self, tmp_path):
        """Dates in sentiment but absent from regime are dropped (inner join)."""
        sentiment_dates = ["2022-01-03", "2022-01-17", "2022-01-18"]  # Jan 17 = MLK Day
        regime_dates    = ["2022-01-03", "2022-01-18"]                 # regime skips holiday
        s_path, r_path = _write_csvs(tmp_path, sentiment_dates, regime_dates)
        cfg = _make_config(s_path, r_path)
        df = load_merged_features(cfg)
        # Only dates present in BOTH should survive: 2022-01-03 and 2022-01-18
        unique_dates = df["Date"].dt.date.astype(str).unique()
        assert "2022-01-17" not in unique_dates
        assert "2022-01-03" in unique_dates
        assert "2022-01-18" in unique_dates
        assert len(df) == 2 * len(TICKERS)

    def test_merge_zero_nan_in_numeric_cols(self, tmp_path):
        dates = pd.date_range("2022-01-03", periods=10, freq="B").strftime("%Y-%m-%d").tolist()
        s_path, r_path = _write_csvs(tmp_path, dates, dates)
        cfg = _make_config(s_path, r_path)
        df = load_merged_features(cfg)
        numeric_cols = df.select_dtypes(include="number").columns
        assert df[numeric_cols].isna().sum().sum() == 0

    def test_merge_contains_expected_columns(self, tmp_path):
        dates = ["2022-01-03", "2022-01-04"]
        s_path, r_path = _write_csvs(tmp_path, dates, dates)
        cfg = _make_config(s_path, r_path)
        df = load_merged_features(cfg)
        required = [
            "Date", "Ticker", "tone_score", "risk_score", "growth_outlook",
            "x_sentiment", "hype_velocity", "trending_score",
            "sentiment_velocity", "risk_momentum", "data_source",
            "days_since_filing", "has_sentiment_data",
            "prob_low", "prob_med", "prob_high", "prob_crash",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_regime_probs_sum_to_one(self, tmp_path):
        dates = ["2022-01-03", "2022-01-04"]
        s_path, r_path = _write_csvs(tmp_path, dates, dates)
        cfg = _make_config(s_path, r_path)
        df = load_merged_features(cfg)
        prob_sum = (
            df["prob_low"] + df["prob_med"] + df["prob_high"] + df["prob_crash"]
        )
        np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-6)

    def test_missing_sentiment_file_raises(self, tmp_path):
        cfg = _make_config(
            str(tmp_path / "missing_sentiment.csv"),
            str(tmp_path / "missing_regime.csv"),
        )
        with pytest.raises(FileNotFoundError, match="SENTIMENT_FEATURES_PATH"):
            load_merged_features(cfg)

    def test_missing_regime_file_raises(self, tmp_path):
        dates = ["2022-01-03"]
        s_path, _ = _write_csvs(tmp_path, dates, dates)
        cfg = _make_config(s_path, str(tmp_path / "missing_regime.csv"))
        with pytest.raises(FileNotFoundError, match="REGIME_PROBS_PATH"):
            load_merged_features(cfg)

    def test_date_column_is_datetime(self, tmp_path):
        dates = ["2022-01-03", "2022-01-04"]
        s_path, r_path = _write_csvs(tmp_path, dates, dates)
        cfg = _make_config(s_path, r_path)
        df = load_merged_features(cfg)
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])


# ---------------------------------------------------------------------------
# split_data tests
# ---------------------------------------------------------------------------

class TestSplitData:
    @pytest.fixture
    def sample_df(self):
        """Synthetic merged DataFrame covering 2022-2025 with 5 tickers."""
        dates = pd.date_range("2022-01-03", "2025-12-31", freq="B")
        rows = []
        for d in dates:
            for t in TICKERS:
                rows.append({"Date": d, "Ticker": t, "value": 1.0})
        return pd.DataFrame(rows)

    @pytest.fixture
    def config(self):
        cfg_path = Path(__file__).parent.parent / "config.yaml"
        return load_config(str(cfg_path))

    def test_splits_are_non_overlapping(self, sample_df, config):
        train, val, test = split_data(sample_df, config)
        assert train["Date"].max() < val["Date"].min()
        assert val["Date"].max() < test["Date"].min()

    def test_train_within_bounds(self, sample_df, config):
        train, _, _ = split_data(sample_df, config)
        s = config["splits"]
        assert train["Date"].min() >= pd.Timestamp(s["train_start"])
        assert train["Date"].max() <= pd.Timestamp(s["train_end"])

    def test_val_within_bounds(self, sample_df, config):
        _, val, _ = split_data(sample_df, config)
        s = config["splits"]
        assert val["Date"].min() >= pd.Timestamp(s["val_start"])
        assert val["Date"].max() <= pd.Timestamp(s["val_end"])

    def test_test_within_bounds(self, sample_df, config):
        _, _, test = split_data(sample_df, config)
        s = config["splits"]
        assert test["Date"].min() >= pd.Timestamp(s["test_start"])
        assert test["Date"].max() <= pd.Timestamp(s["test_end"])

    def test_no_rows_lost_between_splits(self, sample_df, config):
        """All rows in [train_start, test_end] must appear in one of the splits."""
        s = config["splits"]
        full_mask = (
            (sample_df["Date"] >= s["train_start"]) &
            (sample_df["Date"] <= s["test_end"])
        )
        total_expected = full_mask.sum()
        train, val, test = split_data(sample_df, config)
        assert len(train) + len(val) + len(test) == total_expected

    def test_splits_reset_index(self, sample_df, config):
        train, val, test = split_data(sample_df, config)
        assert train.index[0] == 0
        assert val.index[0] == 0
        assert test.index[0] == 0

    def test_approximate_train_size(self, sample_df, config):
        """Train set should contain roughly 494-522 unique dates.

        The lower bound (494) allows for NYSE-only calendars (~504 trading
        days minus holiday variation); the upper bound (522) covers a full
        business-day calendar (freq="B") which includes ~16-18 US bank
        holidays that are weekdays, giving up to ~522 dates over 2022-2023.
        """
        train, _, _ = split_data(sample_df, config)
        unique_dates = train["Date"].nunique()
        assert 494 <= unique_dates <= 522, f"Unexpected train dates: {unique_dates}"

    def test_approximate_val_size(self, sample_df, config):
        """Val set should contain ~252 unique trading dates × 5 tickers."""
        _, val, _ = split_data(sample_df, config)
        unique_dates = val["Date"].nunique()
        assert 242 <= unique_dates <= 262, f"Unexpected val dates: {unique_dates}"
