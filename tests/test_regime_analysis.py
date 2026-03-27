"""
tests/test_regime_analysis.py — Unit tests for src/regime_analysis.py.

All tests use synthetic numpy arrays.  No trained model, data files, or
network access required.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.regime_analysis import (
    REGIME_LABELS,
    analyse_regime_performance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_TICKERS = 5
ACTION_DIM = 6  # 5 tickers + 1 cash
T = 100
ASSET_NAMES = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "cash"]


def _make_synthetic_data(T: int = T, seed: int = 42):
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(0.001, 0.01, T).astype(np.float32)
    daily_weights = np.abs(rng.normal(0.16, 0.02, (T, ACTION_DIM))).astype(np.float32)
    daily_weights /= daily_weights.sum(axis=1, keepdims=True)
    # Assign regimes: 4 roughly equal groups
    regime_indices = np.array([i % 4 for i in range(T)])
    return daily_returns, daily_weights, regime_indices


# ---------------------------------------------------------------------------
# analyse_regime_performance
# ---------------------------------------------------------------------------

class TestAnalyseRegimePerformance:

    def test_returns_dict_with_all_regime_labels(self):
        daily_returns, daily_weights, regime_indices = _make_synthetic_data()
        result = analyse_regime_performance(
            daily_returns, daily_weights, regime_indices, ASSET_NAMES
        )
        for label in REGIME_LABELS:
            assert label in result, f"Missing regime label: {label}"

    def test_each_regime_has_required_keys(self):
        daily_returns, daily_weights, regime_indices = _make_synthetic_data()
        result = analyse_regime_performance(
            daily_returns, daily_weights, regime_indices, ASSET_NAMES
        )
        for label in REGIME_LABELS:
            r = result[label]
            assert "mean_weights" in r
            assert "mean_daily_return" in r
            assert "sharpe" in r
            assert "n_days" in r

    def test_mean_weights_sum_to_approximately_one(self):
        daily_returns, daily_weights, regime_indices = _make_synthetic_data()
        result = analyse_regime_performance(
            daily_returns, daily_weights, regime_indices, ASSET_NAMES
        )
        for label in REGIME_LABELS:
            w = result[label]["mean_weights"]
            total = sum(w.values())
            np.testing.assert_allclose(total, 1.0, atol=0.01)

    def test_n_days_sums_to_total(self):
        daily_returns, daily_weights, regime_indices = _make_synthetic_data()
        result = analyse_regime_performance(
            daily_returns, daily_weights, regime_indices, ASSET_NAMES
        )
        total_days = sum(result[label]["n_days"] for label in REGIME_LABELS)
        assert total_days == T

    def test_sharpe_is_finite(self):
        daily_returns, daily_weights, regime_indices = _make_synthetic_data()
        result = analyse_regime_performance(
            daily_returns, daily_weights, regime_indices, ASSET_NAMES
        )
        for label in REGIME_LABELS:
            assert np.isfinite(result[label]["sharpe"])

    def test_empty_regime_returns_zeros(self):
        """If no days have a given regime, return zeros gracefully."""
        daily_returns, daily_weights, _ = _make_synthetic_data(T=20)
        # All days are regime 0 — regimes 1,2,3 have 0 days
        regime_indices = np.zeros(20, dtype=int)
        result = analyse_regime_performance(
            daily_returns, daily_weights, regime_indices, ASSET_NAMES
        )
        assert result["medium"]["n_days"] == 0
        assert result["medium"]["sharpe"] == 0.0
        assert result["low_vol"]["n_days"] == 20

    def test_different_regimes_get_correct_days(self):
        """Verify regime assignment splits data correctly."""
        T_ = 40
        daily_returns = np.ones(T_, dtype=np.float32) * 0.001
        daily_weights = np.ones((T_, ACTION_DIM), dtype=np.float32) / ACTION_DIM
        # First 10 = regime 0, next 10 = regime 1, etc.
        regime_indices = np.array([i // 10 for i in range(T_)])
        result = analyse_regime_performance(
            daily_returns, daily_weights, regime_indices, ASSET_NAMES
        )
        for label in REGIME_LABELS:
            assert result[label]["n_days"] == 10

    def test_mean_weights_match_manual(self):
        """With uniform weights across all days, mean weight per asset = 1/n."""
        T_ = 20
        daily_returns = np.zeros(T_, dtype=np.float32)
        uniform_w = np.ones((T_, ACTION_DIM), dtype=np.float32) / ACTION_DIM
        regime_indices = np.zeros(T_, dtype=int)
        result = analyse_regime_performance(
            daily_returns, uniform_w, regime_indices, ASSET_NAMES
        )
        for name in ASSET_NAMES:
            np.testing.assert_allclose(
                result["low_vol"]["mean_weights"][name],
                1.0 / ACTION_DIM, atol=1e-5
            )
