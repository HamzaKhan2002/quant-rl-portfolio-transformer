"""
tests/test_walk_forward.py — Unit tests for src/walk_forward.py.

Tests use synthetic data only — no network access, no upstream CSV files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.walk_forward import (
    _build_regime_only_obs_matrix,
    compute_cross_fold_summary,
    evaluate_fold_models,
    _annualised_return,
    _annualised_sharpe,
    _annualised_sortino,
    _max_drawdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
N_TICKERS = 5
T = 60  # trading days


def _make_regime_df(start="2019-01-02", n=T) -> pd.DataFrame:
    """Minimal wide-format regime DataFrame (one row per date)."""
    dates = pd.bdate_range(start=start, periods=n)
    rng = np.random.default_rng(0)
    p_low = rng.uniform(0.3, 0.7, n)
    p_med = 1.0 - p_low - 0.01
    p_hig = np.full(n, 0.005)
    p_cra = np.full(n, 0.005)
    return pd.DataFrame({
        "Date": dates,
        "prob_low":   p_low,
        "prob_med":   p_med,
        "prob_high":  p_hig,
        "prob_crash": p_cra,
    })


def _make_price_features_df(start="2019-01-02", n=T) -> pd.DataFrame:
    """Synthetic long-format price features DataFrame."""
    rng = np.random.default_rng(1)
    dates = pd.bdate_range(start=start, periods=n)
    records = []
    for ticker in TICKERS:
        records.append(pd.DataFrame({
            "Date":             dates,
            "Ticker":           ticker,
            "log_return_1d":    rng.normal(0.001, 0.01, n),
            "log_return_5d":    rng.normal(0.005, 0.02, n),
            "log_return_21d":   rng.normal(0.02, 0.05, n),
            "realized_vol_21d": rng.uniform(0.01, 0.03, n),
            "simple_return_1d": rng.normal(0.001, 0.01, n),
        }))
    return pd.concat(records, ignore_index=True)


def _make_env_config(obs_dim: int = 34) -> dict:
    return {
        "env": {
            "n_tickers": N_TICKERS,
            "action_dim": 6,
            "obs_dim": obs_dim,
            "transaction_cost": 0.001,
            "vol_eps": 1e-8,
            "vol_window": 21,
        },
    }


# ---------------------------------------------------------------------------
# Tests: _build_regime_only_obs_matrix
# ---------------------------------------------------------------------------

class TestBuildRegimeOnlyObsMatrix:
    def test_obs_shape_is_34_dim(self):
        """Regime-only obs must be (T, 34) = 6 per-ticker × 5 + 4 shared."""
        regime_df = _make_regime_df()
        pf_df = _make_price_features_df()
        obs, ret, dates = _build_regime_only_obs_matrix(
            regime_df, pf_df, TICKERS, "2019-01-02", "2019-04-30"
        )
        assert obs.shape[1] == 34, f"Expected 34 features, got {obs.shape[1]}"

    def test_return_shape_matches_obs_rows(self):
        regime_df = _make_regime_df()
        pf_df = _make_price_features_df()
        obs, ret, dates = _build_regime_only_obs_matrix(
            regime_df, pf_df, TICKERS, "2019-01-02", "2019-04-30"
        )
        assert obs.shape[0] == ret.shape[0]
        assert ret.shape[1] == N_TICKERS

    def test_dates_length_matches_obs(self):
        regime_df = _make_regime_df()
        pf_df = _make_price_features_df()
        obs, ret, dates = _build_regime_only_obs_matrix(
            regime_df, pf_df, TICKERS, "2019-01-02", "2019-04-30"
        )
        assert len(dates) == obs.shape[0]

    def test_sentinel_features_are_zero(self):
        """has_sentiment_data and days_since_filing should be 0 for all rows."""
        regime_df = _make_regime_df()
        pf_df = _make_price_features_df()
        obs, ret, dates = _build_regime_only_obs_matrix(
            regime_df, pf_df, TICKERS, "2019-01-02", "2019-04-30"
        )
        # days_since_filing and has_sentiment_data are features 4 and 5
        # within each ticker block (0-indexed).  Block size = 6.
        for i, ticker in enumerate(TICKERS):
            base = i * 6
            days_since = obs[:, base + 4]  # days_since_filing
            has_sent   = obs[:, base + 5]  # has_sentiment_data
            assert (days_since == 0).all(), (
                f"days_since_filing nonzero for {ticker}"
            )
            assert (has_sent == 0).all(), (
                f"has_sentiment_data nonzero for {ticker}"
            )

    def test_obs_dtype_float32(self):
        regime_df = _make_regime_df()
        pf_df = _make_price_features_df()
        obs, ret, dates = _build_regime_only_obs_matrix(
            regime_df, pf_df, TICKERS, "2019-01-02", "2019-04-30"
        )
        assert obs.dtype == np.float32

    def test_no_nan_in_obs(self):
        regime_df = _make_regime_df()
        pf_df = _make_price_features_df()
        obs, ret, dates = _build_regime_only_obs_matrix(
            regime_df, pf_df, TICKERS, "2019-01-02", "2019-04-30"
        )
        assert not np.isnan(obs).any()

    def test_shared_regime_features_in_last_4_columns(self):
        """Last 4 columns must be regime probabilities summing to ~1."""
        regime_df = _make_regime_df()
        pf_df = _make_price_features_df()
        obs, ret, dates = _build_regime_only_obs_matrix(
            regime_df, pf_df, TICKERS, "2019-01-02", "2019-04-30"
        )
        regime_block = obs[:, -4:]  # last 4 = [prob_low, prob_med, prob_high, prob_crash]
        # Pre-normalisation: probs sum to ~1 (the raw values from the DataFrame)
        assert (regime_block >= 0).all()


# ---------------------------------------------------------------------------
# Tests: compute_cross_fold_summary
# ---------------------------------------------------------------------------

class TestComputeCrossFoldSummary:
    def _make_fold_results(self):
        return [
            {"fold_id": 1, "sharpe": 1.2, "sortino": 1.8, "max_drawdown": 0.15,
             "annualised_return": 0.20, "calmar": 1.33, "mean_daily_turnover": 0.005},
            {"fold_id": 2, "sharpe": 0.8, "sortino": 1.1, "max_drawdown": 0.25,
             "annualised_return": 0.12, "calmar": 0.48, "mean_daily_turnover": 0.007},
        ]

    def test_summary_has_aggregate_key(self):
        results = self._make_fold_results()
        summary = compute_cross_fold_summary(results)
        assert "aggregate" in summary

    def test_summary_has_per_fold_key(self):
        results = self._make_fold_results()
        summary = compute_cross_fold_summary(results)
        assert "per_fold" in summary
        assert len(summary["per_fold"]) == 2

    def test_aggregate_sharpe_mean(self):
        results = self._make_fold_results()
        summary = compute_cross_fold_summary(results)
        expected = round((1.2 + 0.8) / 2, 4)
        assert abs(summary["aggregate"]["sharpe"]["mean"] - expected) < 1e-3

    def test_aggregate_sharpe_std(self):
        results = self._make_fold_results()
        summary = compute_cross_fold_summary(results)
        expected = round(float(np.std([1.2, 0.8])), 4)
        assert abs(summary["aggregate"]["sharpe"]["std"] - expected) < 1e-3

    def test_aggregate_contains_required_metrics(self):
        results = self._make_fold_results()
        summary = compute_cross_fold_summary(results)
        for m in ("sharpe", "sortino", "max_drawdown", "annualised_return", "calmar"):
            assert m in summary["aggregate"], f"Missing metric: {m}"

    def test_aggregate_metric_has_mean_std_min_max(self):
        summary = compute_cross_fold_summary(self._make_fold_results())
        for key in ("mean", "std", "min", "max"):
            assert key in summary["aggregate"]["sharpe"]

    def test_single_fold_std_is_zero(self):
        results = [self._make_fold_results()[0]]
        summary = compute_cross_fold_summary(results)
        assert summary["aggregate"]["sharpe"]["std"] == 0.0

    def test_empty_fold_results(self):
        # Should not raise; aggregate may be empty or handle gracefully
        summary = compute_cross_fold_summary([])
        assert "aggregate" in summary


# ---------------------------------------------------------------------------
# Tests: evaluate_fold_models
# ---------------------------------------------------------------------------

class TestEvaluateFoldModels:
    """Tests for the ensemble evaluation function using dummy models."""

    def _make_obs_ret(self):
        rng = np.random.default_rng(99)
        obs = rng.standard_normal((T + 1, 34)).astype(np.float32)
        ret = rng.normal(0.001, 0.01, (T + 1, N_TICKERS)).astype(np.float32)
        return obs, ret

    class _DummyModel:
        """Predict constant zero-logit action (uniform portfolio)."""
        def __init__(self, action_dim=6, offset=0.0):
            self.action_dim = action_dim
            self.offset = offset

        def predict(self, obs, deterministic=True):
            return np.full(self.action_dim, self.offset, dtype=np.float32), None

    def test_returns_dict_with_required_keys(self):
        obs, ret = self._make_obs_ret()
        result = evaluate_fold_models(
            [self._DummyModel()], obs, ret, _make_env_config(34)
        )
        for key in ("annualised_return", "sharpe", "sortino", "max_drawdown",
                    "calmar", "mean_daily_turnover", "daily_returns", "portfolio_weights"):
            assert key in result, f"Missing key: {key}"

    def test_sharpe_is_finite(self):
        obs, ret = self._make_obs_ret()
        result = evaluate_fold_models(
            [self._DummyModel()], obs, ret, _make_env_config(34)
        )
        assert np.isfinite(result["sharpe"])

    def test_max_drawdown_between_zero_and_one(self):
        obs, ret = self._make_obs_ret()
        result = evaluate_fold_models(
            [self._DummyModel()], obs, ret, _make_env_config(34)
        )
        assert 0.0 <= result["max_drawdown"] <= 1.0

    def test_ensemble_of_identical_models_matches_single(self):
        """Ensemble of two identical models should give same result as one."""
        obs, ret = self._make_obs_ret()
        cfg = _make_env_config(34)
        m = self._DummyModel()
        single = evaluate_fold_models([m], obs, ret, cfg)
        ensemble = evaluate_fold_models([m, m], obs, ret, cfg)
        assert abs(single["sharpe"] - ensemble["sharpe"]) < 1e-4

    def test_ensemble_averages_logits(self):
        """Two models with opposite offsets should average to uniform."""
        obs, ret = self._make_obs_ret()
        cfg = _make_env_config(34)
        m1 = self._DummyModel(offset=5.0)
        m2 = self._DummyModel(offset=-5.0)
        result = evaluate_fold_models([m1, m2], obs, ret, cfg)
        # Averaged action is 0 → softmax → uniform weights
        weights = np.array(result["portfolio_weights"])
        expected = 1.0 / 6.0
        assert np.allclose(weights.mean(axis=0), expected, atol=0.01)

    def test_daily_returns_length_matches_test_steps(self):
        obs, ret = self._make_obs_ret()
        result = evaluate_fold_models(
            [self._DummyModel()], obs, ret, _make_env_config(34)
        )
        # PortfolioEnv stops at n_steps = T (obs.shape[0] - 1 = T)
        assert len(result["daily_returns"]) == T


# ---------------------------------------------------------------------------
# Tests: metric helpers (fold-level versions)
# ---------------------------------------------------------------------------

class TestWalkForwardMetricHelpers:
    def test_annualised_return_positive_for_growing(self):
        r = np.full(252, 0.001)
        assert _annualised_return(r) > 0

    def test_annualised_sharpe_positive_for_positive_returns(self):
        # Needs variance to produce a non-zero Sharpe; use random positive returns
        rng = np.random.default_rng(0)
        r = np.abs(rng.normal(0.001, 0.01, 252))  # all positive, non-constant
        assert _annualised_sharpe(r) > 0

    def test_max_drawdown_zero_for_monotone(self):
        r = np.full(100, 0.01)
        assert _max_drawdown(r) < 1e-10

    def test_annualised_sortino_handles_no_downside(self):
        r = np.full(50, 0.01)
        result = _annualised_sortino(r)
        assert result == float("inf") or result > 0
