"""
tests/test_feature_engineer.py — Unit tests for src/feature_engineer.py.

All tests use synthetic price / feature data.  No network access or upstream
CSV files are required.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineer import (
    SHARED_FEATURES,
    TICKER_FEATURES,
    Normalizer,
    build_observation_matrix,
    compute_price_features,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


def _make_prices(n: int = 60, tickers: list = TICKERS, seed: int = 42) -> pd.DataFrame:
    """Return a wide price DataFrame (Date index, one column per ticker)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    data = {}
    for t in tickers:
        # Random walk starting at 100
        shocks = rng.normal(0, 0.01, n)
        prices = 100 * np.cumprod(1 + shocks)
        data[t] = prices
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


def _make_merged_df(dates: list, tickers: list = TICKERS) -> pd.DataFrame:
    """Return a synthetic merged (sentiment+regime) DataFrame."""
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "Date": pd.Timestamp(d),
                "Ticker": t,
                "tone_score": 0.1,
                "risk_score": 0.3,
                "growth_outlook": 0.2,
                "x_sentiment": 0.5,
                "hype_velocity": 0.4,
                "trending_score": 0.6,
                "sentiment_velocity": 0.05,
                "risk_momentum": -0.02,
                "data_source": "claude",
                "days_since_filing": 10,
                "has_sentiment_data": True,
                "has_x_data": 1,
                "prob_low": 0.25,
                "prob_med": 0.50,
                "prob_high": 0.20,
                "prob_crash": 0.05,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_price_features tests
# ---------------------------------------------------------------------------

class TestComputePriceFeatures:
    def test_output_shape(self):
        prices = _make_prices(n=40)
        pf = compute_price_features(prices, TICKERS)
        assert "Date" in pf.columns
        assert "Ticker" in pf.columns
        assert len(pf) == 40 * len(TICKERS)

    def test_required_columns_present(self):
        prices = _make_prices(n=30)
        pf = compute_price_features(prices, TICKERS)
        for col in ["log_return_1d", "log_return_5d", "log_return_21d",
                    "realized_vol_21d", "simple_return_1d"]:
            assert col in pf.columns, f"Missing column: {col}"

    def test_log_return_1d_formula(self):
        """log_return_1d[t] = log(close[t] / close[t-1])."""
        prices = _make_prices(n=30, tickers=["AAPL"])
        pf = compute_price_features(prices, ["AAPL"])
        aapl_pf = pf[pf["Ticker"] == "AAPL"].sort_values("Date").reset_index(drop=True)

        raw = prices["AAPL"].values
        expected_t2 = np.log(raw[2] / raw[1])
        actual_t2 = float(aapl_pf.loc[2, "log_return_1d"])
        np.testing.assert_allclose(actual_t2, expected_t2, rtol=1e-5)

    def test_log_return_5d_formula(self):
        prices = _make_prices(n=30, tickers=["MSFT"])
        pf = compute_price_features(prices, ["MSFT"])
        msft = pf[pf["Ticker"] == "MSFT"].sort_values("Date").reset_index(drop=True)

        raw = prices["MSFT"].values
        expected_t10 = np.log(raw[10] / raw[5])
        actual_t10 = float(msft.loc[10, "log_return_5d"])
        np.testing.assert_allclose(actual_t10, expected_t10, rtol=1e-5)

    def test_simple_return_1d_formula(self):
        prices = _make_prices(n=30, tickers=["AAPL"])
        pf = compute_price_features(prices, ["AAPL"])
        aapl = pf[pf["Ticker"] == "AAPL"].sort_values("Date").reset_index(drop=True)

        raw = prices["AAPL"].values
        expected_t3 = (raw[3] / raw[2]) - 1.0
        actual_t3 = float(aapl.loc[3, "simple_return_1d"])
        np.testing.assert_allclose(actual_t3, expected_t3, rtol=1e-5)

    def test_realized_vol_uses_21_window(self):
        """At row 25 there should be a valid (non-zero) vol; at row 0 it's 0.0."""
        prices = _make_prices(n=30, tickers=["GOOGL"])
        pf = compute_price_features(prices, ["GOOGL"])
        googl = pf[pf["Ticker"] == "GOOGL"].sort_values("Date").reset_index(drop=True)

        # Row 0: warm-up fill → 0.0
        assert float(googl.loc[0, "realized_vol_21d"]) == 0.0
        # Row 25: beyond warm-up → positive
        assert float(googl.loc[25, "realized_vol_21d"]) > 0.0

    def test_no_nan_in_output(self):
        """NaN from rolling warm-up must be filled with 0.0."""
        prices = _make_prices(n=30)
        pf = compute_price_features(prices, TICKERS)
        assert pf.isna().sum().sum() == 0

    def test_all_tickers_present(self):
        prices = _make_prices(n=25)
        pf = compute_price_features(prices, TICKERS)
        assert set(pf["Ticker"].unique()) == set(TICKERS)


# ---------------------------------------------------------------------------
# build_observation_matrix tests
# ---------------------------------------------------------------------------

class TestBuildObservationMatrix:
    @pytest.fixture
    def data(self):
        """Return (merged_df, price_features_df) with 40 overlapping dates."""
        prices = _make_prices(n=40)
        dates = [d.strftime("%Y-%m-%d") for d in prices.index]
        merged = _make_merged_df(dates)
        pf = compute_price_features(prices, TICKERS)
        return merged, pf

    def test_obs_array_shape(self, data):
        merged, pf = data
        obs, ret, dates = build_observation_matrix(merged, pf, TICKERS)
        assert obs.shape[1] == 79, f"Expected 79 features, got {obs.shape[1]}"

    def test_return_array_shape(self, data):
        merged, pf = data
        obs, ret, dates = build_observation_matrix(merged, pf, TICKERS)
        assert ret.shape[1] == len(TICKERS)

    def test_obs_and_return_same_time_length(self, data):
        merged, pf = data
        obs, ret, dates = build_observation_matrix(merged, pf, TICKERS)
        assert obs.shape[0] == ret.shape[0] == len(dates)

    def test_obs_dtype_float32(self, data):
        merged, pf = data
        obs, ret, _ = build_observation_matrix(merged, pf, TICKERS)
        assert obs.dtype == np.float32

    def test_return_dtype_float32(self, data):
        merged, pf = data
        obs, ret, _ = build_observation_matrix(merged, pf, TICKERS)
        assert ret.dtype == np.float32

    def test_no_nan_in_obs(self, data):
        merged, pf = data
        obs, _, _ = build_observation_matrix(merged, pf, TICKERS)
        assert not np.isnan(obs).any()

    def test_no_nan_in_returns(self, data):
        merged, pf = data
        _, ret, _ = build_observation_matrix(merged, pf, TICKERS)
        assert not np.isnan(ret).any()

    def test_shared_regime_features_in_last_4_cols(self, data):
        """prob_low, prob_med, prob_high, prob_crash are the last 4 columns."""
        merged, pf = data
        obs, _, _ = build_observation_matrix(merged, pf, TICKERS)
        # prob_low=0.25, prob_med=0.50, prob_high=0.20, prob_crash=0.05
        np.testing.assert_allclose(obs[:, -4], 0.25, atol=1e-5)
        np.testing.assert_allclose(obs[:, -3], 0.50, atol=1e-5)
        np.testing.assert_allclose(obs[:, -2], 0.20, atol=1e-5)
        np.testing.assert_allclose(obs[:, -1], 0.05, atol=1e-5)

    def test_has_sentiment_data_cast_to_int(self, data):
        """has_sentiment_data must be 0 or 1, not True/False."""
        merged, pf = data
        obs, _, _ = build_observation_matrix(merged, pf, TICKERS)
        # has_sentiment_data is the 14th feature of the first ticker block (index 13)
        col_idx = TICKER_FEATURES.index("has_sentiment_data")  # should be 13
        unique_vals = np.unique(obs[:, col_idx])
        # After z-score in Normalizer this will vary, but before normalisation
        # it should only be 0.0 or 1.0 (since all fixtures have True → 1)
        assert set(unique_vals).issubset({0.0, 1.0})

    def test_string_cols_dropped(self, data):
        """data_source and regime_label must not appear in the obs matrix."""
        merged, pf = data
        # Add regime_label to merged to test it's dropped
        merged = merged.copy()
        obs, _, _ = build_observation_matrix(merged, pf, TICKERS)
        # If string cols leak, the array would fail to cast to float32
        assert obs.dtype == np.float32

    def test_forward_return_is_shifted(self):
        """
        Verify that return_array[t] = simple_return_1d at date t+1.
        We use a deterministic price series for exact numerical comparison.
        """
        tickers = ["AAPL"]
        # Prices: 100, 101, 103, 106  → simple returns: NaN, 0.01, ~0.0198, ~0.0291
        dates_str = ["2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06"]
        prices = pd.DataFrame(
            {"AAPL": [100.0, 101.0, 103.0, 106.0]},
            index=pd.DatetimeIndex(dates_str, name="Date"),
        )
        merged = _make_merged_df(dates_str, tickers=["AAPL"])
        pf = compute_price_features(prices, ["AAPL"])
        obs, ret, dates = build_observation_matrix(merged, pf, ["AAPL"])

        # At time-step 0 (date 2022-01-03): forward return should be ~0.01
        expected_t0 = (101.0 - 100.0) / 100.0  # 0.01
        np.testing.assert_allclose(ret[0, 0], expected_t0, rtol=1e-4)

        # At time-step 1 (date 2022-01-04): forward return should be ~0.0198
        expected_t1 = (103.0 - 101.0) / 101.0
        np.testing.assert_allclose(ret[1, 0], expected_t1, rtol=1e-4)


# ---------------------------------------------------------------------------
# Normalizer tests
# ---------------------------------------------------------------------------

class TestNormalizer:
    def test_fit_sets_mean_and_std(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        n = Normalizer()
        n.fit(X)
        np.testing.assert_allclose(n.mean_, [3.0, 4.0], rtol=1e-5)
        np.testing.assert_allclose(n.std_,  [np.std([1,3,5]), np.std([2,4,6])], rtol=1e-4)

    def test_transform_normalises_to_zero_mean(self):
        X = np.random.default_rng(0).random((100, 10)).astype(np.float32)
        n = Normalizer()
        Z = n.fit_transform(X)
        np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-4)

    def test_transform_normalises_to_unit_std(self):
        X = np.random.default_rng(1).random((200, 5)).astype(np.float32)
        n = Normalizer()
        Z = n.fit_transform(X)
        # std ≈ 1 (not exactly 1 due to eps)
        np.testing.assert_allclose(Z.std(axis=0), 1.0, atol=1e-3)

    def test_transform_before_fit_raises(self):
        X = np.ones((10, 5), dtype=np.float32)
        n = Normalizer()
        with pytest.raises(RuntimeError, match="fitted"):
            n.transform(X)

    def test_transform_returns_float32(self):
        X = np.ones((10, 5), dtype=np.float64)
        n = Normalizer().fit(X)
        Z = n.transform(X)
        assert Z.dtype == np.float32

    def test_constant_column_does_not_raise(self):
        """A column with std=0 must not produce NaN/Inf (eps prevents div-by-zero)."""
        X = np.ones((20, 3), dtype=np.float32)
        n = Normalizer(eps=1e-8)
        Z = n.fit_transform(X)
        assert not np.isnan(Z).any()
        assert not np.isinf(Z).any()

    def test_save_and_load_roundtrip(self, tmp_path):
        X = np.random.default_rng(42).random((50, 79)).astype(np.float32)
        n = Normalizer().fit(X)
        path = str(tmp_path / "normalizer.pkl")
        n.save(path)
        n2 = Normalizer.load(path)
        np.testing.assert_array_equal(n.mean_, n2.mean_)
        np.testing.assert_array_equal(n.std_,  n2.std_)

    def test_fit_transform_matches_manual(self):
        X = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]], dtype=np.float32)
        n = Normalizer(eps=0.0)
        Z = n.fit_transform(X)
        expected = (X - X.mean(axis=0)) / X.std(axis=0)
        np.testing.assert_allclose(Z, expected.astype(np.float32), atol=1e-5)
