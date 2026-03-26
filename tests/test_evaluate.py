"""
tests/test_evaluate.py — Unit tests for src/evaluate.py.

All tests use synthetic numpy arrays.  No trained model, data files, or
network access required.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import (
    _annualised_return,
    _annualised_sharpe,
    _annualised_sortino,
    _max_drawdown,
    _mean_daily_turnover,
    _run_sixty_forty_from_returns,
    bootstrap_ci,
    compute_metrics,
    run_equal_weight_buyhold,
    run_equal_weight_monthly,
    run_max_sharpe,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_TICKERS = 5
T_TRAIN = 200
T_TEST = 60

RNG = np.random.default_rng(42)
TRAIN_RET = RNG.normal(0.001, 0.01, (T_TRAIN, N_TICKERS)).astype(np.float32)
TEST_RET  = RNG.normal(0.001, 0.01, (T_TEST,  N_TICKERS)).astype(np.float32)

# Simple deterministic returns: all tickers return +0.01 every day
CONST_RET = np.full((T_TEST, N_TICKERS), 0.01, dtype=np.float32)

# Dates aligned with TEST_RET
TEST_DATES = np.array(
    [np.datetime64("2024-01-01") + np.timedelta64(i, "D") for i in range(T_TEST)]
)


# ---------------------------------------------------------------------------
# run_equal_weight_buyhold
# ---------------------------------------------------------------------------

class TestEqualWeightBuyhold:

    def test_output_is_tuple_of_two_arrays(self):
        result = run_equal_weight_buyhold(TEST_RET)
        assert isinstance(result, tuple) and len(result) == 2

    def test_returns_shape(self):
        port_returns, _ = run_equal_weight_buyhold(TEST_RET)
        assert port_returns.shape == (T_TEST,)

    def test_weights_shape(self):
        _, weights = run_equal_weight_buyhold(TEST_RET)
        assert weights.shape == (T_TEST, N_TICKERS)

    def test_initial_weights_are_equal(self):
        """First-step weights should be exactly 1/n."""
        _, weights = run_equal_weight_buyhold(TEST_RET)
        expected = 1.0 / N_TICKERS
        np.testing.assert_allclose(weights[0], expected, atol=1e-6)

    def test_weights_sum_to_one(self):
        _, weights = run_equal_weight_buyhold(TEST_RET)
        row_sums = weights.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_weights_non_negative(self):
        _, weights = run_equal_weight_buyhold(TEST_RET)
        assert (weights >= 0).all()

    def test_weights_drift_with_unequal_returns(self):
        """With variable returns, weights must drift away from 1/n over time."""
        _, weights = run_equal_weight_buyhold(TEST_RET)
        # After 60 steps with random returns, final weights should differ from initial
        assert not np.allclose(weights[0], weights[-1], atol=1e-6)

    def test_returns_correct_for_constant_return_series(self):
        """With all tickers returning +r every day, portfolio return == r."""
        port_returns, _ = run_equal_weight_buyhold(CONST_RET)
        np.testing.assert_allclose(port_returns, 0.01, atol=1e-6)

    def test_returns_correct_for_zero_return_series(self):
        zero_ret = np.zeros((T_TEST, N_TICKERS), dtype=np.float32)
        port_returns, _ = run_equal_weight_buyhold(zero_ret)
        np.testing.assert_allclose(port_returns, 0.0, atol=1e-10)

    def test_returns_match_manual_calculation(self):
        """port_return[t] == sum(weights[t] * TEST_RET[t]) with drifting weights."""
        port_returns, weights = run_equal_weight_buyhold(TEST_RET)
        expected = (weights * TEST_RET).sum(axis=1)
        np.testing.assert_allclose(port_returns, expected, atol=1e-6)

    def test_returns_dtype_float32(self):
        port_returns, weights = run_equal_weight_buyhold(TEST_RET)
        assert port_returns.dtype == np.float32 or port_returns.dtype == np.float64
        assert weights.dtype == np.float32

    def test_single_step_episode(self):
        one_step = np.array([[0.05, -0.02, 0.01, 0.03, -0.01]], dtype=np.float32)
        port_returns, weights = run_equal_weight_buyhold(one_step)
        assert port_returns.shape == (1,)
        assert weights.shape == (1, 5)
        expected_ret = np.mean([0.05, -0.02, 0.01, 0.03, -0.01])
        np.testing.assert_allclose(port_returns[0], expected_ret, atol=1e-6)

    def test_no_cash_position(self):
        """The function takes n_tickers return columns only — all weight goes to stocks."""
        _, weights = run_equal_weight_buyhold(TEST_RET)
        assert weights.shape[1] == N_TICKERS
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-5)

    def test_weight_drift_formula(self):
        """Verify w[t+1] = w[t] * (1 + r[t]) / sum(w[t] * (1 + r[t]))."""
        _, weights = run_equal_weight_buyhold(TEST_RET)
        for t in range(min(10, T_TEST - 1)):
            raw = weights[t] * (1.0 + TEST_RET[t])
            expected_next = raw / raw.sum()
            np.testing.assert_allclose(weights[t + 1], expected_next, atol=1e-5)

    def test_portfolio_return_uses_beginning_of_period_weights(self):
        """port_return[t] = sum(w[t] * r[t]) where w[t] is BOD weight, not EOD."""
        port_returns, weights = run_equal_weight_buyhold(TEST_RET)
        for t in range(min(10, T_TEST)):
            expected = np.dot(weights[t], TEST_RET[t])
            np.testing.assert_allclose(port_returns[t], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Metric helpers (smoke tests)
# ---------------------------------------------------------------------------

class TestMetricHelpers:

    def test_annualised_return_positive_for_growing_series(self):
        daily = np.full(252, 0.001, dtype=np.float64)
        assert _annualised_return(daily) > 0

    def test_annualised_return_zero_for_flat_series(self):
        daily = np.zeros(252, dtype=np.float64)
        assert abs(_annualised_return(daily)) < 1e-10

    def test_annualised_sharpe_zero_for_flat(self):
        daily = np.zeros(100, dtype=np.float64)
        assert _annualised_sharpe(daily) == 0.0

    def test_max_drawdown_zero_for_monotone_increasing(self):
        daily = np.full(50, 0.002, dtype=np.float64)
        assert _max_drawdown(daily) < 1e-10

    def test_max_drawdown_between_zero_and_one(self):
        rng = np.random.default_rng(0)
        daily = rng.normal(0.0, 0.02, 252)
        mdd = _max_drawdown(daily)
        assert 0.0 <= mdd <= 1.0

    def test_mean_daily_turnover_zero_for_static_weights(self):
        weights = np.tile([0.2, 0.2, 0.2, 0.2, 0.2], (50, 1)).astype(np.float32)
        assert _mean_daily_turnover(weights) < 1e-10

    def test_mean_daily_turnover_positive_for_changing_weights(self):
        rng = np.random.default_rng(1)
        weights = np.abs(rng.normal(0, 1, (50, 5)));  weights /= weights.sum(axis=1, keepdims=True)
        assert _mean_daily_turnover(weights) > 0


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_returns_dict_with_all_keys(self):
        port_returns, weights = run_equal_weight_buyhold(TEST_RET)
        m = compute_metrics(port_returns, weights, label="test")
        for key in ["label", "annualised_return", "sharpe", "sortino",
                    "max_drawdown", "calmar", "mean_daily_turnover"]:
            assert key in m, f"Missing key: {key}"

    def test_label_is_preserved(self):
        port_returns, weights = run_equal_weight_buyhold(TEST_RET)
        m = compute_metrics(port_returns, weights, label="equal_weight_buyhold")
        assert m["label"] == "equal_weight_buyhold"

    def test_max_drawdown_in_valid_range(self):
        port_returns, weights = run_equal_weight_buyhold(TEST_RET)
        m = compute_metrics(port_returns, weights)
        assert 0.0 <= m["max_drawdown"] <= 1.0

    def test_turnover_low_for_buyhold(self):
        """Buy-and-hold has small natural turnover from drift only (no rebalancing)."""
        port_returns, weights = run_equal_weight_buyhold(TEST_RET)
        m = compute_metrics(port_returns, weights, label="eq_bh")
        # Drift causes small positive turnover, but no large rebalancing events
        assert m["mean_daily_turnover"] > 0
        assert m["mean_daily_turnover"] < 0.05


# ---------------------------------------------------------------------------
# Integration: buyhold vs monthly — turnover comparison
# ---------------------------------------------------------------------------

class TestBaselinesIntegration:

    def test_buyhold_turnover_lower_than_monthly(self):
        """
        Buy-and-hold (constant weights) must have lower turnover than
        monthly rebalance (which resets to equal weights monthly).
        """
        _, bh_weights  = run_equal_weight_buyhold(TEST_RET)
        _, mly_weights = run_equal_weight_monthly(TEST_RET, TEST_DATES)
        bh_to  = _mean_daily_turnover(bh_weights)
        mly_to = _mean_daily_turnover(mly_weights)
        assert bh_to <= mly_to, (
            f"Expected buyhold turnover ({bh_to:.4f}) <= monthly ({mly_to:.4f})"
        )

    def test_all_three_baselines_return_valid_metrics(self):
        """Smoke test: run all three baselines end-to-end without errors."""
        bh_ret,  bh_w  = run_equal_weight_buyhold(TEST_RET)
        mly_ret, mly_w = run_equal_weight_monthly(TEST_RET, TEST_DATES)
        ms_ret,  ms_w  = run_max_sharpe(TRAIN_RET, TEST_RET)

        for rets, weights, label in [
            (bh_ret,  bh_w,  "equal_weight_buyhold"),
            (mly_ret, mly_w, "equal_weight_monthly"),
            (ms_ret,  ms_w,  "max_sharpe"),
        ]:
            m = compute_metrics(rets, weights, label=label)
            assert m["label"] == label
            assert not np.isnan(m["sharpe"])
            assert not np.isnan(m["max_drawdown"])


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCi:

    def test_returns_tuple_of_two_floats(self):
        returns = TEST_RET.mean(axis=1)
        lower, upper = bootstrap_ci(returns, _annualised_sharpe, n_bootstrap=100)
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_lower_leq_upper(self):
        returns = TEST_RET.mean(axis=1)
        lower, upper = bootstrap_ci(returns, _annualised_sharpe, n_bootstrap=1000)
        assert lower <= upper

    def test_ci_wider_for_volatile_series(self):
        """High-volatility returns should have wider CI than low-volatility."""
        rng = np.random.default_rng(99)
        low_vol = rng.normal(0.001, 0.005, 252)
        high_vol = rng.normal(0.001, 0.05, 252)
        lo_l, lo_u = bootstrap_ci(low_vol, _annualised_sharpe, n_bootstrap=5000, seed=0)
        hi_l, hi_u = bootstrap_ci(high_vol, _annualised_sharpe, n_bootstrap=5000, seed=0)
        width_low = lo_u - lo_l
        width_high = hi_u - hi_l
        assert width_high > width_low

    def test_ci_narrows_with_more_data(self):
        """1000-day series should have narrower CI than 50-day series."""
        rng = np.random.default_rng(7)
        short = rng.normal(0.001, 0.01, 50)
        long = rng.normal(0.001, 0.01, 1000)
        s_l, s_u = bootstrap_ci(short, _annualised_sharpe, n_bootstrap=5000, seed=1)
        l_l, l_u = bootstrap_ci(long, _annualised_sharpe, n_bootstrap=5000, seed=1)
        assert (l_u - l_l) < (s_u - s_l)

    def test_deterministic_with_seed(self):
        returns = TEST_RET.mean(axis=1)
        r1 = bootstrap_ci(returns, _annualised_sharpe, n_bootstrap=500, seed=42)
        r2 = bootstrap_ci(returns, _annualised_sharpe, n_bootstrap=500, seed=42)
        assert r1[0] == r2[0]
        assert r1[1] == r2[1]

    def test_zero_returns_gives_zero_sharpe(self):
        zeros = np.zeros(100)
        lower, upper = bootstrap_ci(zeros, _annualised_sharpe, n_bootstrap=100)
        assert lower == 0.0
        assert upper == 0.0

    def test_works_with_sortino(self):
        returns = TEST_RET.mean(axis=1)
        lower, upper = bootstrap_ci(returns, _annualised_sortino, n_bootstrap=100)
        assert isinstance(lower, float) and isinstance(upper, float)

    def test_works_with_annualised_return(self):
        returns = TEST_RET.mean(axis=1)
        lower, upper = bootstrap_ci(returns, _annualised_return, n_bootstrap=100)
        assert isinstance(lower, float) and isinstance(upper, float)


# ---------------------------------------------------------------------------
# _run_sixty_forty_from_returns (60/40 SPY/AGG inner)
# ---------------------------------------------------------------------------

class TestSixtyFortyFromReturns:

    def test_output_is_tuple_of_two_arrays(self):
        spy = np.full(T_TEST, 0.001, dtype=np.float32)
        agg = np.full(T_TEST, 0.0003, dtype=np.float32)
        result = _run_sixty_forty_from_returns(spy, agg, TEST_DATES)
        assert isinstance(result, tuple) and len(result) == 2

    def test_returns_shape(self):
        spy = np.full(T_TEST, 0.001, dtype=np.float32)
        agg = np.full(T_TEST, 0.0003, dtype=np.float32)
        port_returns, _ = _run_sixty_forty_from_returns(spy, agg, TEST_DATES)
        assert port_returns.shape == (T_TEST,)

    def test_weights_shape(self):
        spy = np.full(T_TEST, 0.001, dtype=np.float32)
        agg = np.full(T_TEST, 0.0003, dtype=np.float32)
        _, weights = _run_sixty_forty_from_returns(spy, agg, TEST_DATES)
        assert weights.shape == (T_TEST, 2)

    def test_initial_weights_sixty_forty(self):
        """Initial weights must be [0.6, 0.4]."""
        spy = np.full(T_TEST, 0.001, dtype=np.float32)
        agg = np.full(T_TEST, 0.0003, dtype=np.float32)
        _, weights = _run_sixty_forty_from_returns(spy, agg, TEST_DATES)
        np.testing.assert_allclose(weights[0], [0.6, 0.4], atol=1e-6)

    def test_weights_sum_to_one(self):
        spy = np.full(T_TEST, 0.001, dtype=np.float32)
        agg = np.full(T_TEST, 0.0003, dtype=np.float32)
        _, weights = _run_sixty_forty_from_returns(spy, agg, TEST_DATES)
        row_sums = weights.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_weights_drift_between_rebalances(self):
        """With unequal returns, weights should drift from 60/40 within a month."""
        rng = np.random.default_rng(0)
        spy = rng.normal(0.005, 0.01, T_TEST).astype(np.float32)
        agg = rng.normal(0.0003, 0.003, T_TEST).astype(np.float32)
        _, weights = _run_sixty_forty_from_returns(spy, agg, TEST_DATES)
        # Day 15 (within first month) should differ from 60/40
        assert not np.allclose(weights[15], [0.6, 0.4], atol=1e-3)

    def test_monthly_rebalance_resets_to_sixty_forty(self):
        """At month boundary, weights should reset to [0.6, 0.4]."""
        # Dates: Jan 1 through Mar 1 (to cross a month boundary)
        dates = np.array([
            np.datetime64("2024-01-01") + np.timedelta64(i, "D")
            for i in range(60)
        ])
        rng = np.random.default_rng(5)
        spy = rng.normal(0.005, 0.02, 60).astype(np.float32)
        agg = rng.normal(0.0003, 0.005, 60).astype(np.float32)
        _, weights = _run_sixty_forty_from_returns(spy, agg, dates)
        # Find the first day of February (day index 31)
        feb_start_idx = 31  # Feb 1 is day 31 (0-indexed)
        np.testing.assert_allclose(weights[feb_start_idx], [0.6, 0.4], atol=1e-5)

    def test_returns_correct_for_constant_series(self):
        """With constant spy=0.01 and agg=0.005, return = 0.6*0.01 + 0.4*0.005 = 0.008."""
        spy = np.full(T_TEST, 0.01, dtype=np.float32)
        agg = np.full(T_TEST, 0.005, dtype=np.float32)
        port_returns, _ = _run_sixty_forty_from_returns(spy, agg, TEST_DATES)
        np.testing.assert_allclose(port_returns[0], 0.008, atol=1e-5)

    def test_returns_zero_for_zero_returns(self):
        spy = np.zeros(T_TEST, dtype=np.float32)
        agg = np.zeros(T_TEST, dtype=np.float32)
        port_returns, _ = _run_sixty_forty_from_returns(spy, agg, TEST_DATES)
        np.testing.assert_allclose(port_returns, 0.0, atol=1e-10)

    def test_compute_metrics_integration(self):
        """Smoke test: compute_metrics works with 60/40 SPY/AGG outputs."""
        spy = np.full(T_TEST, 0.001, dtype=np.float32)
        agg = np.full(T_TEST, 0.0003, dtype=np.float32)
        port_returns, weights = _run_sixty_forty_from_returns(spy, agg, TEST_DATES)
        m = compute_metrics(port_returns, weights, label="sixty_forty")
        assert m["label"] == "sixty_forty"
        assert not np.isnan(m["sharpe"])
