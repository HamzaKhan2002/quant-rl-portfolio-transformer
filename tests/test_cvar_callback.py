"""
tests/test_cvar_callback.py — Unit tests for src/cvar_callback.py.

Tests use only synthetic data — no network access, no upstream CSV files.
Stable-baselines3 is required but no actual model training is done.
"""

from __future__ import annotations

import numpy as np
import pytest

sb3 = pytest.importorskip("stable_baselines3")

from src.cvar_callback import CVaRLagrangianCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> dict:
    base = {
        "env": {
            "cvar_alpha":       0.05,
            "cvar_threshold":   0.02,
            "cvar_lr":          0.001,
            "lambda_cvar_init": 0.0,
            "var_alpha_init":   -100.0,
        }
    }
    base["env"].update(overrides)
    return base


def _make_callback(**cfg_overrides) -> CVaRLagrangianCallback:
    return CVaRLagrangianCallback(_make_config(**cfg_overrides), verbose=0)


def _fill_buffer(cb: CVaRLagrangianCallback, returns: list) -> None:
    """Directly fill the callback's internal buffer."""
    for r in returns:
        cb._buffer.append(r)


# ---------------------------------------------------------------------------
# Tests: Initialisation
# ---------------------------------------------------------------------------

class TestCVaRCallbackInit:
    def test_initial_lambda_is_zero(self):
        cb = _make_callback()
        assert cb.lambda_cvar == 0.0

    def test_initial_var_alpha_is_minus_100(self):
        cb = _make_callback()
        assert cb.var_alpha == -100.0

    def test_config_overrides_alpha(self):
        cb = _make_callback(cvar_alpha=0.01)
        assert cb.cvar_alpha == 0.01

    def test_config_overrides_threshold(self):
        cb = _make_callback(cvar_threshold=0.03)
        assert cb.cvar_threshold == 0.03

    def test_config_overrides_lr(self):
        cb = _make_callback(cvar_lr=0.01)
        assert cb.cvar_lr == 0.01

    def test_buffer_starts_empty(self):
        cb = _make_callback()
        assert len(cb._buffer) == 0


# ---------------------------------------------------------------------------
# Tests: compute_cvar static method
# ---------------------------------------------------------------------------

class TestComputeCVaR:
    def test_var_alpha_is_alpha_percentile(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0, 0.01, 1000)
        var, cvar = CVaRLagrangianCallback.compute_cvar(r, alpha=0.05)
        expected_var = np.percentile(r, 5)
        assert abs(var - expected_var) < 1e-10

    def test_cvar_leq_var(self):
        """CVaR (tail mean) must be ≤ VaR (tail quantile)."""
        rng = np.random.default_rng(1)
        r = rng.normal(0, 0.01, 500)
        var, cvar = CVaRLagrangianCallback.compute_cvar(r, alpha=0.05)
        assert cvar <= var + 1e-10

    def test_cvar_strictly_less_for_fat_tail(self):
        """For a fat-tailed loss distribution, CVaR < VaR."""
        rng = np.random.default_rng(2)
        normal_part = rng.normal(0.001, 0.005, 450)
        tail_part   = rng.normal(-0.05, 0.01, 50)
        r = np.concatenate([normal_part, tail_part])
        var, cvar = CVaRLagrangianCallback.compute_cvar(r, alpha=0.05)
        assert cvar < var

    def test_empty_returns(self):
        var, cvar = CVaRLagrangianCallback.compute_cvar(np.array([]), alpha=0.05)
        assert var == 0.0
        assert cvar == 0.0

    def test_all_positive_returns(self):
        r = np.full(100, 0.001)
        var, cvar = CVaRLagrangianCallback.compute_cvar(r, alpha=0.05)
        # All returns are equal → VaR and CVaR are the same
        assert cvar <= var + 1e-10

    def test_returns_are_floats(self):
        r = np.random.default_rng(99).normal(0, 0.01, 100)
        var, cvar = CVaRLagrangianCallback.compute_cvar(r, alpha=0.05)
        assert isinstance(var, float)
        assert isinstance(cvar, float)


# ---------------------------------------------------------------------------
# Tests: dual_ascent_update
# ---------------------------------------------------------------------------

class TestDualAscentUpdate:
    def test_lambda_increases_when_cvar_exceeds_threshold(self):
        """If CVaR_loss > τ, λ should increase."""
        cb = _make_callback(cvar_threshold=0.005, cvar_lr=0.01)
        cb._push_cvar_params = lambda: None  # stub out env_method
        # Fill buffer with large losses (mean tail ≈ -0.05)
        rng = np.random.default_rng(0)
        returns = np.concatenate([
            rng.normal(0.001, 0.005, 400),
            np.full(100, -0.05),  # 5th-pct tail is large loss
        ])
        _fill_buffer(cb, list(returns))

        initial_lambda = cb.lambda_cvar
        cb._dual_ascent_update()
        assert cb.lambda_cvar > initial_lambda

    def test_lambda_decreases_when_cvar_below_threshold(self):
        """If CVaR_loss < τ but λ > 0, λ should decrease toward 0."""
        cb = _make_callback(cvar_threshold=0.10, cvar_lr=0.001)
        cb.lambda_cvar = 0.5  # start with a positive multiplier
        cb._push_cvar_params = lambda: None
        # Fill buffer with very small losses (CVaR_loss << 0.10)
        rng = np.random.default_rng(1)
        returns = rng.normal(0.005, 0.001, 500)  # consistently positive
        _fill_buffer(cb, list(returns))

        initial_lambda = cb.lambda_cvar
        cb._dual_ascent_update()
        assert cb.lambda_cvar < initial_lambda

    def test_lambda_stays_nonnegative(self):
        """λ is projected to ≥ 0 (cannot go negative)."""
        cb = _make_callback(cvar_threshold=1.0, cvar_lr=1.0)
        cb._push_cvar_params = lambda: None
        rng = np.random.default_rng(2)
        returns = rng.normal(0.001, 0.005, 500)
        _fill_buffer(cb, list(returns))

        cb._dual_ascent_update()
        assert cb.lambda_cvar >= 0.0

    def test_var_alpha_is_updated(self):
        """VaR estimate should change after a dual-ascent update."""
        cb = _make_callback()
        cb._push_cvar_params = lambda: None
        rng = np.random.default_rng(3)
        returns = rng.normal(0.001, 0.01, 300)
        _fill_buffer(cb, list(returns))

        old_var = cb.var_alpha  # starts at -100.0
        cb._dual_ascent_update()
        assert cb.var_alpha != old_var

    def test_var_alpha_matches_empirical_percentile(self):
        cb = _make_callback(cvar_alpha=0.10)
        cb._push_cvar_params = lambda: None
        rng = np.random.default_rng(4)
        returns = rng.normal(0.0, 0.01, 500)
        _fill_buffer(cb, list(returns))

        cb._dual_ascent_update()
        expected_var = float(np.percentile(list(cb._buffer), 10))
        assert abs(cb.var_alpha - expected_var) < 1e-10


# ---------------------------------------------------------------------------
# Tests: buffer management
# ---------------------------------------------------------------------------

class TestBufferManagement:
    def test_buffer_size_respects_maxlen(self):
        cb = CVaRLagrangianCallback(_make_config(), cvar_buffer_size=50)
        for i in range(200):
            cb._buffer.append(float(i))
        assert len(cb._buffer) == 50

    def test_buffer_keeps_most_recent(self):
        cb = CVaRLagrangianCallback(_make_config(), cvar_buffer_size=5)
        for i in range(10):
            cb._buffer.append(float(i))
        assert list(cb._buffer) == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_get_cvar_params_returns_expected_keys(self):
        cb = _make_callback()
        params = cb.get_cvar_params()
        assert "lambda_cvar" in params
        assert "var_alpha" in params
        assert "buffer_size" in params

    def test_get_cvar_params_buffer_size_accurate(self):
        cb = _make_callback()
        _fill_buffer(cb, [0.001] * 30)
        assert cb.get_cvar_params()["buffer_size"] == 30


# ---------------------------------------------------------------------------
# Tests: on_step buffer collection (via locals injection)
# ---------------------------------------------------------------------------

class TestOnStepCollection:
    """Tests for the _on_step buffer-filling logic via locals injection."""

    def _make_callback_with_locals(self, infos):
        cb = _make_callback()
        cb.locals = {"infos": infos}
        cb._push_cvar_params = lambda: None  # stub out env interaction
        # Prevent actual dual-ascent by setting update_freq very high
        cb.update_freq = 10_000
        return cb

    def test_portfolio_return_added_to_buffer(self):
        cb = self._make_callback_with_locals(
            [{"portfolio_return": 0.005}, {"portfolio_return": -0.003}]
        )
        cb._on_step()
        assert len(cb._buffer) == 2

    def test_info_without_portfolio_return_is_skipped(self):
        cb = self._make_callback_with_locals(
            [{"some_other_key": 1.0}, {"portfolio_return": 0.002}]
        )
        cb._on_step()
        assert len(cb._buffer) == 1

    def test_on_step_returns_true(self):
        cb = self._make_callback_with_locals([{"portfolio_return": 0.001}])
        result = cb._on_step()
        assert result is True
