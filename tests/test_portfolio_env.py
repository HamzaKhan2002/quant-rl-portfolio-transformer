"""
tests/test_portfolio_env.py — Unit tests for src/portfolio_env.py.

All tests use synthetic numpy arrays.  No upstream data files or network
access required.  Gymnasium is the only non-standard dependency needed.
"""

from __future__ import annotations

import numpy as np
import pytest

# PortfolioEnv requires gymnasium; skip the whole module gracefully if absent
gymnasium = pytest.importorskip("gymnasium")

from src.portfolio_env import PortfolioEnv  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_TICKERS = 5
ACTION_DIM = 6   # 5 tickers + 1 cash
OBS_DIM    = 74
T          = 50  # number of time steps in synthetic episode


def _make_config() -> dict:
    return {
        "env": {
            "n_tickers":        N_TICKERS,
            "action_dim":       ACTION_DIM,
            "obs_dim":          OBS_DIM,
            "transaction_cost": 0.001,
            "vol_eps":          1e-8,
            "vol_window":       21,
        }
    }


def _make_env(T: int = T, seed: int = 0) -> PortfolioEnv:
    rng = np.random.default_rng(seed)
    obs_array     = rng.standard_normal((T, OBS_DIM)).astype(np.float32)
    return_array  = rng.normal(0.001, 0.01, (T, N_TICKERS)).astype(np.float32)
    return PortfolioEnv(obs_array, return_array, _make_config())


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------

class TestSpaces:
    def test_observation_space_shape(self):
        env = _make_env()
        assert env.observation_space.shape == (OBS_DIM,)

    def test_action_space_shape(self):
        env = _make_env()
        assert env.action_space.shape == (ACTION_DIM,)

    def test_observation_space_dtype(self):
        env = _make_env()
        assert env.observation_space.dtype == np.float32

    def test_action_space_bounds(self):
        env = _make_env()
        assert env.action_space.low[0]  == 0.0
        assert env.action_space.high[0] == 1.0


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_obs_of_correct_shape(self):
        env = _make_env()
        obs, info = env.reset()
        assert obs.shape == (OBS_DIM,)

    def test_reset_returns_dict_info(self):
        env = _make_env()
        _, info = env.reset()
        assert isinstance(info, dict)

    def test_reset_obs_matches_first_row(self):
        rng = np.random.default_rng(7)
        obs_array    = rng.standard_normal((T, OBS_DIM)).astype(np.float32)
        return_array = rng.normal(0, 0.01, (T, N_TICKERS)).astype(np.float32)
        env = PortfolioEnv(obs_array, return_array, _make_config())
        obs, _ = env.reset()
        np.testing.assert_array_equal(obs, obs_array[0])

    def test_reset_reinitialises_state(self):
        env = _make_env()
        env.reset()
        for _ in range(10):
            env.step(env.action_space.sample())
        env.reset()
        assert env._t == 0

    def test_initial_weights_equal_allocation(self):
        env = _make_env()
        env.reset()
        expected = 1.0 / N_TICKERS
        np.testing.assert_allclose(
            env._prev_weights[:N_TICKERS], expected, atol=1e-6
        )
        assert env._prev_weights[N_TICKERS] == 0.0  # cash


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_output_types(self):
        env = _make_env()
        env.reset()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_obs_shape(self):
        env = _make_env()
        env.reset()
        obs, *_ = env.step(env.action_space.sample())
        assert obs.shape == (OBS_DIM,)

    def test_weights_sum_to_one(self):
        env = _make_env()
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        np.testing.assert_allclose(info["weights"].sum(), 1.0, atol=1e-5)

    def test_weights_non_negative(self):
        env = _make_env()
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        assert (info["weights"] >= 0.0).all()

    def test_not_terminated_before_end(self):
        env = _make_env(T=10)
        env.reset()
        for _ in range(9):
            _, _, terminated, _, _ = env.step(env.action_space.sample())
            assert not terminated

    def test_terminated_at_last_step(self):
        env = _make_env(T=10)
        env.reset()
        terminated = False
        for _ in range(10):
            _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert terminated

    def test_truncated_always_false(self):
        env = _make_env()
        env.reset()
        _, _, _, truncated, _ = env.step(env.action_space.sample())
        assert not truncated

    def test_reward_formula(self):
        """
        Verify reward = port_return / (vol + eps) - lambda * turnover.
        Uses a deterministic single-step episode.
        """
        T_ = 2
        # Obs and returns — simple values
        obs_array    = np.ones((T_, OBS_DIM), dtype=np.float32)
        return_array = np.full((T_, N_TICKERS), 0.01, dtype=np.float32)

        env = PortfolioEnv(obs_array, return_array, _make_config())
        env.reset()

        # Apply uniform action [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] → uniform weights
        action = np.full(ACTION_DIM, 0.5, dtype=np.float32)
        _, reward, _, _, info = env.step(action)

        # weights should be uniform = 1/6
        expected_weights = np.full(ACTION_DIM, 1.0 / ACTION_DIM, dtype=np.float32)
        np.testing.assert_allclose(info["weights"], expected_weights, atol=1e-5)

        port_return  = info["portfolio_return"]
        roll_vol     = info["rolling_vol"]
        turnover     = info["turnover"]
        lambda_tc    = _make_config()["env"]["transaction_cost"]
        vol_eps      = _make_config()["env"]["vol_eps"]

        expected_reward = port_return / (roll_vol + vol_eps) - lambda_tc * turnover
        np.testing.assert_allclose(reward, expected_reward, atol=1e-6)

    def test_turnover_zero_when_weights_unchanged(self):
        """If action produces the same weights as prev step, turnover ≈ 0."""
        env = _make_env(T=5)
        env.reset()
        # Identical action each step
        fixed_action = np.full(ACTION_DIM, 0.5, dtype=np.float32)
        env.step(fixed_action)
        _, _, _, _, info = env.step(fixed_action)
        np.testing.assert_allclose(info["turnover"], 0.0, atol=1e-5)

    def test_portfolio_return_in_info(self):
        env = _make_env()
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "portfolio_return" in info
        assert isinstance(info["portfolio_return"], float)

    def test_info_contains_required_keys(self):
        env = _make_env()
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        for key in ["portfolio_return", "rolling_vol", "turnover", "weights", "step"]:
            assert key in info, f"Missing info key: {key}"

    def test_step_advances_time(self):
        env = _make_env()
        env.reset()
        assert env._t == 0
        env.step(env.action_space.sample())
        assert env._t == 1

    def test_obs_after_step_matches_next_row(self):
        rng = np.random.default_rng(99)
        obs_array    = rng.standard_normal((T, OBS_DIM)).astype(np.float32)
        return_array = rng.normal(0, 0.01, (T, N_TICKERS)).astype(np.float32)
        env = PortfolioEnv(obs_array, return_array, _make_config())
        env.reset()
        obs, _, _, _, _ = env.step(env.action_space.sample())
        np.testing.assert_array_equal(obs, obs_array[1])


# ---------------------------------------------------------------------------
# Softmax projection
# ---------------------------------------------------------------------------

class TestSoftmax:
    def test_output_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0, 0.5, -1.0, 4.0], dtype=np.float32)
        w = PortfolioEnv._softmax(x)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)

    def test_output_non_negative(self):
        x = np.array([-10.0, 0.0, 5.0, 3.0, -2.0, 1.0], dtype=np.float32)
        w = PortfolioEnv._softmax(x)
        assert (w >= 0.0).all()

    def test_output_dtype_float32(self):
        x = np.ones(6, dtype=np.float64)
        w = PortfolioEnv._softmax(x)
        assert w.dtype == np.float32

    def test_uniform_input_gives_uniform_output(self):
        x = np.zeros(ACTION_DIM, dtype=np.float32)
        w = PortfolioEnv._softmax(x)
        np.testing.assert_allclose(w, 1.0 / ACTION_DIM, atol=1e-6)

    def test_large_values_do_not_overflow(self):
        x = np.array([1000.0, -1000.0, 500.0, -500.0, 0.0, 100.0], dtype=np.float32)
        w = PortfolioEnv._softmax(x)
        assert not np.isnan(w).any()
        assert not np.isinf(w).any()


# ---------------------------------------------------------------------------
# Constructor guards
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_mismatched_array_lengths_raise(self):
        obs    = np.zeros((10, OBS_DIM), dtype=np.float32)
        ret    = np.zeros((15, N_TICKERS), dtype=np.float32)
        with pytest.raises(AssertionError):
            PortfolioEnv(obs, ret, _make_config())
