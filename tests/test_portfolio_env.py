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
        np.testing.assert_allclose(env.action_space.low, -20.0, atol=1e-6)
        np.testing.assert_allclose(env.action_space.high, 20.0, atol=1e-6)


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
        for _ in range(8):  # T-1 = 9 episode steps; first 8 are non-terminal
            _, _, terminated, _, _ = env.step(env.action_space.sample())
            assert not terminated

    def test_terminated_at_last_step(self):
        env = _make_env(T=10)
        env.reset()
        terminated = False
        for _ in range(9):  # episode length = T-1 = 9
            _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert terminated

    def test_truncated_always_false(self):
        env = _make_env()
        env.reset()
        _, _, _, truncated, _ = env.step(env.action_space.sample())
        assert not truncated

    def test_reward_formula(self):
        """
        Verify reward formula in both regimes:
          - warm-up (buffer_count < vol_window): reward = port_return - lambda * turnover
          - steady-state (buffer >= vol_window): reward = port_return / (vol + eps) - lambda * turnover
        """
        vol_window = 21
        T_ = vol_window + 5  # enough steps to exit warm-up
        obs_array    = np.ones((T_, OBS_DIM), dtype=np.float32)
        return_array = np.full((T_, N_TICKERS), 0.01, dtype=np.float32)

        env = PortfolioEnv(obs_array, return_array, _make_config())
        env.reset()

        # Step 0: warm-up regime (buffer_count=1 < vol_window=21)
        action = np.full(ACTION_DIM, 0.5, dtype=np.float32)
        _, reward, _, _, info = env.step(action)

        port_return = info["portfolio_return"]
        turnover    = info["turnover"]
        lambda_tc   = _make_config()["env"]["transaction_cost"]

        expected_warmup = port_return - lambda_tc * turnover
        np.testing.assert_allclose(reward, expected_warmup, atol=1e-6)

        # Run through warm-up into steady-state
        for _ in range(vol_window):
            _, reward, _, _, info = env.step(action)

        # Now buffer is full — should use Sharpe-normalised formula
        port_return = info["portfolio_return"]
        roll_vol    = info["rolling_vol"]
        turnover    = info["turnover"]
        vol_eps     = _make_config()["env"]["vol_eps"]

        expected_steady = port_return / (roll_vol + vol_eps) - lambda_tc * turnover
        np.testing.assert_allclose(reward, expected_steady, atol=1e-6)

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
        for key in ["portfolio_return", "rolling_vol", "turnover", "weights", "step",
                     "current_drawdown"]:
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


# ---------------------------------------------------------------------------
# Drawdown penalty
# ---------------------------------------------------------------------------

def _make_config_with_dd(lambda_dd: float = 0.5, dd_threshold: float = 0.05) -> dict:
    cfg = _make_config()
    cfg["env"]["lambda_dd"] = lambda_dd
    cfg["env"]["dd_threshold"] = dd_threshold
    return cfg


def _make_env_with_dd(T: int = T, seed: int = 0, lambda_dd: float = 0.5,
                       dd_threshold: float = 0.05) -> PortfolioEnv:
    rng = np.random.default_rng(seed)
    obs_array = rng.standard_normal((T, OBS_DIM)).astype(np.float32)
    return_array = rng.normal(0.001, 0.01, (T, N_TICKERS)).astype(np.float32)
    return PortfolioEnv(obs_array, return_array,
                        _make_config_with_dd(lambda_dd, dd_threshold))


class TestDrawdownPenalty:

    def test_drawdown_zero_when_portfolio_rising(self):
        """With constant positive returns, drawdown should be 0.0."""
        T_ = 10
        obs_array = np.ones((T_, OBS_DIM), dtype=np.float32)
        return_array = np.full((T_, N_TICKERS), 0.01, dtype=np.float32)
        env = PortfolioEnv(obs_array, return_array,
                           _make_config_with_dd(lambda_dd=1.0, dd_threshold=0.0))
        env.reset()
        action = np.full(ACTION_DIM, 0.5, dtype=np.float32)
        for _ in range(T_ - 1):
            _, _, _, _, info = env.step(action)
            np.testing.assert_allclose(info["current_drawdown"], 0.0, atol=1e-8)

    def test_drawdown_positive_after_loss(self):
        """After a negative return, drawdown should become positive."""
        T_ = 5
        obs_array = np.ones((T_, OBS_DIM), dtype=np.float32)
        return_array = np.zeros((T_, N_TICKERS), dtype=np.float32)
        return_array[0] = 0.05   # first step: gain
        return_array[1] = -0.10  # second step: loss
        env = PortfolioEnv(obs_array, return_array,
                           _make_config_with_dd(lambda_dd=1.0, dd_threshold=0.0))
        env.reset()
        action = np.full(ACTION_DIM, 0.5, dtype=np.float32)
        env.step(action)  # gain step
        _, _, _, _, info = env.step(action)  # loss step
        assert info["current_drawdown"] > 0.0

    def test_dd_penalty_zero_when_lambda_dd_zero(self):
        """With lambda_dd=0, reward should match original formula (no dd penalty)."""
        T_ = 10
        obs_array = np.ones((T_, OBS_DIM), dtype=np.float32)
        return_array = np.zeros((T_, N_TICKERS), dtype=np.float32)
        return_array[0] = 0.05
        return_array[1] = -0.10

        env_no_dd = PortfolioEnv(obs_array, return_array, _make_config())
        env_with_dd = PortfolioEnv(obs_array.copy(), return_array.copy(),
                                    _make_config_with_dd(lambda_dd=0.0))
        env_no_dd.reset()
        env_with_dd.reset()
        action = np.full(ACTION_DIM, 0.5, dtype=np.float32)
        for _ in range(3):
            _, r1, _, _, _ = env_no_dd.step(action)
            _, r2, _, _, _ = env_with_dd.step(action)
            np.testing.assert_allclose(r1, r2, atol=1e-8)

    def test_dd_penalty_lowers_reward_during_drawdown(self):
        """With lambda_dd > 0, reward should be lower during drawdown than without."""
        T_ = 5
        obs_array = np.ones((T_, OBS_DIM), dtype=np.float32)
        return_array = np.zeros((T_, N_TICKERS), dtype=np.float32)
        return_array[0] = 0.05
        return_array[1] = -0.10

        env_no_dd = PortfolioEnv(obs_array, return_array.copy(), _make_config())
        env_with_dd = PortfolioEnv(obs_array.copy(), return_array.copy(),
                                    _make_config_with_dd(lambda_dd=1.0, dd_threshold=0.0))
        env_no_dd.reset()
        env_with_dd.reset()
        action = np.full(ACTION_DIM, 0.5, dtype=np.float32)
        env_no_dd.step(action)     # gain step
        env_with_dd.step(action)   # gain step
        _, r_no, _, _, _ = env_no_dd.step(action)     # loss step
        _, r_dd, _, _, _ = env_with_dd.step(action)   # loss step
        assert r_dd < r_no

    def test_dd_threshold_prevents_small_dd_penalty(self):
        """With dd_threshold=0.20, a 10% drawdown should incur no penalty."""
        T_ = 5
        obs_array = np.ones((T_, OBS_DIM), dtype=np.float32)
        return_array = np.zeros((T_, N_TICKERS), dtype=np.float32)
        return_array[0] = 0.05
        return_array[1] = -0.10  # ~10% drawdown (below 20% threshold)

        env_no_dd = PortfolioEnv(obs_array, return_array.copy(), _make_config())
        env_threshold = PortfolioEnv(obs_array.copy(), return_array.copy(),
                                      _make_config_with_dd(lambda_dd=1.0, dd_threshold=0.20))
        env_no_dd.reset()
        env_threshold.reset()
        action = np.full(ACTION_DIM, 0.5, dtype=np.float32)
        env_no_dd.step(action)
        env_threshold.step(action)
        _, r_no, _, _, _ = env_no_dd.step(action)
        _, r_thresh, _, _, _ = env_threshold.step(action)
        # With threshold above actual drawdown, reward should match no-penalty env
        np.testing.assert_allclose(r_no, r_thresh, atol=1e-8)

    def test_reset_clears_drawdown_state(self):
        """After reset, drawdown tracking should start fresh."""
        env = _make_env_with_dd(T=10, lambda_dd=1.0, dd_threshold=0.0)
        env.reset()
        action = np.full(ACTION_DIM, 0.5, dtype=np.float32)
        for _ in range(5):
            env.step(action)
        env.reset()
        assert env._portfolio_value == 1.0
        assert env._peak_value == 1.0

    def test_drawdown_formula_matches_manual(self):
        """Verify drawdown = (peak - value) / peak with controlled returns."""
        T_ = 5
        obs_array = np.ones((T_, OBS_DIM), dtype=np.float32)
        # Equal-weight portfolio of 5 tickers + cash (uniform softmax)
        # with 0.5 logits → softmax → ~1/6 each
        # Only ticker returns matter; cash returns 0
        return_array = np.zeros((T_, N_TICKERS), dtype=np.float32)
        return_array[0] = 0.06   # all tickers +6%
        return_array[1] = -0.12  # all tickers -12%

        env = PortfolioEnv(obs_array, return_array,
                           _make_config_with_dd(lambda_dd=0.0))
        env.reset()
        action = np.full(ACTION_DIM, 0.5, dtype=np.float32)

        # Step 0: gain
        _, _, _, _, info0 = env.step(action)
        # Compute expected portfolio return for uniform-ish weights
        w = PortfolioEnv._softmax(action)
        port_ret_0 = float(np.dot(w[:N_TICKERS], return_array[0]))
        expected_nav = 1.0 * (1.0 + port_ret_0)
        assert info0["current_drawdown"] == 0.0  # new high

        # Step 1: loss
        _, _, _, _, info1 = env.step(action)
        port_ret_1 = float(np.dot(w[:N_TICKERS], return_array[1]))
        expected_nav *= (1.0 + port_ret_1)
        expected_peak = 1.0 * (1.0 + port_ret_0)  # peak was after step 0
        expected_dd = (expected_peak - expected_nav) / expected_peak
        np.testing.assert_allclose(info1["current_drawdown"], expected_dd, atol=1e-6)
