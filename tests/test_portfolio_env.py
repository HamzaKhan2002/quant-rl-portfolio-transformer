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
OBS_DIM    = 79
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
                     "current_drawdown", "weight_entropy", "div_bonus", "cvar_penalty"]:
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


# ---------------------------------------------------------------------------
# Constrained softmax (Phase 4, Task 1)
# ---------------------------------------------------------------------------

class TestConstrainedSoftmax:

    def test_unconstrained_matches_standard_softmax(self):
        """max_weight=1.0 should give identical output to standard softmax."""
        x = np.array([1.0, 2.0, -1.0, 0.5, 3.0, -2.0], dtype=np.float32)
        w_constrained = PortfolioEnv._constrained_softmax(x, max_weight=1.0)
        w_standard    = PortfolioEnv._softmax(x)
        np.testing.assert_allclose(w_constrained, w_standard, atol=1e-5)

    def test_output_sums_to_one(self):
        x = np.array([5.0, 1.0, 0.5, -1.0, 2.0, 4.0], dtype=np.float32)
        for cap in [0.25, 0.30, 0.35, 0.50]:
            w = PortfolioEnv._constrained_softmax(x, max_weight=cap)
            np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6,
                                       err_msg=f"sum ≠ 1 for cap={cap}")

    def test_no_weight_exceeds_cap(self):
        """After constrained projection, all weights must be ≤ max_weight."""
        x = np.array([10.0, -10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for cap in [0.20, 0.30, 0.35, 0.50]:
            w = PortfolioEnv._constrained_softmax(x, max_weight=cap)
            assert w.max() <= cap + 1e-6, (
                f"max weight {w.max():.6f} exceeds cap {cap} for x={x}"
            )

    def test_weights_non_negative(self):
        x = np.random.default_rng(42).normal(0, 5, 6).astype(np.float32)
        w = PortfolioEnv._constrained_softmax(x, max_weight=0.35)
        assert (w >= -1e-9).all()

    def test_output_dtype_float32(self):
        x = np.ones(6, dtype=np.float64)
        w = PortfolioEnv._constrained_softmax(x, max_weight=0.35)
        assert w.dtype == np.float32

    def test_tight_cap_produces_near_uniform(self):
        """With cap = 1/n, output should approach 1/n (uniform)."""
        n = 6
        cap = 1.0 / n
        x = np.array([10.0, -10.0, 0.0, 5.0, -5.0, 3.0], dtype=np.float32)
        w = PortfolioEnv._constrained_softmax(x, max_weight=cap)
        np.testing.assert_allclose(w, cap, atol=1e-5)

    def test_env_uses_constrained_projection(self):
        """After Task 1, no asset weight should exceed max_weight_per_ticker."""
        cap = 0.30
        cfg = _make_config()
        cfg["env"]["max_weight_per_ticker"] = cap
        rng = np.random.default_rng(7)
        obs_array    = rng.standard_normal((T, OBS_DIM)).astype(np.float32)
        return_array = rng.normal(0.001, 0.01, (T, N_TICKERS)).astype(np.float32)
        env = PortfolioEnv(obs_array, return_array, cfg)
        env.reset()
        # Use extreme logits to stress-test the constraint
        extreme_action = np.array([20.0, -20.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(extreme_action)
        assert info["weights"].max() <= cap + 1e-6

    def test_large_logit_overflow_safe(self):
        """Extreme logit values should not produce NaN or Inf."""
        x = np.array([1000.0, -1000.0, 500.0, -500.0, 0.0, 100.0], dtype=np.float32)
        w = PortfolioEnv._constrained_softmax(x, max_weight=0.35)
        assert not np.isnan(w).any()
        assert not np.isinf(w).any()


# ---------------------------------------------------------------------------
# Diversification entropy bonus (Phase 4, Task 1)
# ---------------------------------------------------------------------------

def _make_config_with_div(lambda_div: float = 0.1,
                           max_weight: float = 1.0) -> dict:
    cfg = _make_config()
    cfg["env"]["lambda_div"] = lambda_div
    cfg["env"]["max_weight_per_ticker"] = max_weight
    return cfg


class TestDiversificationBonus:

    def test_div_bonus_zero_when_lambda_div_zero(self):
        """lambda_div=0 → div_bonus = 0 at every step."""
        cfg = _make_config_with_div(lambda_div=0.0)
        rng = np.random.default_rng(1)
        obs_array    = rng.standard_normal((T, OBS_DIM)).astype(np.float32)
        return_array = rng.normal(0.001, 0.01, (T, N_TICKERS)).astype(np.float32)
        env = PortfolioEnv(obs_array, return_array, cfg)
        env.reset()
        for _ in range(5):
            _, _, _, _, info = env.step(env.action_space.sample())
            np.testing.assert_allclose(info["div_bonus"], 0.0, atol=1e-8)

    def test_div_bonus_higher_for_uniform_than_concentrated(self):
        """Uniform weights should earn a higher diversification bonus."""
        cfg = _make_config_with_div(lambda_div=0.1)
        rng = np.random.default_rng(2)
        obs_array    = rng.standard_normal((T, OBS_DIM)).astype(np.float32)
        return_array = rng.normal(0.001, 0.01, (T, N_TICKERS)).astype(np.float32)
        env = PortfolioEnv(obs_array, return_array, cfg)
        env.reset()

        # Uniform action → uniform-ish weights
        _, _, _, _, info_uniform = env.step(np.zeros(ACTION_DIM, dtype=np.float32))

        env.reset()
        # Concentrated action → nearly all weight on first asset
        concentrated = np.array([20.0, -20.0, -20.0, -20.0, -20.0, -20.0], dtype=np.float32)
        _, _, _, _, info_conc = env.step(concentrated)

        assert info_uniform["div_bonus"] > info_conc["div_bonus"]

    def test_div_bonus_adds_to_reward(self):
        """With lambda_div > 0, reward should be strictly higher than with lambda_div=0
        for the same action (for uniform-ish weights where entropy > 0)."""
        cfg_no  = _make_config_with_div(lambda_div=0.0)
        cfg_yes = _make_config_with_div(lambda_div=0.1)

        rng = np.random.default_rng(3)
        obs_array    = rng.standard_normal((T, OBS_DIM)).astype(np.float32)
        return_array = rng.normal(0.001, 0.01, (T, N_TICKERS)).astype(np.float32)

        env_no  = PortfolioEnv(obs_array.copy(), return_array.copy(), cfg_no)
        env_yes = PortfolioEnv(obs_array.copy(), return_array.copy(), cfg_yes)
        env_no.reset()
        env_yes.reset()

        action = np.zeros(ACTION_DIM, dtype=np.float32)  # uniform-ish
        _, r_no,  *_ = env_no.step(action)
        _, r_yes, *_ = env_yes.step(action)
        assert r_yes > r_no

    def test_weight_entropy_in_info(self):
        """weight_entropy should be in info dict and be non-negative."""
        cfg = _make_config_with_div(lambda_div=0.1)
        rng = np.random.default_rng(4)
        obs_array    = rng.standard_normal((T, OBS_DIM)).astype(np.float32)
        return_array = rng.normal(0.001, 0.01, (T, N_TICKERS)).astype(np.float32)
        env = PortfolioEnv(obs_array, return_array, cfg)
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "weight_entropy" in info
        assert info["weight_entropy"] >= 0.0


# ---------------------------------------------------------------------------
# CVaR Lagrangian interface (Phase 4, Task 3 integration tests)
# ---------------------------------------------------------------------------

class TestCVaRInterface:

    def test_set_cvar_params_updates_env(self):
        """set_cvar_params should update lambda_cvar and var_alpha on the env."""
        env = _make_env()
        env.set_cvar_params(lambda_cvar=0.5, var_alpha=-0.02)
        assert env.lambda_cvar == 0.5
        assert env.var_alpha == -0.02

    def test_cvar_penalty_zero_when_lambda_zero(self):
        """With lambda_cvar=0, cvar_penalty should always be 0."""
        env = _make_env()
        env.reset()
        env.set_cvar_params(lambda_cvar=0.0, var_alpha=-0.01)
        for _ in range(5):
            _, _, _, _, info = env.step(env.action_space.sample())
            assert info["cvar_penalty"] == 0.0

    def test_cvar_penalty_nonzero_for_large_losses(self):
        """With large lambda and VaR near 0, tail losses incur a penalty."""
        T_ = 5
        obs_array = np.ones((T_, OBS_DIM), dtype=np.float32)
        return_array = np.full((T_, N_TICKERS), -0.05, dtype=np.float32)  # all losses

        cfg = _make_config()
        cfg["env"]["lambda_cvar_init"] = 2.0
        cfg["env"]["var_alpha_init"] = 0.0  # VaR = 0 → penalty fires on any loss
        env = PortfolioEnv(obs_array, return_array, cfg)
        env.reset()
        action = np.full(ACTION_DIM, 0.5, dtype=np.float32)
        for _ in range(T_ - 1):
            _, _, _, _, info = env.step(action)
            assert info["cvar_penalty"] > 0.0
