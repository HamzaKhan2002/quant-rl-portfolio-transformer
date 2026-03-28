"""
tests/test_ppo_agent.py — Unit tests for src/ppo_agent.py.

Tests the linear LR schedule and PPO factory function.
Requires stable-baselines3 and gymnasium.
"""

from __future__ import annotations

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")
sb3 = pytest.importorskip("stable_baselines3")

from src.ppo_agent import linear_schedule, make_ppo
from src.portfolio_env import PortfolioEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_TICKERS = 5
ACTION_DIM = 6
OBS_DIM = 94
T = 50


def _make_config(lr_schedule: str = "constant") -> dict:
    return {
        "env": {
            "n_tickers": N_TICKERS,
            "action_dim": ACTION_DIM,
            "obs_dim": OBS_DIM,
            "transaction_cost": 0.001,
            "vol_eps": 1e-8,
            "vol_window": 21,
        },
        "ppo": {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "lr_schedule": lr_schedule,
            "n_steps": 50,
            "batch_size": 50,
            "n_epochs": 1,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {
                "net_arch": [64, 64],
                "activation_fn": "Tanh",
            },
            "total_timesteps": 100,
            "eval_freq": 100,
            "checkpoint_freq": 100,
        },
        "training": {"seed": 42},
    }


def _make_env() -> PortfolioEnv:
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((T, OBS_DIM)).astype(np.float32)
    ret = rng.normal(0.001, 0.01, (T, N_TICKERS)).astype(np.float32)
    return PortfolioEnv(obs, ret, _make_config())


# ---------------------------------------------------------------------------
# linear_schedule tests
# ---------------------------------------------------------------------------

class TestLinearSchedule:
    def test_start_equals_initial_value(self):
        fn = linear_schedule(3e-4)
        np.testing.assert_allclose(fn(1.0), 3e-4, rtol=1e-7)

    def test_end_equals_zero(self):
        fn = linear_schedule(3e-4)
        assert fn(0.0) == 0.0

    def test_midpoint(self):
        fn = linear_schedule(3e-4)
        np.testing.assert_allclose(fn(0.5), 1.5e-4, rtol=1e-7)

    def test_quarter_point(self):
        fn = linear_schedule(1e-3)
        np.testing.assert_allclose(fn(0.25), 2.5e-4, rtol=1e-7)

    def test_returns_float(self):
        fn = linear_schedule(3e-4)
        assert isinstance(fn(0.5), float)

    def test_monotonically_decreasing(self):
        fn = linear_schedule(1e-3)
        values = [fn(p) for p in [1.0, 0.75, 0.5, 0.25, 0.0]]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]


# ---------------------------------------------------------------------------
# make_ppo tests
# ---------------------------------------------------------------------------

class TestMakePpo:
    def test_constant_lr_returns_ppo(self):
        env = _make_env()
        config = _make_config(lr_schedule="constant")
        model = make_ppo(env, config, seed=42)
        assert isinstance(model, sb3.PPO)

    def test_linear_lr_returns_ppo(self):
        env = _make_env()
        config = _make_config(lr_schedule="linear")
        model = make_ppo(env, config, seed=42)
        assert isinstance(model, sb3.PPO)

    def test_constant_lr_is_float(self):
        env = _make_env()
        config = _make_config(lr_schedule="constant")
        model = make_ppo(env, config, seed=42)
        assert isinstance(model.learning_rate, float)

    def test_linear_lr_is_callable(self):
        env = _make_env()
        config = _make_config(lr_schedule="linear")
        model = make_ppo(env, config, seed=42)
        assert callable(model.learning_rate)

    def test_default_schedule_is_constant(self):
        """When lr_schedule key is absent, should default to constant."""
        env = _make_env()
        config = _make_config()
        del config["ppo"]["lr_schedule"]
        model = make_ppo(env, config, seed=42)
        assert isinstance(model.learning_rate, float)

    def test_invalid_activation_raises(self):
        env = _make_env()
        config = _make_config()
        config["ppo"]["policy_kwargs"]["activation_fn"] = "InvalidFn"
        with pytest.raises(ValueError, match="Unknown activation_fn"):
            make_ppo(env, config, seed=42)

    def test_seed_is_set(self):
        env = _make_env()
        config = _make_config()
        model = make_ppo(env, config, seed=123)
        assert model.seed == 123
