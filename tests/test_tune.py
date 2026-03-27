"""
tests/test_tune.py — Unit tests for src/tune.py.

Tests the Optuna objective function with synthetic data and tiny timesteps.
No network access or upstream data files required.
"""

from __future__ import annotations

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")
sb3 = pytest.importorskip("stable_baselines3")
optuna = pytest.importorskip("optuna")

from src.tune import objective


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_TICKERS = 5
OBS_DIM = 79
T_TRAIN = 100
T_VAL = 50


def _make_config() -> dict:
    return {
        "env": {
            "n_tickers": N_TICKERS,
            "action_dim": 6,
            "obs_dim": OBS_DIM,
            "transaction_cost": 0.001,
            "vol_eps": 1e-8,
            "vol_window": 21,
        },
        "ppo": {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "lr_schedule": "constant",
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
        "tuning": {"n_trials": 1, "tuning_timesteps": 100},
    }


def _make_data():
    rng = np.random.default_rng(42)
    train_obs = rng.standard_normal((T_TRAIN, OBS_DIM)).astype(np.float32)
    train_ret = rng.normal(0.001, 0.01, (T_TRAIN, N_TICKERS)).astype(np.float32)
    val_obs = rng.standard_normal((T_VAL, OBS_DIM)).astype(np.float32)
    val_ret = rng.normal(0.001, 0.01, (T_VAL, N_TICKERS)).astype(np.float32)
    return train_obs, train_ret, val_obs, val_ret


# ---------------------------------------------------------------------------
# objective tests
# ---------------------------------------------------------------------------

class TestObjective:
    def test_returns_finite_float(self):
        """Objective should return a finite float Sharpe value."""
        config = _make_config()
        train_obs, train_ret, val_obs, val_ret = _make_data()

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(
                trial, config, train_obs, train_ret, val_obs, val_ret
            ),
            n_trials=1,
        )
        assert isinstance(study.best_value, float)
        assert np.isfinite(study.best_value)

    def test_returns_sharpe_type(self):
        """Single trial should return a numeric Sharpe ratio."""
        config = _make_config()
        train_obs, train_ret, val_obs, val_ret = _make_data()

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(
                trial, config, train_obs, train_ret, val_obs, val_ret
            ),
            n_trials=1,
        )
        # Sharpe should be a real number (could be negative)
        assert study.best_value is not None


class TestSearchSpace:
    def test_trial_params_match_spec(self):
        """Verify the search space matches the user spec."""
        config = _make_config()
        train_obs, train_ret, val_obs, val_ret = _make_data()

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(
                trial, config, train_obs, train_ret, val_obs, val_ret
            ),
            n_trials=1,
        )
        params = study.best_trial.params
        assert "learning_rate" in params
        assert "n_steps" in params
        assert params["n_steps"] in [126, 252, 504]
        assert "gamma" in params
        assert "ent_coef" in params
        assert "tc_penalty_lambda" in params
        assert "lambda_dd" in params
        assert "dd_threshold" in params
        assert "lambda_div" in params
        assert "max_weight_per_ticker" in params
