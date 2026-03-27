"""
tune.py — Optuna hyperparameter tuning for the PPO portfolio agent.

Usage:
    python -m src.tune                        # uses config.yaml in cwd
    python -m src.tune --config path/to/config.yaml --n-trials 50

Searches over PPO hyperparameters and evaluates each configuration on
the validation split using the annualised Sharpe ratio as the objective.
The best parameters are saved to output/best_hparams.yaml for use with
``python -m src.train --use-tuned``.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import numpy as np
import yaml

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "optuna is required for hyperparameter tuning.  "
        "Install with: pip install optuna"
    ) from exc


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(
    trial: "optuna.Trial",
    config: dict,
    train_obs: np.ndarray,
    train_ret: np.ndarray,
    val_obs: np.ndarray,
    val_ret: np.ndarray,
) -> float:
    """
    Single Optuna trial: sample hyperparameters, train a short PPO run,
    evaluate on the validation set, and return the annualised Sharpe ratio.

    Parameters
    ----------
    trial : optuna.Trial
    config : dict
        Base config (will be deep-copied and overridden with sampled HPs).
    train_obs, train_ret : np.ndarray
        Normalised train observations and forward returns.
    val_obs, val_ret : np.ndarray
        Normalised validation observations and forward returns.

    Returns
    -------
    float — annualised Sharpe ratio on the validation set (higher is better).
    """
    import torch
    torch.set_num_threads(1)

    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from src.evaluate import _annualised_sharpe
    from src.portfolio_env import PortfolioEnv
    from src.ppo_agent import make_ppo

    # --- Sample hyperparameters -----------------------------------------------
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [126, 252, 504])
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.05, log=True)
    tc_penalty_lambda = trial.suggest_float("tc_penalty_lambda", 1e-4, 5e-3, log=True)

    # --- Build trial config ---------------------------------------------------
    trial_config = copy.deepcopy(config)
    trial_config["ppo"]["learning_rate"] = lr
    trial_config["ppo"]["n_steps"] = n_steps
    # Pick batch_size that evenly divides n_steps (63 divides 126, 252, 504)
    trial_config["ppo"]["batch_size"] = 63
    trial_config["ppo"]["gamma"] = gamma
    trial_config["ppo"]["ent_coef"] = ent_coef
    trial_config["env"]["transaction_cost"] = tc_penalty_lambda

    tuning_timesteps = int(
        config.get("tuning", {}).get("tuning_timesteps", 100_000)
    )

    # --- Build envs -----------------------------------------------------------
    train_env = DummyVecEnv([
        lambda: Monitor(PortfolioEnv(train_obs, train_ret, trial_config))
    ])
    train_env = VecNormalize(
        train_env, norm_obs=False, norm_reward=True, clip_reward=10.0,
    )

    # --- Train ----------------------------------------------------------------
    seed = int(config.get("training", {}).get("seed", 42))
    model = make_ppo(train_env, trial_config, seed=seed)
    model.learn(total_timesteps=tuning_timesteps)

    # --- Evaluate on validation -----------------------------------------------
    val_env = PortfolioEnv(val_obs, val_ret, trial_config)
    obs, _ = val_env.reset()
    port_returns = []
    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, info = val_env.step(action)
        port_returns.append(info["portfolio_return"])

    sharpe = _annualised_sharpe(np.array(port_returns))
    return float(sharpe)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str = "config.yaml", n_trials: int | None = None) -> dict:
    """
    Run Optuna hyperparameter search and save best parameters.

    Returns
    -------
    dict — best hyperparameters found.
    """
    from src.data_loader import load_config, load_merged_features, split_data
    from src.feature_engineer import (
        Normalizer,
        build_observation_matrix,
        compute_price_features,
        fetch_price_data,
    )

    # 1. Load data (same pipeline as train.py) --------------------------------
    config = load_config(config_path)
    tickers = config["tickers"]
    splits = config["splits"]
    out_cfg = config["output"]

    os.makedirs(out_cfg["dir"], exist_ok=True)

    print("Loading merged features for tuning ...")
    merged_df = load_merged_features(config)
    train_df, val_df, _ = split_data(merged_df, config)

    print("Fetching price data ...")
    prices = fetch_price_data(
        tickers,
        start_date=splits["train_start"],
        end_date=splits["val_end"],
    )

    print("Computing price features ...")
    price_features = compute_price_features(prices, tickers)

    train_obs, train_ret, _ = build_observation_matrix(
        train_df, price_features, tickers
    )
    val_obs, val_ret, _ = build_observation_matrix(
        val_df, price_features, tickers
    )

    normalizer = Normalizer()
    train_obs_norm = normalizer.fit_transform(train_obs)
    val_obs_norm = normalizer.transform(val_obs)

    # 2. Run Optuna study -----------------------------------------------------
    if n_trials is None:
        n_trials = int(config.get("tuning", {}).get("n_trials", 50))
    seed = int(config.get("training", {}).get("seed", 42))

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10, n_warmup_steps=5,
    )

    storage_path = out_cfg.get("optuna_storage", "output/optuna_study.db")
    storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name="ppo_portfolio",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        lambda trial: objective(
            trial, config,
            train_obs_norm, train_ret,
            val_obs_norm, val_ret,
        ),
        n_trials=n_trials,
    )

    # 3. Print top-10 trials summary -------------------------------------------
    sorted_trials = sorted(
        study.trials,
        key=lambda t: t.value if t.value is not None else float("-inf"),
        reverse=True,
    )
    print("\n" + "=" * 80)
    print(f"{'Rank':<6} {'Trial':<8} {'Sharpe':>8}   {'lr':>10} {'n_steps':>8} "
          f"{'gamma':>7} {'ent_coef':>10} {'tc_lambda':>10}")
    print("-" * 80)
    for rank, t in enumerate(sorted_trials[:10], 1):
        if t.value is None:
            continue
        p = t.params
        print(
            f"{rank:<6} {t.number:<8} {t.value:>8.4f}   "
            f"{p.get('learning_rate', 0):>10.2e} "
            f"{p.get('n_steps', 0):>8} "
            f"{p.get('gamma', 0):>7.4f} "
            f"{p.get('ent_coef', 0):>10.2e} "
            f"{p.get('tc_penalty_lambda', 0):>10.2e}"
        )
    print("=" * 80)

    # 4. Save best parameters -------------------------------------------------
    best = study.best_params

    best_path = str(Path(out_cfg["dir"]) / "best_hparams.yaml")
    with open(best_path, "w") as f:
        yaml.dump(best, f, default_flow_style=False)

    print(f"\nBest Sharpe: {study.best_value:.4f}")
    print(f"Best params saved -> {best_path}")
    print(f"Best params: {best}")

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for PPO portfolio agent"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml in cwd)"
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials (default: 50)"
    )
    args = parser.parse_args()
    main(config_path=args.config, n_trials=args.n_trials)
