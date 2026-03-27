"""
train.py — Full PPO training pipeline for Phase 3 portfolio agent.

Usage:
    python -m src.train                        # uses config.yaml in cwd
    python -m src.train --config path/to/config.yaml

Pipeline:
  1. Load config and merged features (from Phase 1 + Phase 2 upstream CSVs)
  2. Fetch price data via yfinance for the full date range
  3. Compute price features (log returns, realised vol, simple returns)
  4. Build observation + forward-return matrices for each split
  5. Fit z-score Normalizer on train split only; apply to train + val
  6. Instantiate PortfolioEnv for train and val
  7. Train PPO with EvalCallback + CheckpointCallback
  8. Save best model weights and normalizer to output/trained_model/
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np


def main(config_path: str = "config.yaml", use_tuned: bool = False, seed: int | None = None) -> None:
    # --- Lazy imports so the module can be imported without all deps --------
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for servers
    import matplotlib.pyplot as plt

    import yaml

    from src.data_loader import load_config, load_merged_features, split_data
    from src.feature_engineer import (
        Normalizer,
        build_observation_matrix,
        compute_price_features,
        fetch_price_data,
    )
    from src.portfolio_env import PortfolioEnv
    from src.ppo_agent import make_callbacks, make_ppo
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    import torch
    torch.set_num_threads(1)  # prevent thread contention across SubprocVecEnv workers

    # 1. Load config ---------------------------------------------------------
    config = load_config(config_path)
    tickers = config["tickers"]
    splits  = config["splits"]
    out_cfg = config["output"]

    # 1b. Override PPO hyper-parameters with Optuna-tuned values if requested --
    if use_tuned:
        tuned_path = Path(out_cfg["dir"]) / "best_hparams.yaml"
        if not tuned_path.exists():
            raise FileNotFoundError(
                f"Tuned params not found at {tuned_path}. "
                "Run `python -m src.tune` first."
            )
        with open(tuned_path, "r") as f:
            tuned_params = yaml.safe_load(f)
        # Apply tc_penalty_lambda to env config, rest to ppo config
        if "tc_penalty_lambda" in tuned_params:
            config["env"]["transaction_cost"] = tuned_params.pop("tc_penalty_lambda")
        config["ppo"].update(tuned_params)
        print(f"Loaded tuned params from {tuned_path}")

    # 1c. Override seed if provided via CLI ------------------------------------
    if seed is not None:
        config["training"]["seed"] = seed

    os.makedirs(out_cfg["model_dir"], exist_ok=True)

    # 2. Load and merge Phase 1 + Phase 2 features ---------------------------
    print("Loading merged features ...")
    merged_df = load_merged_features(config)
    train_df, val_df, _ = split_data(merged_df, config)

    # 3. Fetch price data (with buffer for rolling warm-up) ------------------
    print("Fetching price data via yfinance ...")
    prices = fetch_price_data(
        tickers,
        start_date = splits["train_start"],
        end_date   = splits["val_end"],
    )

    # 4. Compute price features ----------------------------------------------
    print("Computing price features ...")
    price_features = compute_price_features(prices, tickers)

    # 5. Build obs + return matrices -----------------------------------------
    print("Building observation matrices ...")
    train_obs, train_ret, train_dates = build_observation_matrix(
        train_df, price_features, tickers
    )
    val_obs, val_ret, val_dates = build_observation_matrix(
        val_df, price_features, tickers
    )

    print(f"  Train: {train_obs.shape[0]} steps | Val: {val_obs.shape[0]} steps")

    # 6. Fit normalizer on train only; apply to train + val ------------------
    print("Fitting z-score normalizer on train split ...")
    normalizer = Normalizer()
    train_obs_norm = normalizer.fit_transform(train_obs)
    val_obs_norm   = normalizer.transform(val_obs)

    normalizer_path = str(Path(out_cfg["model_dir"]) / "normalizer.pkl")
    normalizer.save(normalizer_path)
    print(f"  Normalizer saved -> {normalizer_path}")

    # 7. Instantiate environments --------------------------------------------
    train_cfg = config.get("training", {})
    norm_reward = train_cfg.get("norm_reward", True)
    clip_reward = float(train_cfg.get("clip_reward", 10.0))
    n_envs = int(train_cfg.get("n_envs", 1))

    def _make_train_env(obs_arr, ret_arr, cfg):
        """Factory that returns a closure for SubprocVecEnv.
        Each worker gets its own copy of arrays to avoid shared-memory issues.
        """
        _obs = obs_arr.copy()
        _ret = ret_arr.copy()
        _cfg = copy.deepcopy(cfg)
        def _init():
            return Monitor(PortfolioEnv(_obs, _ret, _cfg))
        return _init

    env_fns = [_make_train_env(train_obs_norm, train_ret, config)
               for _ in range(n_envs)]

    if n_envs > 1:
        train_env = SubprocVecEnv(env_fns)
    else:
        train_env = DummyVecEnv(env_fns)

    train_env = VecNormalize(
        train_env,
        norm_obs=False,       # obs already z-scored by our Normalizer
        norm_reward=norm_reward,
        clip_reward=clip_reward,
    )

    val_env = DummyVecEnv([
        lambda: Monitor(PortfolioEnv(val_obs_norm, val_ret, config))
    ])
    val_env = VecNormalize(
        val_env,
        norm_obs=False,
        norm_reward=False,   # don't normalize val rewards
        training=False,      # don't update running stats during eval
    )

    # 8. Build PPO + callbacks -----------------------------------------------
    print("Initialising PPO agent ...")
    model = make_ppo(train_env, config, seed=int(config["training"]["seed"]))
    callbacks = make_callbacks(val_env, config, out_cfg["model_dir"])

    # 9. Train ---------------------------------------------------------------
    total_timesteps = int(config["ppo"]["total_timesteps"])
    print(f"Training for {total_timesteps:,} timesteps ...")
    model.learn(
        total_timesteps = total_timesteps,
        callback        = callbacks,
        reset_num_timesteps = True,
    )

    # 10. Save final model ---------------------------------------------------
    final_model_path = str(Path(out_cfg["model_dir"]) / "final_model")
    model.save(final_model_path)
    print(f"Final model saved -> {final_model_path}.zip")
    print(
        f"Best model (highest val reward) -> "
        f"{out_cfg['model_dir']}/best_model.zip"
    )

    # 10b. Save VecNormalize running statistics --------------------------------
    vec_norm_path = out_cfg.get(
        "vecnormalize_stats",
        str(Path(out_cfg["model_dir"]) / "vecnormalize_stats.pkl"),
    )
    train_env.save(vec_norm_path)
    print(f"VecNormalize stats saved -> {vec_norm_path}")

    # 11. Plot training reward curve (from Monitor logs) ---------------------
    try:
        rewards = model.ep_info_buffer
        if rewards:
            ep_rewards = [ep["r"] for ep in rewards]
            plt.figure(figsize=(10, 4))
            plt.plot(ep_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Cumulative reward")
            plt.title("Training reward curve")
            plt.tight_layout()
            plt.savefig(out_cfg["curves_file"], dpi=150)
            print(f"Training curve saved -> {out_cfg['curves_file']}")
    except Exception as e:
        print(f"Warning: could not save training curves — {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phase 3 PPO portfolio agent")
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml in cwd)"
    )
    parser.add_argument(
        "--use-tuned", action="store_true",
        help="Override PPO hyper-parameters with Optuna-tuned values from output/best_hparams.yaml"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override training seed (default: use config.yaml value)"
    )
    args = parser.parse_args()
    main(config_path=args.config, use_tuned=args.use_tuned, seed=args.seed)
