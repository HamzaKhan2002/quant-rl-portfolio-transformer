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
import os
import sys
from pathlib import Path

import numpy as np


def main(config_path: str = "config.yaml") -> None:
    # --- Lazy imports so the module can be imported without all deps --------
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for servers
    import matplotlib.pyplot as plt

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

    # 1. Load config ---------------------------------------------------------
    config = load_config(config_path)
    tickers = config["tickers"]
    splits  = config["splits"]
    out_cfg = config["output"]

    os.makedirs(out_cfg["model_dir"], exist_ok=True)

    # 2. Load and merge Phase 1 + Phase 2 features ---------------------------
    print("Loading merged features …")
    merged_df = load_merged_features(config)
    train_df, val_df, _ = split_data(merged_df, config)

    # 3. Fetch price data (with buffer for rolling warm-up) ------------------
    print("Fetching price data via yfinance …")
    prices = fetch_price_data(
        tickers,
        start_date = splits["train_start"],
        end_date   = splits["val_end"],
    )

    # 4. Compute price features ----------------------------------------------
    print("Computing price features …")
    price_features = compute_price_features(prices, tickers)

    # 5. Build obs + return matrices -----------------------------------------
    print("Building observation matrices …")
    train_obs, train_ret, train_dates = build_observation_matrix(
        train_df, price_features, tickers
    )
    val_obs, val_ret, val_dates = build_observation_matrix(
        val_df, price_features, tickers
    )

    print(f"  Train: {train_obs.shape[0]} steps | Val: {val_obs.shape[0]} steps")

    # 6. Fit normalizer on train only; apply to train + val ------------------
    print("Fitting z-score normalizer on train split …")
    normalizer = Normalizer()
    train_obs_norm = normalizer.fit_transform(train_obs)
    val_obs_norm   = normalizer.transform(val_obs)

    normalizer_path = str(Path(out_cfg["model_dir"]) / "normalizer.pkl")
    normalizer.save(normalizer_path)
    print(f"  Normalizer saved → {normalizer_path}")

    # 7. Instantiate environments --------------------------------------------
    train_env = Monitor(PortfolioEnv(train_obs_norm, train_ret, config))
    val_env   = PortfolioEnv(val_obs_norm, val_ret, config)

    # 8. Build PPO + callbacks -----------------------------------------------
    print("Initialising PPO agent …")
    model = make_ppo(train_env, config, seed=42)
    callbacks = make_callbacks(val_env, config, out_cfg["model_dir"])

    # 9. Train ---------------------------------------------------------------
    total_timesteps = int(config["ppo"]["total_timesteps"])
    print(f"Training for {total_timesteps:,} timesteps …")
    model.learn(
        total_timesteps = total_timesteps,
        callback        = callbacks,
        reset_num_timesteps = True,
    )

    # 10. Save final model ---------------------------------------------------
    final_model_path = str(Path(out_cfg["model_dir"]) / "final_model")
    model.save(final_model_path)
    print(f"Final model saved → {final_model_path}.zip")
    print(
        f"Best model (highest val reward) → "
        f"{out_cfg['model_dir']}/best_model.zip"
    )

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
            print(f"Training curve saved → {out_cfg['curves_file']}")
    except Exception:
        pass  # non-critical


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phase 3 PPO portfolio agent")
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml in cwd)"
    )
    args = parser.parse_args()
    main(config_path=args.config)
