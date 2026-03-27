"""
regime_analysis.py — Regime-conditional performance analysis.

Loads the best PPO model, runs a full episode on the test set, and
breaks down portfolio weights and returns by the Phase 1 regime label
(argmax of prob_low / prob_med / prob_high / prob_crash).

Usage:
    python -m src.regime_analysis
    python -m src.regime_analysis --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Regime labels (canonical order matches feature_engineer.SHARED_FEATURES)
# ---------------------------------------------------------------------------

REGIME_LABELS = ["low_vol", "medium", "high_vol", "crash"]
REGIME_PROB_COLS = ["prob_low", "prob_med", "prob_high", "prob_crash"]


# ---------------------------------------------------------------------------
# Core analysis function (testable with synthetic data)
# ---------------------------------------------------------------------------

def analyse_regime_performance(
    daily_returns: np.ndarray,
    daily_weights: np.ndarray,
    regime_indices: np.ndarray,
    ticker_names: list[str],
) -> dict:
    """
    Group daily returns and weights by regime and compute per-regime statistics.

    Parameters
    ----------
    daily_returns : np.ndarray, shape (T,)
        Daily portfolio returns.
    daily_weights : np.ndarray, shape (T, n_assets)
        Portfolio weights at each step (n_assets = n_tickers + cash).
    regime_indices : np.ndarray, shape (T,)
        Integer regime labels (0=low_vol, 1=medium, 2=high_vol, 3=crash).
    ticker_names : list of str
        Names for each weight column (e.g. ["AAPL", "MSFT", ..., "cash"]).

    Returns
    -------
    dict with keys per regime: mean_weights, mean_return, sharpe, n_days
    """
    from src.evaluate import _annualised_sharpe

    results = {}
    for regime_idx, label in enumerate(REGIME_LABELS):
        mask = regime_indices == regime_idx
        n_days = int(mask.sum())

        if n_days == 0:
            results[label] = {
                "mean_weights": {t: 0.0 for t in ticker_names},
                "mean_daily_return": 0.0,
                "sharpe": 0.0,
                "n_days": 0,
            }
            continue

        regime_returns = daily_returns[mask]
        regime_weights = daily_weights[mask]

        mean_weights = {}
        for i, name in enumerate(ticker_names):
            mean_weights[name] = round(float(regime_weights[:, i].mean()), 6)

        sharpe = float(_annualised_sharpe(regime_returns)) if n_days >= 2 else 0.0

        results[label] = {
            "mean_weights": mean_weights,
            "mean_daily_return": round(float(regime_returns.mean()), 8),
            "sharpe": round(sharpe, 4),
            "n_days": n_days,
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str = "config.yaml") -> dict:
    from src.data_loader import load_config, load_merged_features, split_data
    from src.feature_engineer import (
        Normalizer,
        build_observation_matrix,
        compute_price_features,
        fetch_price_data,
    )
    from src.portfolio_env import PortfolioEnv
    from stable_baselines3 import PPO

    # 1. Load config and data ------------------------------------------------
    config = load_config(config_path)
    tickers = config["tickers"]
    splits = config["splits"]
    out_cfg = config["output"]

    print("Loading merged features ...")
    merged_df = load_merged_features(config)
    _, _, test_df = split_data(merged_df, config)

    print("Fetching price data ...")
    prices = fetch_price_data(
        tickers,
        start_date=splits["train_start"],
        end_date=splits["test_end"],
    )

    print("Computing price features ...")
    price_features = compute_price_features(prices, tickers)

    test_obs, test_ret, test_dates = build_observation_matrix(
        test_df, price_features, tickers
    )

    # 2. Normalise with train-fitted normalizer ------------------------------
    normalizer_path = str(Path(out_cfg["model_dir"]) / "normalizer.pkl")
    normalizer = Normalizer.load(normalizer_path)
    test_obs_norm = normalizer.transform(test_obs)

    # 3. Extract regime labels from the last 4 columns of obs_array ----------
    #    obs layout: [AAPL×14 | MSFT×14 | GOOGL×14 | AMZN×14 | TSLA×14 | regime×4]
    #    The regime probs are in the UNNORMALIZED feature array (test_obs),
    #    not the z-scored version.
    regime_probs = test_obs[:, -4:]  # (T, 4): prob_low, prob_med, prob_high, prob_crash
    regime_indices = np.argmax(regime_probs, axis=1)  # (T,)

    # 4. Load model and run episode ------------------------------------------
    model_path = str(Path(out_cfg["model_dir"]) / "best_model")
    print(f"Loading model from {model_path}.zip ...")
    model = PPO.load(model_path)

    env = PortfolioEnv(test_obs_norm, test_ret, config)
    obs, _ = env.reset()

    daily_returns = []
    daily_weights = []
    terminated = False

    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, info = env.step(action)
        daily_returns.append(info["portfolio_return"])
        daily_weights.append(info["weights"])

    daily_returns = np.array(daily_returns, dtype=np.float32)
    daily_weights = np.array(daily_weights, dtype=np.float32)

    # Align regime_indices to the episode length (T-1 steps from T obs rows)
    regime_aligned = regime_indices[: len(daily_returns)]

    # 5. Analyse per regime ---------------------------------------------------
    asset_names = tickers + ["cash"]
    results = analyse_regime_performance(
        daily_returns, daily_weights, regime_aligned, asset_names
    )

    # 6. Save and print -------------------------------------------------------
    output_path = str(Path(out_cfg["dir"]) / "regime_analysis.json")
    os.makedirs(out_cfg["dir"], exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Regime analysis saved -> {output_path}")

    # Print formatted table
    header_tickers = "".join(f"{t:>8}" for t in tickers)
    print(f"\n{'Regime':<12} |{header_tickers} {'Cash':>8} | {'Sharpe':>8} {'Days':>6}")
    print("-" * (12 + 1 + 8 * len(tickers) + 8 + 3 + 8 + 6))
    for label in REGIME_LABELS:
        r = results[label]
        w = r["mean_weights"]
        w_str = "".join(f"{w.get(t, 0):>8.3f}" for t in tickers)
        cash_str = f"{w.get('cash', 0):>8.3f}"
        print(f"{label:<12} |{w_str} {cash_str} | {r['sharpe']:>8.4f} {r['n_days']:>6}")
    print()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regime-conditional analysis of PPO portfolio agent"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml in cwd)"
    )
    args = parser.parse_args()
    main(config_path=args.config)
