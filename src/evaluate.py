"""
evaluate.py — Evaluate a trained PPO agent on the held-out test split.

Usage:
    python -m src.evaluate                         # uses config.yaml in cwd
    python -m src.evaluate --config path/to/config.yaml

Produces:
  output/eval_metrics.json   — Sharpe, Sortino, max drawdown, Calmar, turnover
  output/portfolio_weights.csv — daily allocation per ticker + cash
  output/training_curves.png  — (already saved by train.py; updated here if missing)

Baselines compared:
  1. Equal-weight buy-and-hold  (1/n_tickers each, never rebalanced)
  2. Equal-weight monthly rebalance
  3. Max-Sharpe portfolio (mean-variance optimisation on train data, held fixed)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

TRADING_DAYS = 252


def _annualised_return(daily_returns: np.ndarray) -> float:
    total = np.prod(1.0 + daily_returns) - 1.0
    n = len(daily_returns)
    return float((1.0 + total) ** (TRADING_DAYS / max(n, 1)) - 1.0)


def _annualised_sharpe(daily_returns: np.ndarray, rf: float = 0.0) -> float:
    excess = daily_returns - rf / TRADING_DAYS
    std = np.std(excess)
    if std < 1e-10:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(TRADING_DAYS))


def _annualised_sortino(daily_returns: np.ndarray, rf: float = 0.0) -> float:
    excess = daily_returns - rf / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = np.std(downside)
    if downside_std < 1e-10:
        return float("inf")
    return float(np.mean(excess) / downside_std * np.sqrt(TRADING_DAYS))


def _max_drawdown(daily_returns: np.ndarray) -> float:
    cum = np.cumprod(1.0 + daily_returns)
    peak = np.maximum.accumulate(cum)
    drawdown = (peak - cum) / peak
    return float(drawdown.max())


def _calmar_ratio(daily_returns: np.ndarray) -> float:
    ann_ret = _annualised_return(daily_returns)
    mdd = _max_drawdown(daily_returns)
    if mdd < 1e-10:
        return float("inf")
    return ann_ret / mdd


def _mean_daily_turnover(weights: np.ndarray) -> float:
    """Mean of sum_i |w_i(t) - w_i(t-1)| over all steps."""
    if len(weights) < 2:
        return 0.0
    diffs = np.abs(np.diff(weights, axis=0)).sum(axis=1)
    return float(diffs.mean())


def compute_metrics(
    daily_returns: np.ndarray,
    weights: np.ndarray,
    label: str = "ppo",
) -> Dict[str, float]:
    """
    Compute standard portfolio evaluation metrics.

    Parameters
    ----------
    daily_returns : np.ndarray, shape (T,)
        Portfolio daily simple returns.
    weights : np.ndarray, shape (T, n_assets)
        Portfolio weights at each step (for turnover calculation).
    label : str
        Strategy label for display.

    Returns
    -------
    dict with keys: annualised_return, sharpe, sortino, max_drawdown, calmar,
                    mean_daily_turnover
    """
    return {
        "label":              label,
        "annualised_return":  round(_annualised_return(daily_returns), 6),
        "sharpe":             round(_annualised_sharpe(daily_returns), 6),
        "sortino":            round(_annualised_sortino(daily_returns), 6),
        "max_drawdown":       round(_max_drawdown(daily_returns), 6),
        "calmar":             round(_calmar_ratio(daily_returns), 6),
        "mean_daily_turnover": round(_mean_daily_turnover(weights), 6),
    }


# ---------------------------------------------------------------------------
# Baseline strategies
# ---------------------------------------------------------------------------

def run_equal_weight_buyhold(
    return_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """1/n equal-weight buy-and-hold. No rebalancing."""
    n = return_array.shape[1]
    weights = np.full((len(return_array), n), 1.0 / n, dtype=np.float32)
    port_returns = (weights * return_array).sum(axis=1)
    return port_returns, weights


def run_equal_weight_monthly(
    return_array: np.ndarray,
    dates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """1/n equal-weight with monthly rebalancing."""
    n = return_array.shape[1]
    T = len(return_array)
    weights = np.zeros((T, n), dtype=np.float32)
    current_weights = np.full(n, 1.0 / n, dtype=np.float32)
    current_month = -1

    timestamps = pd.DatetimeIndex(dates)

    for t in range(T):
        month = timestamps[t].month if t < len(timestamps) else current_month
        if month != current_month:
            current_weights = np.full(n, 1.0 / n, dtype=np.float32)
            current_month = month
        weights[t] = current_weights
        # Drift weights with realised returns
        raw = current_weights * (1.0 + return_array[t])
        total = raw.sum()
        if total > 1e-8:
            current_weights = raw / total

    port_returns = (weights * return_array).sum(axis=1)
    return port_returns, weights


def run_max_sharpe(
    train_return_array: np.ndarray,
    test_return_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Maximum-Sharpe (mean-variance) portfolio estimated from train returns,
    applied as a fixed allocation on the test set.

    Uses scipy's minimize with SLSQP (analytical quadratic programming).
    Falls back to equal-weight if optimisation fails.
    """
    from scipy.optimize import minimize

    n = train_return_array.shape[1]
    mu = train_return_array.mean(axis=0)
    cov = np.cov(train_return_array.T) + np.eye(n) * 1e-8  # regularise

    def neg_sharpe(w):
        ret = w @ mu
        std = np.sqrt(w @ cov @ w)
        return -(ret / (std + 1e-10))

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds = [(0.0, 1.0)] * n
    w0 = np.full(n, 1.0 / n)

    result = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-9})
    opt_weights = result.x if result.success else w0
    opt_weights = np.clip(opt_weights, 0, 1)
    opt_weights /= opt_weights.sum()

    T = len(test_return_array)
    weights = np.tile(opt_weights, (T, 1)).astype(np.float32)
    port_returns = (weights * test_return_array).sum(axis=1)
    return port_returns, weights


# ---------------------------------------------------------------------------
# Main evaluation entrypoint
# ---------------------------------------------------------------------------

def main(config_path: str = "config.yaml") -> None:
    from src.data_loader import load_config, load_merged_features, split_data
    from src.feature_engineer import (
        Normalizer,
        build_observation_matrix,
        compute_price_features,
        fetch_price_data,
    )
    from src.portfolio_env import PortfolioEnv
    from stable_baselines3 import PPO

    # 1. Load config ---------------------------------------------------------
    config = load_config(config_path)
    tickers = config["tickers"]
    splits  = config["splits"]
    out_cfg = config["output"]

    os.makedirs(out_cfg["dir"], exist_ok=True)

    # 2. Load data for all splits -------------------------------------------
    print("Loading merged features …")
    merged_df = load_merged_features(config)
    train_df, _, test_df = split_data(merged_df, config)

    print("Fetching price data …")
    prices = fetch_price_data(
        tickers,
        start_date = splits["train_start"],
        end_date   = splits["test_end"],
    )

    print("Computing price features …")
    price_features = compute_price_features(prices, tickers)

    # 3. Build observation matrices -----------------------------------------
    train_obs, train_ret, train_dates = build_observation_matrix(
        train_df, price_features, tickers
    )
    test_obs, test_ret, test_dates = build_observation_matrix(
        test_df, price_features, tickers
    )

    # 4. Normalise with train-fitted normalizer ------------------------------
    normalizer_path = str(Path(out_cfg["model_dir"]) / "normalizer.pkl")
    if not Path(normalizer_path).exists():
        raise FileNotFoundError(
            f"Normalizer not found at {normalizer_path}.  "
            "Run train.py first."
        )
    normalizer = Normalizer.load(normalizer_path)
    test_obs_norm = normalizer.transform(test_obs)

    # 5. Load best PPO model ------------------------------------------------
    model_path = str(Path(out_cfg["model_dir"]) / "best_model")
    if not Path(model_path + ".zip").exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.zip.  "
            "Run train.py first."
        )
    print(f"Loading model from {model_path}.zip …")
    model = PPO.load(model_path)

    # 6. Roll out PPO on test set -------------------------------------------
    test_env = PortfolioEnv(test_obs_norm, test_ret, config)
    obs, _ = test_env.reset()
    ppo_returns = []
    ppo_weights = []

    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, info = test_env.step(action)
        ppo_returns.append(info["portfolio_return"])
        ppo_weights.append(info["weights"])

    ppo_returns = np.array(ppo_returns, dtype=np.float32)
    ppo_weights = np.array(ppo_weights, dtype=np.float32)

    # 7. Compute metrics for PPO and all baselines --------------------------
    print("Computing evaluation metrics …")

    results = [
        compute_metrics(ppo_returns, ppo_weights, label="ppo"),
        compute_metrics(*run_equal_weight_buyhold(test_ret), label="equal_weight_buyhold"),
        compute_metrics(*run_equal_weight_monthly(test_ret, test_dates), label="equal_weight_monthly"),
        compute_metrics(*run_max_sharpe(train_ret, test_ret), label="max_sharpe"),
    ]

    # 8. Save metrics --------------------------------------------------------
    metrics_path = out_cfg["metrics_file"]
    with open(metrics_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Metrics saved → {metrics_path}")

    # Print summary table
    print("\n" + "=" * 72)
    print(f"{'Strategy':<30} {'Sharpe':>8} {'Sortino':>8} {'MDD':>8} {'AnnRet':>8}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['label']:<30} "
            f"{r['sharpe']:>8.3f} "
            f"{r['sortino']:>8.3f} "
            f"{r['max_drawdown']:>8.3f} "
            f"{r['annualised_return']:>8.3f}"
        )
    print("=" * 72)

    # 9. Save daily weight series -------------------------------------------
    weight_cols = tickers + ["cash"]
    date_strs = pd.DatetimeIndex(test_dates[: len(ppo_weights)]).strftime("%Y-%m-%d")
    weight_df = pd.DataFrame(
        ppo_weights,
        columns=weight_cols[: ppo_weights.shape[1]],
        index=date_strs,
    )
    weight_df.index.name = "Date"
    weight_df.to_csv(out_cfg["weights_file"])
    print(f"Daily weights saved → {out_cfg['weights_file']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO portfolio agent on test split"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml in cwd)"
    )
    args = parser.parse_args()
    main(config_path=args.config)
