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
  4. 60/40 SPY/AGG (monthly rebalance, requires yfinance network access)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict

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


def bootstrap_ci(
    daily_returns: np.ndarray,
    metric_fn,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    block_size: int = 21,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Block bootstrap confidence interval for a portfolio metric.

    Uses non-overlapping block resampling (default block_size=21 = 1 trading
    month) to preserve the autocorrelation structure of financial returns.
    This avoids the bias of iid resampling on time-series data.

    Parameters
    ----------
    daily_returns : np.ndarray, shape (T,)
        Daily simple returns of the portfolio.
    metric_fn : callable
        Function (daily_returns → float) to compute the metric.
    n_bootstrap : int
        Number of bootstrap re-samples.
    ci : float
        Confidence level (e.g. 0.95 for 95%).
    block_size : int
        Block length for block bootstrap (default 21 = 1 trading month).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (lower, upper) : tuple of float
        Lower and upper bounds of the confidence interval.
    """
    rng = np.random.default_rng(seed)
    T = len(daily_returns)

    # Build blocks: divide the time series into non-overlapping blocks
    n_blocks = math.ceil(T / block_size)
    block_starts = np.arange(0, T, block_size)  # start indices of each block

    stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        # Resample blocks with replacement
        chosen = rng.integers(0, len(block_starts), size=n_blocks)
        pieces = []
        for c in chosen:
            start = block_starts[c]
            end = min(start + block_size, T)
            pieces.append(daily_returns[start:end])
        resampled = np.concatenate(pieces)[:T]  # trim to original length
        stats[i] = metric_fn(resampled)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(stats, 100 * alpha))
    upper = float(np.percentile(stats, 100 * (1 - alpha)))
    return lower, upper


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
    """
    1/n equal-weight buy-and-hold.  Weights drift with realised returns
    and are never rebalanced.

    weights[t] is the beginning-of-period allocation that earns return_array[t].
    After each step, weights drift: w *= (1 + r) then renormalize.
    """
    n = return_array.shape[1]
    T = len(return_array)
    weights = np.zeros((T, n), dtype=np.float32)
    current_weights = np.full(n, 1.0 / n, dtype=np.float32)

    for t in range(T):
        weights[t] = current_weights
        # Drift weights with realised returns (buy-and-hold: never rebalance)
        raw = current_weights * (1.0 + return_array[t])
        total = raw.sum()
        if total > 1e-8:
            current_weights = raw / total

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


def _run_sixty_forty_from_returns(
    spy_returns: np.ndarray,
    agg_returns: np.ndarray,
    dates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inner (testable) implementation of 60/40 SPY/AGG with monthly rebalancing.

    Parameters
    ----------
    spy_returns : np.ndarray, shape (T,)
        Daily simple returns of SPY.
    agg_returns : np.ndarray, shape (T,)
        Daily simple returns of AGG.
    dates : np.ndarray of datetime64
        Business dates aligned with the return arrays.

    Returns
    -------
    port_returns : np.ndarray, shape (T,)
    weights : np.ndarray, shape (T, 2)
        Columns: [SPY_weight, AGG_weight], sum to 1.0.
    """
    T = len(spy_returns)
    return_array = np.column_stack([spy_returns, agg_returns])
    weights = np.zeros((T, 2), dtype=np.float32)
    current_weights = np.array([0.6, 0.4], dtype=np.float32)
    current_month = -1

    timestamps = pd.DatetimeIndex(dates)

    for t in range(T):
        month = timestamps[t].month if t < len(timestamps) else current_month
        if month != current_month:
            current_weights = np.array([0.6, 0.4], dtype=np.float32)
            current_month = month
        weights[t] = current_weights
        # Drift weights with realised returns
        raw = current_weights * (1.0 + return_array[t])
        total = raw.sum()
        if total > 1e-8:
            current_weights = raw / total

    port_returns = (weights * return_array).sum(axis=1)
    return port_returns, weights


def run_sixty_forty_spy_agg(
    test_dates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    60% SPY / 40% AGG (iShares Core US Aggregate Bond ETF), rebalanced monthly.

    Fetches SPY and AGG prices via yfinance for the test period.  Between
    monthly rebalance dates, weights drift with returns.  At each month-end
    boundary, reset to 60/40.

    Parameters
    ----------
    test_dates : np.ndarray of datetime64
        Business dates for the test period.

    Returns
    -------
    port_returns : np.ndarray, shape (T,)
    weights : np.ndarray, shape (T, 2)
    """
    import warnings

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for the 60/40 SPY/AGG baseline.  "
            "Install with: pip install yfinance"
        ) from exc

    # Date range with buffer for first-day return computation
    start = (pd.Timestamp(test_dates[0]) - pd.offsets.BDay(5)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(test_dates[-1]) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            ["SPY", "AGG"], start=start, end=end,
            auto_adjust=True, progress=False,
        )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][["SPY", "AGG"]]
    else:
        prices = raw[["Close"]].rename(columns={"Close": "SPY"})

    # Compute daily simple returns
    returns = (prices / prices.shift(1) - 1.0).dropna()
    returns.index = pd.to_datetime(returns.index)

    # Align to test_dates
    test_idx = pd.DatetimeIndex(test_dates)
    aligned = returns.reindex(test_idx).fillna(0.0)

    spy_ret = aligned["SPY"].values.astype(np.float32)
    agg_ret = aligned["AGG"].values.astype(np.float32)

    return _run_sixty_forty_from_returns(spy_ret, agg_ret, test_dates)


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
    print("Loading merged features ...")
    merged_df = load_merged_features(config)
    train_df, _, test_df = split_data(merged_df, config)

    print("Fetching price data ...")
    prices = fetch_price_data(
        tickers,
        start_date = splits["train_start"],
        end_date   = splits["test_end"],
    )

    print("Computing price features ...")
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
    print(f"Loading model from {model_path}.zip ...")
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
    print("Computing evaluation metrics ...")

    # Collect all strategy returns + weights for metrics and bootstrap
    strategies = [
        ("ppo", ppo_returns, ppo_weights),
        ("equal_weight_buyhold", *run_equal_weight_buyhold(test_ret)),
        ("equal_weight_monthly", *run_equal_weight_monthly(test_ret, test_dates)),
        ("max_sharpe", *run_max_sharpe(train_ret, test_ret)),
    ]

    # 60/40 SPY/AGG baseline (requires yfinance / network access)
    try:
        sf_ret, sf_wt = run_sixty_forty_spy_agg(test_dates)
        strategies.append(("sixty_forty_spy_bonds", sf_ret, sf_wt))
    except Exception as e:
        print(f"Warning: 60/40 SPY/AGG baseline skipped — {e}")

    results = []
    for label, rets, wts in strategies:
        m = compute_metrics(rets, wts, label=label)

        # Bootstrap CIs for Sharpe, Sortino, and annualized return
        sharpe_lo, sharpe_hi = bootstrap_ci(rets, _annualised_sharpe)
        sortino_lo, sortino_hi = bootstrap_ci(rets, _annualised_sortino)
        ann_ret_lo, ann_ret_hi = bootstrap_ci(rets, _annualised_return)
        m["sharpe_ci_lower"] = round(sharpe_lo, 6)
        m["sharpe_ci_upper"] = round(sharpe_hi, 6)
        m["sortino_ci_lower"] = round(sortino_lo, 6)
        m["sortino_ci_upper"] = round(sortino_hi, 6)
        m["ann_return_ci_lower"] = round(ann_ret_lo, 6)
        m["ann_return_ci_upper"] = round(ann_ret_hi, 6)
        results.append(m)

    # 8. Save metrics --------------------------------------------------------
    metrics_path = out_cfg["metrics_file"]
    with open(metrics_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Metrics saved -> {metrics_path}")

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
    print(f"Daily weights saved -> {out_cfg['weights_file']}")


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
