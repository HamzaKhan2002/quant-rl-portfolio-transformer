"""
walk_forward.py — Walk-forward cross-validation for the PPO portfolio agent.

Six evaluation folds are defined in config.yaml under ``evaluation.folds``:

  Fold 1 — Full model (74-dim): train 2014-2018, val 2019, test 2020 (COVID crash)
  Fold 2 — Full model (74-dim): train 2014-2020, val 2021, test 2022 (bear market)
  Fold 3 — Full model (74-dim): train 2014-2021, val 2022, test 2023 (recovery)
  Fold 4 — Full model (74-dim): train 2014-2022, val 2023, test 2024 (AI bull run)
  Fold 5 — Full model (74-dim): train 2014-2023, val 2024, test 2025 [default]
  Fold R — Regime-only (34-dim): train 2019-2021, val 2022-H1, test 2022-H2

Sentiment data now extends back to 2014 (EDGAR 10-K filings); X/Grok social
data is available from 2022 only (has_x_data=0 flag for pre-2022 rows).
``Fold R`` uses a reduced 34-dim observation (price + regime, no sentiment)
to provide a second evaluation on the 2022 bear market at half-year granularity.

Usage::

    python -m src.walk_forward                      # all folds, seed 42
    python -m src.walk_forward --fold 1             # single fold
    python -m src.walk_forward --seeds 42 1 7 2026  # ensemble seeds
    python -m src.walk_forward --use-tuned          # load best_hparams.yaml

Output:
    output/walk_forward/fold_{id}/  — per-fold model + metrics
    output/walk_forward/summary.json  — cross-fold aggregate statistics
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS = 252
_PRICE_REGIME_FEATURES = [
    "log_return_1d",
    "log_return_5d",
    "log_return_21d",
    "realized_vol_21d",
    "days_since_filing",
    "has_sentiment_data",
]
_SHARED_FEATURES = ["prob_low", "prob_med", "prob_high", "prob_crash"]


# ---------------------------------------------------------------------------
# Regime-only observation builder (Fold R)
# ---------------------------------------------------------------------------

def _build_regime_only_obs_matrix(
    regime_df: pd.DataFrame,
    price_features_df: pd.DataFrame,
    tickers: List[str],
    fold_train_start: str,
    fold_test_end: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the 34-dim observation matrix for regime-only folds (no sentiment).

    Per-ticker features (6 each):
        log_return_1d, log_return_5d, log_return_21d, realized_vol_21d,
        days_since_filing (= 0, no filings), has_sentiment_data (= 0)

    Shared features (4):
        prob_low, prob_med, prob_high, prob_crash  (from regime_df)

    Total: 6 × 5 + 4 = 34 dimensions.

    Parameters
    ----------
    regime_df : pd.DataFrame
        Raw regime CSV with columns [Date, Ticker, prob_low, prob_med,
        prob_high, prob_crash] (one row per (Date, Ticker) or wide per Date).
        If the regime CSV has a single row per date (wide format with one set
        of prob_* columns), it is broadcasted across all tickers.
    price_features_df : pd.DataFrame
        Output of ``compute_price_features`` (long format).
    tickers : list of str
        Canonical ticker order.
    fold_train_start, fold_test_end : str
        ISO date bounds; used to trim the observation matrix to the fold.

    Returns
    -------
    obs_array : np.ndarray, shape (T, 34)
    forward_return_array : np.ndarray, shape (T, 5)
    dates : np.ndarray of np.datetime64
    """
    from src.feature_engineer import SHARED_FEATURES

    # --- Trim price features to fold dates -----------------------------------
    pf = price_features_df.copy()
    pf["Date"] = pd.to_datetime(pf["Date"])
    pf = pf[
        (pf["Date"] >= fold_train_start) & (pf["Date"] <= fold_test_end)
    ]

    # --- Trim and prepare regime data ----------------------------------------
    rd = regime_df.copy()
    rd["Date"] = pd.to_datetime(rd["Date"])
    rd = rd[(rd["Date"] >= fold_train_start) & (rd["Date"] <= fold_test_end)]

    # Regime CSV may have one row per date (wide) or per (Date, Ticker).
    # Detect by checking if Ticker column exists.
    if "Ticker" not in rd.columns:
        # Wide format: one row per date; select shared features only.
        shared_df = rd.set_index("Date")[SHARED_FEATURES]
    else:
        # Long format: pick first ticker's row per date (regime is date-level).
        shared_df = (
            rd[rd["Ticker"] == tickers[0]]
            .set_index("Date")[SHARED_FEATURES]
        )

    # --- Per-ticker price pivot (4 features) ---------------------------------
    ticker_rank = {t: i for i, t in enumerate(tickers)}
    pf["_rank"] = pf["Ticker"].map(ticker_rank)
    pf = pf.sort_values(["Date", "_rank"]).drop(columns=["_rank"])

    price_feats = ["log_return_1d", "log_return_5d", "log_return_21d", "realized_vol_21d"]
    price_pivot = pf.pivot_table(
        index="Date", columns="Ticker", values=price_feats, aggfunc="first"
    )
    price_pivot = price_pivot.reorder_levels([1, 0], axis=1)
    ordered_cols = [(ticker, feat) for ticker in tickers for feat in price_feats]
    price_pivot = price_pivot[ordered_cols]

    # Append sentinel features: days_since_filing=0, has_sentiment_data=0
    for ticker in tickers:
        price_pivot[(ticker, "days_since_filing")] = 0
        price_pivot[(ticker, "has_sentiment_data")] = 0

    # Reorder to match PRICE_REGIME_FEATURES block order
    full_cols = [(ticker, feat) for ticker in tickers for feat in _PRICE_REGIME_FEATURES]
    price_pivot = price_pivot[full_cols]

    # --- Forward returns pivot -----------------------------------------------
    return_pivot = pf.pivot_table(
        index="Date", columns="Ticker", values="simple_return_1d", aggfunc="first"
    )[tickers]
    forward_returns = return_pivot.shift(-1).fillna(0.0)

    # --- Align all on common dates -------------------------------------------
    common_dates = (
        price_pivot.index
        .intersection(shared_df.index)
        .intersection(forward_returns.index)
    )
    common_dates = common_dates.sort_values()

    price_matrix  = price_pivot.loc[common_dates].values.astype(np.float32)   # (T, 30)
    shared_matrix = shared_df.loc[common_dates].values.astype(np.float32)     # (T, 4)
    return_matrix = forward_returns.loc[common_dates].values.astype(np.float32)  # (T, 5)

    obs_array = np.hstack([price_matrix, shared_matrix])  # (T, 34)
    obs_array     = np.nan_to_num(obs_array,     nan=0.0, posinf=0.0, neginf=0.0)
    return_matrix = np.nan_to_num(return_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return obs_array, return_matrix, common_dates.values


# ---------------------------------------------------------------------------
# Fold data builder
# ---------------------------------------------------------------------------

def build_fold_data(
    fold_cfg: dict,
    config: dict,
    tickers: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Load, process, and split data for a single walk-forward fold.

    Parameters
    ----------
    fold_cfg : dict
        A single fold entry from ``config["evaluation"]["folds"]``.
    config : dict
        Full config (provides data paths, env settings, etc.).
    tickers : list of str

    Returns
    -------
    train_obs, train_ret, val_obs, val_ret, test_obs, test_ret : np.ndarray
        Normalised observation matrices and raw forward-return arrays.
        Normalizer is fit on train split only; val and test use train stats.
    test_dates : np.ndarray of np.datetime64
    normalizer : src.feature_engineer.Normalizer
    """
    import yaml

    from src.feature_engineer import (
        Normalizer,
        build_observation_matrix,
        compute_price_features,
        fetch_price_data,
    )

    model_type = fold_cfg.get("model_type", "full")
    train_start = fold_cfg["train_start"]
    train_end   = fold_cfg["train_end"]
    val_start   = fold_cfg["val_start"]
    val_end     = fold_cfg["val_end"]
    test_start  = fold_cfg["test_start"]
    test_end    = fold_cfg["test_end"]

    full_start = train_start
    full_end   = test_end

    # --- Price data (common to both model types) ----------------------------
    prices = fetch_price_data(tickers, start_date=train_start, end_date=full_end)
    price_features_df = compute_price_features(prices, tickers)

    if model_type == "regime_only":
        # --- Regime-only (Fold R) -------------------------------------------
        regime_path = config["data"]["regime_path"]
        if not os.path.exists(regime_path):
            raise FileNotFoundError(
                f"Regime file not found: {regime_path}\n"
                "Run: git submodule update --init --recursive"
            )
        regime_raw = pd.read_csv(regime_path)
        regime_raw["Date"] = pd.to_datetime(regime_raw["Date"])

        def _split_regime_obs(start, end):
            pf_split = price_features_df[
                (pd.to_datetime(price_features_df["Date"]) >= start) &
                (pd.to_datetime(price_features_df["Date"]) <= end)
            ]
            return _build_regime_only_obs_matrix(
                regime_raw, pf_split, tickers, start, end,
            )

        train_obs, train_ret, _  = _split_regime_obs(train_start, train_end)
        val_obs,   val_ret,   _  = _split_regime_obs(val_start,   val_end)
        test_obs,  test_ret, test_dates = _split_regime_obs(test_start, test_end)

    else:
        # --- Full model (Folds 1 and 2) -------------------------------------
        from src.data_loader import load_merged_features

        merged_df = load_merged_features(config)

        def _filter_merged(start, end):
            mask = (
                (merged_df["Date"] >= start) & (merged_df["Date"] <= end)
            )
            return merged_df[mask].reset_index(drop=True)

        train_merged = _filter_merged(train_start, train_end)
        val_merged   = _filter_merged(val_start,   val_end)
        test_merged  = _filter_merged(test_start,  test_end)

        train_obs, train_ret, _            = build_observation_matrix(
            train_merged, price_features_df, tickers
        )
        val_obs,   val_ret,   _            = build_observation_matrix(
            val_merged,   price_features_df, tickers
        )
        test_obs,  test_ret,  test_dates   = build_observation_matrix(
            test_merged,  price_features_df, tickers
        )

    # --- Fit normalizer on train split only ----------------------------------
    normalizer = Normalizer()
    train_obs = normalizer.fit_transform(train_obs)
    val_obs   = normalizer.transform(val_obs)
    test_obs  = normalizer.transform(test_obs)

    return train_obs, train_ret, val_obs, val_ret, test_obs, test_ret, test_dates, normalizer


# ---------------------------------------------------------------------------
# Train single seed for a fold
# ---------------------------------------------------------------------------

def train_fold_seed(
    fold_cfg: dict,
    config: dict,
    train_obs: np.ndarray,
    train_ret: np.ndarray,
    val_obs: np.ndarray,
    val_ret: np.ndarray,
    seed: int,
    fold_out_dir: str,
    use_tuned: bool = False,
) -> object:
    """
    Train a PPO model for one fold / seed combination.

    Returns the trained SB3 PPO model (pointing at the best checkpoint).
    """
    import copy as _copy
    import shutil

    import torch
    import yaml
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from src.portfolio_env import PortfolioEnv
    from src.ppo_agent import make_callbacks, make_ppo

    torch.set_num_threads(1)

    fold_config = _copy.deepcopy(config)

    # --- Override obs_dim for regime-only folds ------------------------------
    if "obs_dim" in fold_cfg:
        fold_config["env"]["obs_dim"] = fold_cfg["obs_dim"]
    if "n_per_ticker_features_override" in fold_cfg:
        fold_config["env"]["n_per_ticker_features"] = fold_cfg["n_per_ticker_features_override"]

    # --- Override seed --------------------------------------------------------
    fold_config["training"]["seed"] = seed

    # --- Apply tuned hyperparameters (optional) --------------------------------
    if use_tuned:
        tuned_path = Path(config["output"]["dir"]) / "best_hparams.yaml"
        if tuned_path.exists():
            with open(tuned_path) as f:
                tuned = yaml.safe_load(f)
            if "tc_penalty_lambda" in tuned:
                fold_config["env"]["transaction_cost"] = tuned.pop("tc_penalty_lambda")
            for k in ("lambda_dd", "dd_threshold", "lambda_div", "max_weight_per_ticker"):
                if k in tuned:
                    fold_config["env"][k] = tuned.pop(k)
            fold_config["ppo"].update(tuned)

    os.makedirs(fold_out_dir, exist_ok=True)

    # --- Build environments ---------------------------------------------------
    def _make_env(obs, ret, cfg):
        _obs = obs.copy()
        _ret = ret.copy()
        _cfg = _copy.deepcopy(cfg)
        return lambda: Monitor(PortfolioEnv(_obs, _ret, _cfg))

    train_env = DummyVecEnv([_make_env(train_obs, train_ret, fold_config)])
    train_cfg = fold_config.get("training", {})
    train_env = VecNormalize(
        train_env,
        norm_obs=False,
        norm_reward=bool(train_cfg.get("norm_reward", True)),
        clip_reward=float(train_cfg.get("clip_reward", 10.0)),
    )

    val_env = DummyVecEnv([_make_env(val_obs, val_ret, fold_config)])
    val_env = VecNormalize(
        val_env, norm_obs=False, norm_reward=False, training=False,
    )

    model = make_ppo(train_env, fold_config, seed=seed)
    callbacks = make_callbacks(val_env, fold_config, fold_out_dir)

    total_ts = int(fold_config["ppo"]["total_timesteps"])
    model.learn(total_timesteps=total_ts, callback=callbacks, reset_num_timesteps=True)

    # --- Load best checkpoint ------------------------------------------------
    best_path = Path(fold_out_dir) / "best_model.zip"
    if best_path.exists():
        model = PPO.load(str(best_path), env=train_env)

    # --- Save seed-specific copy ---------------------------------------------
    seed_path = str(Path(fold_out_dir) / f"best_model_seed_{seed}.zip")
    shutil.copy2(str(best_path), seed_path)

    return model


# ---------------------------------------------------------------------------
# Evaluate a list of models on test data (ensemble or single-seed)
# ---------------------------------------------------------------------------

def evaluate_fold_models(
    models: List[object],
    test_obs: np.ndarray,
    test_ret: np.ndarray,
    fold_config: dict,
) -> Dict:
    """
    Run an ensemble of trained models on test data and return metrics.

    If ``models`` contains more than one model, actions are averaged across
    models (ensemble mean logit → constrained softmax), matching the
    approach in ``src.evaluate``.

    Returns
    -------
    dict with keys: annualised_return, sharpe, sortino, max_drawdown,
                    calmar, mean_daily_turnover, daily_returns (list),
                    portfolio_weights (list-of-lists)
    """
    from src.portfolio_env import PortfolioEnv

    env = PortfolioEnv(test_obs, test_ret, fold_config)
    obs, _ = env.reset()

    all_returns: List[float] = []
    all_weights: List[np.ndarray] = []
    terminated = False

    while not terminated:
        if len(models) == 1:
            action, _ = models[0].predict(obs, deterministic=True)
        else:
            actions = np.stack([m.predict(obs, deterministic=True)[0] for m in models])
            action = actions.mean(axis=0)

        obs, _, terminated, _, info = env.step(action)
        all_returns.append(info["portfolio_return"])
        all_weights.append(info["weights"])

    daily_returns = np.array(all_returns, dtype=np.float64)
    weights_arr   = np.array(all_weights, dtype=np.float32)

    ann_ret = _annualised_return(daily_returns)
    sharpe  = _annualised_sharpe(daily_returns)
    sortino = _annualised_sortino(daily_returns)
    mdd     = _max_drawdown(daily_returns)
    calmar  = ann_ret / mdd if mdd > 1e-10 else float("inf")
    turnover = float(np.mean(np.sum(np.abs(np.diff(weights_arr, axis=0)), axis=1))) if len(weights_arr) > 1 else 0.0

    return {
        "annualised_return":    round(ann_ret, 6),
        "sharpe":               round(sharpe,  6),
        "sortino":              round(sortino, 6),
        "max_drawdown":         round(mdd,     6),
        "calmar":               round(calmar,  6),
        "mean_daily_turnover":  round(turnover, 6),
        "daily_returns":        daily_returns.tolist(),
        "portfolio_weights":    weights_arr.tolist(),
    }


# ---------------------------------------------------------------------------
# Metric helpers (duplicated from evaluate.py to avoid circular imports)
# ---------------------------------------------------------------------------

def _annualised_return(r: np.ndarray) -> float:
    total = float(np.prod(1.0 + r) - 1.0)
    n = len(r)
    return float((1.0 + total) ** (TRADING_DAYS / max(n, 1)) - 1.0)


def _annualised_sharpe(r: np.ndarray, rf: float = 0.0) -> float:
    excess = r - rf / TRADING_DAYS
    std = float(np.std(excess))
    if std < 1e-10:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(TRADING_DAYS))


def _annualised_sortino(r: np.ndarray, rf: float = 0.0) -> float:
    excess = r - rf / TRADING_DAYS
    down = excess[excess < 0]
    if len(down) == 0 or np.std(down) < 1e-10:
        return float("inf")
    return float(np.mean(excess) / np.std(down) * np.sqrt(TRADING_DAYS))


def _max_drawdown(r: np.ndarray) -> float:
    cum = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(cum)
    return float(((peak - cum) / peak).max())


# ---------------------------------------------------------------------------
# Cross-fold summary statistics
# ---------------------------------------------------------------------------

def compute_cross_fold_summary(fold_results: List[Dict]) -> Dict:
    """
    Aggregate per-fold metrics into cross-fold mean ± std statistics.

    Parameters
    ----------
    fold_results : list of dict
        Each dict must have keys: fold_id, sharpe, sortino, max_drawdown,
        annualised_return, calmar, mean_daily_turnover.

    Returns
    -------
    dict with per-metric mean, std, and per-fold values.
    """
    metrics = ["sharpe", "sortino", "max_drawdown", "annualised_return",
               "calmar", "mean_daily_turnover"]

    summary: Dict = {"per_fold": fold_results, "aggregate": {}}
    for m in metrics:
        vals = [r[m] for r in fold_results if m in r and r[m] != float("inf")]
        if vals:
            summary["aggregate"][m] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)),  4),
                "min":  round(float(np.min(vals)),  4),
                "max":  round(float(np.max(vals)),  4),
            }
    return summary


# ---------------------------------------------------------------------------
# Top-level runner  (one fold, multiple seeds)
# ---------------------------------------------------------------------------

def run_fold(
    fold_cfg: dict,
    config: dict,
    seeds: List[int],
    use_tuned: bool = False,
    output_root: str = "output/walk_forward",
) -> Dict:
    """
    Run a full train-eval cycle for one fold across all seeds.

    Trains `len(seeds)` models, evaluates the ensemble, saves metrics.

    Returns
    -------
    dict — fold-level metrics (ensemble and per-seed Sharpes).
    """
    import copy as _copy

    fold_id = fold_cfg["fold_id"]
    tickers = config["tickers"]
    fold_out_dir = str(Path(output_root) / f"fold_{fold_id}")
    os.makedirs(fold_out_dir, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  FOLD {fold_id}: {fold_cfg.get('description', '')}")
    print(f"  Train: {fold_cfg['train_start']} → {fold_cfg['train_end']}")
    print(f"  Val:   {fold_cfg['val_start']}   → {fold_cfg['val_end']}")
    print(f"  Test:  {fold_cfg['test_start']}  → {fold_cfg['test_end']}")
    print(f"  Seeds: {seeds}  |  use_tuned={use_tuned}")
    print(f"{'='*72}")

    # --- Build fold data (shared across seeds) --------------------------------
    print("\nBuilding fold data ...")
    (train_obs, train_ret, val_obs, val_ret,
     test_obs, test_ret, test_dates, normalizer) = build_fold_data(
        fold_cfg, config, tickers
    )
    print(f"  Train: {train_obs.shape[0]} steps | Val: {val_obs.shape[0]} steps "
          f"| Test: {test_obs.shape[0]} steps")

    # Save normalizer
    normalizer.save(str(Path(fold_out_dir) / "normalizer.pkl"))

    # --- Build fold-specific config override ---------------------------------
    fold_config = _copy.deepcopy(config)
    if "obs_dim" in fold_cfg:
        fold_config["env"]["obs_dim"] = fold_cfg["obs_dim"]
    if "n_per_ticker_features_override" in fold_cfg:
        fold_config["env"]["n_per_ticker_features"] = fold_cfg["n_per_ticker_features_override"]

    # --- Train each seed -----------------------------------------------------
    trained_models = []
    per_seed_sharpes: List[float] = []

    for seed in seeds:
        seed_out = str(Path(fold_out_dir) / f"seed_{seed}")
        print(f"\nTraining seed {seed} ...")
        model = train_fold_seed(
            fold_cfg, config, train_obs, train_ret, val_obs, val_ret,
            seed=seed, fold_out_dir=seed_out, use_tuned=use_tuned,
        )
        trained_models.append(model)

        # Quick single-seed eval on test
        seed_metrics = evaluate_fold_models([model], test_obs, test_ret, fold_config)
        per_seed_sharpes.append(seed_metrics["sharpe"])
        print(f"  Seed {seed} test Sharpe: {seed_metrics['sharpe']:.4f}  "
              f"MDD: {seed_metrics['max_drawdown']:.4f}")

    # --- Ensemble evaluation --------------------------------------------------
    print("\nEvaluating ensemble ...")
    ensemble_metrics = evaluate_fold_models(trained_models, test_obs, test_ret, fold_config)
    print(f"  Ensemble Sharpe: {ensemble_metrics['sharpe']:.4f}  "
          f"MDD: {ensemble_metrics['max_drawdown']:.4f}  "
          f"AnnRet: {ensemble_metrics['annualised_return']:.4f}")

    # --- Assemble fold result -------------------------------------------------
    result = {
        "fold_id":               fold_id,
        "description":           fold_cfg.get("description", ""),
        "model_type":            fold_cfg.get("model_type", "full"),
        "test_start":            fold_cfg["test_start"],
        "test_end":              fold_cfg["test_end"],
        "n_test_days":           len(test_ret),
        "seeds":                 seeds,
        "per_seed_sharpes":      [round(s, 4) for s in per_seed_sharpes],
        **{f"ensemble_{k}": v
           for k, v in ensemble_metrics.items()
           if k not in ("daily_returns", "portfolio_weights")},
        # expose top-level aliases for cross_fold_summary
        "sharpe":                ensemble_metrics["sharpe"],
        "sortino":               ensemble_metrics["sortino"],
        "max_drawdown":          ensemble_metrics["max_drawdown"],
        "annualised_return":     ensemble_metrics["annualised_return"],
        "calmar":                ensemble_metrics["calmar"],
        "mean_daily_turnover":   ensemble_metrics["mean_daily_turnover"],
    }

    # --- Save fold metrics to JSON -------------------------------------------
    metrics_path = str(Path(fold_out_dir) / "fold_metrics.json")
    # Remove non-serialisable arrays from the result before saving
    saveable_ensemble = {k: v for k, v in ensemble_metrics.items()
                         if k not in ("daily_returns", "portfolio_weights")}
    with open(metrics_path, "w") as f:
        json.dump({**result, "ensemble_metrics": saveable_ensemble}, f, indent=2, default=str)
    print(f"\nFold metrics saved -> {metrics_path}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    config_path: str = "config.yaml",
    fold_ids: Optional[List] = None,
    seeds: Optional[List[int]] = None,
    use_tuned: bool = False,
    output_root: str = "output/walk_forward",
) -> Dict:
    """
    Run walk-forward cross-validation across all configured folds.

    Parameters
    ----------
    config_path : str
    fold_ids : list or None
        Fold IDs to run (1, 2, "R", or mix).  None = all folds.
    seeds : list of int or None
        Training seeds for ensemble.  None = use [config["training"]["seed"]].
    use_tuned : bool
        Load hyperparameters from output/best_hparams.yaml.
    output_root : str
        Base directory for fold outputs.

    Returns
    -------
    dict — cross-fold summary.
    """
    from src.data_loader import load_config

    config = load_config(config_path)
    if seeds is None:
        seeds = [int(config["training"]["seed"])]

    all_folds = config.get("evaluation", {}).get("folds", [])
    if not all_folds:
        raise ValueError("No folds defined in config.yaml under evaluation.folds")

    # Filter by requested fold_ids
    if fold_ids is not None:
        fold_ids_str = [str(f) for f in fold_ids]
        all_folds = [f for f in all_folds if str(f["fold_id"]) in fold_ids_str]

    os.makedirs(output_root, exist_ok=True)

    fold_results: List[Dict] = []
    for fold_cfg in all_folds:
        result = run_fold(
            fold_cfg, config, seeds=seeds,
            use_tuned=use_tuned, output_root=output_root,
        )
        fold_results.append(result)

    # --- Cross-fold summary --------------------------------------------------
    summary = compute_cross_fold_summary(fold_results)
    summary_path = str(Path(output_root) / "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*72}")
    print("CROSS-FOLD SUMMARY")
    print(f"{'='*72}")
    agg = summary["aggregate"]
    print(f"  {'Metric':<22}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for m in ["sharpe", "sortino", "max_drawdown", "annualised_return", "calmar"]:
        if m in agg:
            v = agg[m]
            print(f"  {m:<22}  {v['mean']:>8.4f}  {v['std']:>8.4f}  "
                  f"{v['min']:>8.4f}  {v['max']:>8.4f}")
    print(f"\nCross-fold summary saved -> {summary_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Walk-forward cross-validation for PPO portfolio agent"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fold", nargs="+", default=None,
                        help="Fold IDs to run (e.g. --fold 1 2 R)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Training seeds (e.g. --seeds 42 1 7 2026)")
    parser.add_argument("--use-tuned", action="store_true",
                        help="Load hyperparameters from output/best_hparams.yaml")
    parser.add_argument("--output-root", default="output/walk_forward")
    args = parser.parse_args()

    main(
        config_path=args.config,
        fold_ids=args.fold,
        seeds=args.seeds,
        use_tuned=args.use_tuned,
        output_root=args.output_root,
    )
