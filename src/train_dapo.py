"""
train_dapo.py — CLI entry point for DAPO portfolio agent training.

Mirrors the interface of ``src/train.py`` but uses the DAPO trainer instead
of SB3 PPO.

Usage::

    python -m src.train_dapo                        # default config
    python -m src.train_dapo --config config.yaml --seed 42 --use-tuned
    python -m src.train_dapo --fold 1               # run a specific fold
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path


def main(
    config_path: str = "config.yaml",
    use_tuned: bool = False,
    seed: int | None = None,
    fold_id: str | None = None,
) -> None:
    import yaml

    from src.data_loader import load_config, load_merged_features, split_data
    from src.dapo_trainer import DAPOTrainer
    from src.feature_engineer import (
        Normalizer,
        build_observation_matrix,
        compute_price_features,
        fetch_price_data,
    )

    # 1. Load config -----------------------------------------------------------
    config = load_config(config_path)
    tickers = config["tickers"]
    out_cfg = config["output"]
    env_cfg = config["env"]

    # 1b. Override seed -------------------------------------------------------
    if seed is not None:
        config["training"]["seed"] = seed
    active_seed = int(config["training"]["seed"])

    # 1c. Apply tuned hyperparameters (if requested) --------------------------
    if use_tuned:
        tuned_path = Path(out_cfg["dir"]) / "best_hparams.yaml"
        if not tuned_path.exists():
            raise FileNotFoundError(
                f"Tuned params not found at {tuned_path}. "
                "Run `python -m src.tune` first."
            )
        with open(tuned_path) as f:
            tuned = yaml.safe_load(f)
        if "tc_penalty_lambda" in tuned:
            config["env"]["transaction_cost"] = tuned.pop("tc_penalty_lambda")
        for k in ("lambda_dd", "dd_threshold", "lambda_div", "max_weight_per_ticker"):
            if k in tuned:
                config["env"][k] = tuned.pop(k)
        config["ppo"].update(tuned)
        print(f"Loaded tuned params from {tuned_path}")

    # 2. Determine date splits ------------------------------------------------
    if fold_id is not None:
        # Use fold-specific dates from config.yaml
        folds = config.get("evaluation", {}).get("folds", [])
        fold_cfg = next(
            (f for f in folds if str(f["fold_id"]) == str(fold_id)), None
        )
        if fold_cfg is None:
            raise ValueError(
                f"Fold '{fold_id}' not found in config.yaml evaluation.folds"
            )
        splits = {
            "train_start": fold_cfg["train_start"],
            "train_end":   fold_cfg["train_end"],
            "val_start":   fold_cfg["val_start"],
            "val_end":     fold_cfg["val_end"],
            "test_start":  fold_cfg.get("test_start", fold_cfg["val_end"]),
            "test_end":    fold_cfg.get("test_end",   fold_cfg["val_end"]),
        }
        # Override obs_dim if the fold specifies one
        if "obs_dim" in fold_cfg:
            config["env"]["obs_dim"] = fold_cfg["obs_dim"]
        out_dir = str(Path(out_cfg["dir"]) / f"dapo_fold_{fold_id}_seed_{active_seed}")
    else:
        splits = config["splits"]
        out_dir = str(Path(out_cfg["dir"]) / f"dapo_seed_{active_seed}")

    os.makedirs(out_dir, exist_ok=True)

    # 3. Load and process data ------------------------------------------------
    print("Loading merged features ...")
    merged_df = load_merged_features(config)

    # Trim to train/val date range
    from src.data_loader import split_data as _split_data
    config_for_split = copy.deepcopy(config)
    config_for_split["splits"] = splits
    train_df, val_df, _ = _split_data(merged_df, config_for_split)

    print("Fetching price data via yfinance ...")
    prices = fetch_price_data(
        tickers,
        start_date=splits["train_start"],
        end_date=splits["val_end"],
    )

    print("Computing price features ...")
    price_features = compute_price_features(prices, tickers)

    print("Building observation matrices ...")
    train_obs, train_ret, _ = build_observation_matrix(
        train_df, price_features, tickers
    )
    val_obs, val_ret, _ = build_observation_matrix(
        val_df, price_features, tickers
    )

    print(f"  Train: {train_obs.shape[0]} steps  |  Val: {val_obs.shape[0]} steps")

    # 4. Normalise ------------------------------------------------------------
    normalizer = Normalizer()
    train_obs = normalizer.fit_transform(train_obs)
    val_obs   = normalizer.transform(val_obs)
    normalizer.save(str(Path(out_dir) / "normalizer.pkl"))

    # 5. Instantiate DAPOTrainer and train ------------------------------------
    obs_dim    = int(env_cfg["obs_dim"])
    action_dim = int(env_cfg["action_dim"])

    print(f"Initialising DAPOTrainer (obs_dim={obs_dim}, action_dim={action_dim}) ...")
    trainer = DAPOTrainer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config,
        seed=active_seed,
    )

    trainer.train(
        train_obs, train_ret,
        val_obs,   val_ret,
        out_dir=out_dir,
    )

    print(f"\nDAPO training complete.  Outputs saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DAPO portfolio agent"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--use-tuned", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--fold", default=None,
        help="Fold ID (1, 2, or R) to use fold-specific date splits"
    )
    args = parser.parse_args()
    main(
        config_path=args.config,
        use_tuned=args.use_tuned,
        seed=args.seed,
        fold_id=args.fold,
    )
