"""
ppo_agent.py — PPO agent factory and training callbacks.

Wraps stable-baselines3 PPO with the hyper-parameters from config.yaml.
"""

from __future__ import annotations

from typing import Callable, List

import torch

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
        EvalCallback,
    )
except ImportError as exc:
    raise ImportError(
        "stable-baselines3 >= 2.3.0 is required.  "
        "Install with: pip install stable-baselines3"
    ) from exc

from src.portfolio_env import PortfolioEnv


# ---------------------------------------------------------------------------
# Activation function map
# ---------------------------------------------------------------------------

_ACTIVATION_FN_MAP = {
    "Tanh": torch.nn.Tanh,
    "ReLU": torch.nn.ReLU,
    "ELU":  torch.nn.ELU,
}


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Return a callable that linearly anneals the learning rate from
    *initial_value* down to 0 over the course of training.

    The returned function maps ``progress_remaining`` (1.0 → 0.0) to the
    current learning rate.
    """
    def fn(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return fn


# ---------------------------------------------------------------------------
# PPO factory
# ---------------------------------------------------------------------------

def make_ppo(env: PortfolioEnv, config: dict, seed: int = 0) -> PPO:
    """
    Instantiate a PPO agent with hyper-parameters from ``config["ppo"]``.

    Parameters
    ----------
    env : PortfolioEnv
        Training environment.
    config : dict
        Loaded config.yaml.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    stable_baselines3.PPO
    """
    ppo_cfg = config["ppo"]

    activation_name = ppo_cfg["policy_kwargs"]["activation_fn"]
    if activation_name not in _ACTIVATION_FN_MAP:
        raise ValueError(
            f"Unknown activation_fn '{activation_name}'.  "
            f"Choose from: {list(_ACTIVATION_FN_MAP)}"
        )

    policy_kwargs = {
        "net_arch":      ppo_cfg["policy_kwargs"]["net_arch"],
        "activation_fn": _ACTIVATION_FN_MAP[activation_name],
    }

    # Learning rate: flat float or linear annealing schedule
    lr_value = float(ppo_cfg["learning_rate"])
    lr_schedule_name = ppo_cfg.get("lr_schedule", "constant")
    if lr_schedule_name == "linear":
        learning_rate = linear_schedule(lr_value)
    else:
        learning_rate = lr_value

    return PPO(
        policy         = ppo_cfg["policy"],
        env            = env,
        learning_rate  = learning_rate,
        n_steps        = int(ppo_cfg["n_steps"]),
        batch_size     = int(ppo_cfg["batch_size"]),
        n_epochs       = int(ppo_cfg["n_epochs"]),
        gamma          = float(ppo_cfg["gamma"]),
        gae_lambda     = float(ppo_cfg["gae_lambda"]),
        clip_range     = float(ppo_cfg["clip_range"]),
        ent_coef       = float(ppo_cfg["ent_coef"]),
        vf_coef        = float(ppo_cfg["vf_coef"]),
        max_grad_norm  = float(ppo_cfg["max_grad_norm"]),
        policy_kwargs  = policy_kwargs,
        seed           = seed,
        verbose        = 1,
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def make_callbacks(
    val_env: PortfolioEnv,
    config: dict,
    model_dir: str,
) -> List[BaseCallback]:
    """
    Build the standard callback list for training.

    Includes:
      - EvalCallback  : evaluates on val_env every eval_freq steps, saves
                        best model to model_dir/best_model.zip
      - CheckpointCallback : saves a checkpoint every checkpoint_freq steps
      - RegimeSummaryCallback : logs mean weight per regime label each eval

    Parameters
    ----------
    val_env : PortfolioEnv
        Validation environment (already normalised; not wrapped in Monitor).
    config : dict
        Loaded config.yaml.
    model_dir : str
        Directory where model files are saved.

    Returns
    -------
    list of BaseCallback
    """
    ppo_cfg = config["ppo"]

    eval_cb = EvalCallback(
        eval_env          = val_env,
        best_model_save_path = model_dir,
        log_path          = model_dir,
        eval_freq         = int(ppo_cfg["eval_freq"]),
        n_eval_episodes   = 1,
        deterministic     = True,
        render            = False,
        verbose           = 1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq  = int(ppo_cfg["checkpoint_freq"]),
        save_path  = model_dir,
        name_prefix = "ppo_portfolio",
        verbose    = 1,
    )

    return [eval_cb, checkpoint_cb]
