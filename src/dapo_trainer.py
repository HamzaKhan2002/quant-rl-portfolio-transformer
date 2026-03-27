"""
dapo_trainer.py — Direct Advantage Policy Optimization (DAPO) for portfolio allocation.

DAPO is a critic-free policy gradient algorithm that replaces GAE (used in PPO)
with group-relative advantage estimation.  For each observation at time t, K
actions are sampled from the current policy; the portfolio return is computed
for each action using the realized asset returns at t; and the group-relative
advantage normalises each action's reward by the group mean and standard deviation.

Key differences from PPO:
  - No value function (critic)     — eliminates variance-bias tradeoff in baseline
  - No GAE                         — advantages are state-local, not trajectory-level
  - Group-relative normalisation   — automatic advantage scale without running stats
  - Asymmetric clipping            — ε_low ≠ ε_high; more room to increase
                                     probability of truly good actions
  - Dynamic group filtering        — skip groups with near-zero return variance
                                     (no learning signal)

Reference (adapted from LLM discrete to continuous portfolio):
  Shao et al. (2025) "DAPO: An Open-Source LLM Reinforcement Learning System"
  arXiv:2503.14476

Architecture:
  GaussianPolicy: obs_dim -> [256, 256] -> (μ, log_σ)  (action_dim each)
  Action: a ~ N(μ, σ),  w = constrained_softmax(a, max_weight)

Usage::

    from src.dapo_trainer import DAPOTrainer
    trainer = DAPOTrainer(obs_dim=74, action_dim=6, config=config, seed=42)
    trainer.train(train_obs, train_ret, val_obs, val_ret, out_dir="output/dapo")
    action, _ = trainer.predict(obs, deterministic=True)
    trainer.save("output/dapo/policy.pt")
    trainer2 = DAPOTrainer.load("output/dapo/policy.pt", config=config)
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Normal
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for DAPOTrainer.  "
        "Install with: pip install torch"
    ) from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_STD_MIN = -5.0   # clamp for numerical stability
LOG_STD_MAX = 2.0    # max σ ≈ 7.4 (prevents initial over-exploration)
STD_FLOOR   = 0.01   # minimum group std for advantage normalisation


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class GaussianPolicy(nn.Module):
    """
    Gaussian policy: obs → (μ, log_σ).

    Parameters
    ----------
    obs_dim : int
    action_dim : int
    net_arch : list of int
        Hidden layer sizes (default [256, 256]).
    activation : str
        Hidden activation function ("Tanh" or "ReLU").
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        net_arch: List[int] = None,
        activation: str = "Tanh",
    ) -> None:
        super().__init__()
        if net_arch is None:
            net_arch = [256, 256]

        act_cls = nn.Tanh if activation == "Tanh" else nn.ReLU

        layers: List[nn.Module] = []
        in_dim = obs_dim
        for hidden_size in net_arch:
            layers += [nn.Linear(in_dim, hidden_size), act_cls()]
            in_dim = hidden_size

        self.shared_net = nn.Sequential(*layers)
        self.mu_head     = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)

        # Initialise output heads with small weights for stable initial actions
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.constant_(self.log_std_head.bias, -1.0)  # σ ≈ 0.37 initially

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (μ, log_σ) for a batch of observations."""
        features = self.shared_net(obs)
        mu      = self.mu_head(features)
        log_std = self.log_std_head(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(
        self, obs: torch.Tensor, n: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample n actions per observation.

        Parameters
        ----------
        obs : torch.Tensor, shape (obs_dim,) or (batch, obs_dim)
        n : int — number of samples per observation

        Returns
        -------
        actions  : torch.Tensor, shape (n, action_dim) or (batch, n, action_dim)
        log_probs: torch.Tensor, shape (n,) or (batch, n)
        """
        mu, log_std = self.forward(obs)  # (action_dim,) or (batch, action_dim)
        std = log_std.exp()
        if obs.ndim == 1:
            # Single observation: mu is (action_dim,)
            dist = Normal(mu.unsqueeze(0).expand(n, -1),
                          std.unsqueeze(0).expand(n, -1))
            actions  = dist.rsample()               # (n, action_dim)
            log_probs = dist.log_prob(actions).sum(-1)  # (n,)
        else:
            # Batch: mu is (batch, action_dim)
            batch = mu.shape[0]
            mu_exp  = mu.unsqueeze(1).expand(batch, n, -1)
            std_exp = std.unsqueeze(1).expand(batch, n, -1)
            dist = Normal(mu_exp, std_exp)
            actions   = dist.rsample()                  # (batch, n, action_dim)
            log_probs = dist.log_prob(actions).sum(-1)  # (batch, n)
        return actions, log_probs

    def log_prob(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Log probability of ``actions`` under current policy.

        Parameters
        ----------
        obs     : torch.Tensor, shape (batch, obs_dim)
        actions : torch.Tensor, shape (batch, action_dim)

        Returns
        -------
        torch.Tensor, shape (batch,)
        """
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        return dist.log_prob(actions).sum(-1)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Differential entropy of N(μ, σ): sum_i [log_σ_i + 0.5 * log(2πe)].

        Returns torch.Tensor, shape (batch,).
        """
        _, log_std = self.forward(obs)
        return (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum(-1)


# ---------------------------------------------------------------------------
# DAPO rollout collection
# ---------------------------------------------------------------------------

def collect_dapo_rollout(
    policy: GaussianPolicy,
    obs_array: np.ndarray,
    forward_return_array: np.ndarray,
    n_tickers: int,
    max_weight: float,
    K: int = 8,
    filter_min_std: float = STD_FLOOR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one DAPO rollout over the full episode.

    At each time step t:
      1. Sample K actions from π(·|o_t)  (all from same state, no env advance)
      2. Compute portfolio return for each action using realized asset returns
      3. Group-relative advantage: A_k = (r_k − mean) / max(std, filter_min_std)
      4. Skip groups with std < filter_min_std (zero-signal groups)

    The episode advances along the first-sample trajectory (committed action).

    Parameters
    ----------
    policy : GaussianPolicy
    obs_array : np.ndarray, shape (T+1, obs_dim)   normalised observations
    forward_return_array : np.ndarray, shape (T+1, n_tickers)
    n_tickers : int
    max_weight : float   max weight per asset for constrained_softmax
    K : int   group size (default 8)
    filter_min_std : float   minimum group std to keep (default STD_FLOOR)

    Returns
    -------
    obs_out    : np.ndarray, shape (M, obs_dim)
    act_out    : np.ndarray, shape (M, action_dim)
    logp_out   : np.ndarray, shape (M,)
    adv_out    : np.ndarray, shape (M,)

    where M = (number of kept groups) × K.
    """
    from src.portfolio_env import PortfolioEnv  # import here to avoid circularity

    T = obs_array.shape[0] - 1
    action_dim = forward_return_array.shape[1] + 1  # n_tickers + cash

    all_obs:   List[np.ndarray] = []
    all_acts:  List[np.ndarray] = []
    all_logps: List[float]      = []
    all_advs:  List[float]      = []

    policy.eval()
    with torch.no_grad():
        for t in range(T):
            obs_t  = obs_array[t]               # (obs_dim,)
            rets_t = forward_return_array[t]     # (n_tickers,)

            obs_tensor = torch.tensor(obs_t, dtype=torch.float32)

            # Sample K actions
            actions_k, logprobs_k = policy.sample(obs_tensor, n=K)
            # actions_k : (K, action_dim)  logprobs_k : (K,)

            actions_np  = actions_k.numpy()    # (K, action_dim)
            logprobs_np = logprobs_k.numpy()   # (K,)

            # Portfolio return for each action (using realized asset returns)
            port_rets_k = np.zeros(K, dtype=np.float64)
            for k in range(K):
                w = PortfolioEnv._constrained_softmax(actions_np[k], max_weight)
                port_rets_k[k] = float(np.dot(w[:n_tickers], rets_t))

            # Dynamic filtering: skip groups with near-zero return variance
            group_std = float(port_rets_k.std())
            if group_std < filter_min_std:
                continue

            # Group-relative advantages
            group_mean = float(port_rets_k.mean())
            group_std_eff = max(group_std, filter_min_std)
            advantages_k = (port_rets_k - group_mean) / group_std_eff  # (K,)

            # Append K training samples for this timestep
            for k in range(K):
                all_obs.append(obs_t)
                all_acts.append(actions_np[k])
                all_logps.append(float(logprobs_np[k]))
                all_advs.append(float(advantages_k[k]))

    if len(all_obs) == 0:
        # All groups filtered — return empty arrays
        obs_dim = obs_array.shape[1]
        return (
            np.zeros((0, obs_dim), dtype=np.float32),
            np.zeros((0, action_dim), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    return (
        np.array(all_obs,  dtype=np.float32),
        np.array(all_acts, dtype=np.float32),
        np.array(all_logps, dtype=np.float32),
        np.array(all_advs, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# DAPO policy update
# ---------------------------------------------------------------------------

def dapo_update(
    policy: GaussianPolicy,
    optimizer: torch.optim.Optimizer,
    obs:         np.ndarray,
    actions:     np.ndarray,
    old_logprobs: np.ndarray,
    advantages:  np.ndarray,
    n_epochs:    int = 4,
    batch_size:  int = 256,
    eps_low:     float = 0.1,
    eps_high:    float = 0.2,
    ent_coef:    float = 0.01,
    max_grad_norm: float = 0.5,
) -> Dict[str, float]:
    """
    Perform DAPO policy update on a collected rollout batch.

    DAPO clipped loss (asymmetric):
        L = −min(ρ × A, clip(ρ, 1−ε_low, 1+ε_high) × A)

    This is equivalent to PPO clipping with asymmetric bounds:
    - Positive-advantage actions: allow up to ε_high probability increase
    - Negative-advantage actions: prevent more than ε_low probability decrease
    - In both cases, the clipping prevents overly large policy updates

    Parameters
    ----------
    policy, optimizer
    obs, actions, old_logprobs, advantages : np.ndarray
    n_epochs : int — gradient passes over the batch
    batch_size : int
    eps_low, eps_high : float — asymmetric clip bounds
    ent_coef : float — entropy regularisation coefficient
    max_grad_norm : float — gradient clipping

    Returns
    -------
    dict with keys: policy_loss, entropy_loss, total_loss, approx_kl
    """
    policy.train()

    obs_t    = torch.tensor(obs,          dtype=torch.float32)
    acts_t   = torch.tensor(actions,      dtype=torch.float32)
    oldlp_t  = torch.tensor(old_logprobs, dtype=torch.float32)
    adv_t    = torch.tensor(advantages,   dtype=torch.float32)

    N = len(obs_t)
    metrics = {"policy_loss": 0.0, "entropy_loss": 0.0,
               "total_loss": 0.0, "approx_kl": 0.0}

    for _ in range(n_epochs):
        # Shuffle
        perm = torch.randperm(N)
        obs_t   = obs_t[perm]
        acts_t  = acts_t[perm]
        oldlp_t = oldlp_t[perm]
        adv_t   = adv_t[perm]

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            obs_b   = obs_t[start:end]
            acts_b  = acts_t[start:end]
            oldlp_b = oldlp_t[start:end]
            adv_b   = adv_t[start:end]

            new_logprobs = policy.log_prob(obs_b, acts_b)
            entropy      = policy.entropy(obs_b).mean()

            ratio = torch.exp(new_logprobs - oldlp_b)
            clipped_ratio = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)

            policy_loss = -torch.min(
                ratio * adv_b, clipped_ratio * adv_b
            ).mean()

            total_loss = policy_loss - ent_coef * entropy

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

            metrics["policy_loss"]  += policy_loss.item()
            metrics["entropy_loss"] -= entropy.item()
            metrics["total_loss"]   += total_loss.item()
            metrics["approx_kl"]    += approx_kl

    n_batches = max(1, n_epochs * math.ceil(N / batch_size))
    return {k: v / n_batches for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Validation metric
# ---------------------------------------------------------------------------

def _eval_sharpe(
    policy: GaussianPolicy,
    obs_array: np.ndarray,
    forward_return_array: np.ndarray,
    n_tickers: int,
    max_weight: float,
    trading_days: int = 252,
) -> float:
    """Evaluate deterministic policy on val data; return annualised Sharpe."""
    policy.eval()
    port_returns = []

    with torch.no_grad():
        T = obs_array.shape[0] - 1
        for t in range(T):
            obs_tensor = torch.tensor(obs_array[t], dtype=torch.float32)
            mu, _ = policy(obs_tensor)
            action = mu.numpy()  # deterministic (mean)

            from src.portfolio_env import PortfolioEnv
            w = PortfolioEnv._constrained_softmax(action, max_weight)
            port_returns.append(float(np.dot(w[:n_tickers], forward_return_array[t])))

    r = np.array(port_returns, dtype=np.float64)
    std = float(np.std(r))
    if std < 1e-10:
        return 0.0
    return float(np.mean(r) / std * np.sqrt(trading_days))


# ---------------------------------------------------------------------------
# DAPOTrainer (main API)
# ---------------------------------------------------------------------------

class DAPOTrainer:
    """
    Standalone DAPO trainer for continuous portfolio allocation.

    Provides the same ``predict`` / ``save`` / ``load`` interface as SB3 PPO,
    enabling drop-in replacement in ``walk_forward.run_fold`` and
    ``evaluate.evaluate_fold_models``.

    Parameters
    ----------
    obs_dim : int
    action_dim : int
    config : dict
        Loaded config.yaml.
    seed : int
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: dict,
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.seed       = seed

        env_cfg  = config.get("env",  {})
        ppo_cfg  = config.get("ppo",  {})  # re-use architecture defaults
        dapo_cfg = config.get("dapo", {})

        # --- Policy architecture (shared with PPO) ----------------------------
        net_arch   = ppo_cfg.get("policy_kwargs", {}).get("net_arch", [256, 256])
        activation = ppo_cfg.get("policy_kwargs", {}).get("activation_fn", "Tanh")

        self.policy = GaussianPolicy(obs_dim, action_dim, net_arch, activation)

        # --- Hyperparameters --------------------------------------------------
        self.n_tickers  = int(env_cfg.get("n_tickers", 5))
        self.max_weight = float(env_cfg.get("max_weight_per_ticker", 1.0))

        self.K           = int(dapo_cfg.get("group_size",     8))
        self.eps_low     = float(dapo_cfg.get("eps_low",       0.1))
        self.eps_high    = float(dapo_cfg.get("eps_high",      0.2))
        self.ent_coef    = float(dapo_cfg.get("ent_coef",
                                  ppo_cfg.get("ent_coef", 0.01)))
        self.n_epochs    = int(dapo_cfg.get("n_epochs",
                                ppo_cfg.get("n_epochs", 4)))
        self.batch_size  = int(dapo_cfg.get("batch_size",
                                ppo_cfg.get("batch_size", 256)))
        self.max_grad_norm = float(dapo_cfg.get("max_grad_norm",
                                    ppo_cfg.get("max_grad_norm", 0.5)))
        self.learning_rate = float(dapo_cfg.get("learning_rate",
                                    ppo_cfg.get("learning_rate", 3e-4)))
        self.total_timesteps = int(dapo_cfg.get("total_timesteps",
                                    ppo_cfg.get("total_timesteps", 500_000)))
        self.filter_min_std = float(dapo_cfg.get("filter_min_std", STD_FLOOR))

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.learning_rate
        )
        self._best_val_sharpe = float("-inf")
        self._best_policy_state: Optional[dict] = None

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(
        self,
        train_obs: np.ndarray,
        train_ret: np.ndarray,
        val_obs:   np.ndarray,
        val_ret:   np.ndarray,
        out_dir:   str = "output/dapo",
        eval_freq: int = 10_000,
    ) -> List[Dict]:
        """
        Train the DAPO policy and save the best checkpoint.

        Parameters
        ----------
        train_obs, train_ret : np.ndarray
            Normalised observation matrix and forward-return array.
        val_obs, val_ret : np.ndarray
        out_dir : str
        eval_freq : int  — timesteps between val Sharpe evaluations

        Returns
        -------
        list of dict — training log (one entry per epoch).
        """
        os.makedirs(out_dir, exist_ok=True)
        train_log: List[Dict] = []

        timesteps_done = 0
        epoch = 0
        T_train = train_obs.shape[0] - 1

        print(f"[DAPO] Starting training | total_ts={self.total_timesteps:,} "
              f"| T_train={T_train} | K={self.K}")

        while timesteps_done < self.total_timesteps:
            # One rollout = one full pass over training data (K steps per state)
            obs_rb, act_rb, logp_rb, adv_rb = collect_dapo_rollout(
                self.policy,
                train_obs,
                train_ret,
                n_tickers=self.n_tickers,
                max_weight=self.max_weight,
                K=self.K,
                filter_min_std=self.filter_min_std,
            )

            M = len(obs_rb)
            if M == 0:
                # All groups filtered; skip this epoch
                timesteps_done += T_train
                continue

            # DAPO update
            update_metrics = dapo_update(
                self.policy, self.optimizer,
                obs_rb, act_rb, logp_rb, adv_rb,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                eps_low=self.eps_low,
                eps_high=self.eps_high,
                ent_coef=self.ent_coef,
                max_grad_norm=self.max_grad_norm,
            )

            timesteps_done += T_train
            epoch += 1

            # Periodic evaluation
            if timesteps_done >= eval_freq * epoch or timesteps_done >= self.total_timesteps:
                val_sharpe = _eval_sharpe(
                    self.policy, val_obs, val_ret,
                    self.n_tickers, self.max_weight,
                )

                if val_sharpe > self._best_val_sharpe:
                    self._best_val_sharpe = val_sharpe
                    self._best_policy_state = copy.deepcopy(
                        self.policy.state_dict()
                    )
                    best_path = str(Path(out_dir) / "best_policy.pt")
                    self.save(best_path)

                log_entry = {
                    "epoch":        epoch,
                    "timesteps":    timesteps_done,
                    "n_samples":    M,
                    "val_sharpe":   round(val_sharpe, 4),
                    "best_sharpe":  round(self._best_val_sharpe, 4),
                    **{k: round(v, 6) for k, v in update_metrics.items()},
                }
                train_log.append(log_entry)
                print(
                    f"[DAPO] epoch={epoch}  ts={timesteps_done:>7,}  "
                    f"val_sharpe={val_sharpe:.4f}  best={self._best_val_sharpe:.4f}  "
                    f"kl={update_metrics['approx_kl']:.4f}"
                )

        # Load best checkpoint
        if self._best_policy_state is not None:
            self.policy.load_state_dict(self._best_policy_state)

        # Save training log
        log_path = str(Path(out_dir) / "dapo_train_log.json")
        with open(log_path, "w") as f:
            json.dump(train_log, f, indent=2)
        print(f"[DAPO] Training complete.  Best val Sharpe = {self._best_val_sharpe:.4f}")
        print(f"[DAPO] Train log -> {log_path}")

        return train_log

    # -----------------------------------------------------------------------
    # Inference interface (compatible with SB3 PPO .predict())
    # -----------------------------------------------------------------------

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        """
        Predict action for a single observation.

        Parameters
        ----------
        obs : np.ndarray, shape (obs_dim,)
        deterministic : bool
            If True, return the policy mean (μ).
            If False, sample from N(μ, σ).

        Returns
        -------
        action : np.ndarray, shape (action_dim,)
        state  : None  (SB3 API compatibility)
        """
        self.policy.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32)
            if deterministic:
                mu, _ = self.policy(obs_t)
                action = mu.numpy()
            else:
                actions, _ = self.policy.sample(obs_t, n=1)
                action = actions[0].numpy()
        return action, None

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the policy state dict and metadata to ``path``.

        Format: PyTorch checkpoint dict with keys:
          - policy_state_dict
          - obs_dim, action_dim, seed
          - best_val_sharpe
          - dapo_hparams (for reproducibility)
        """
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "obs_dim":           self.obs_dim,
            "action_dim":        self.action_dim,
            "seed":              self.seed,
            "best_val_sharpe":   self._best_val_sharpe,
            "dapo_hparams": {
                "K":             self.K,
                "eps_low":       self.eps_low,
                "eps_high":      self.eps_high,
                "ent_coef":      self.ent_coef,
                "n_epochs":      self.n_epochs,
                "batch_size":    self.batch_size,
                "max_grad_norm": self.max_grad_norm,
                "learning_rate": self.learning_rate,
                "max_weight":    self.max_weight,
                "n_tickers":     self.n_tickers,
            },
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, config: Optional[dict] = None) -> "DAPOTrainer":
        """
        Load a saved DAPOTrainer from ``path``.

        Parameters
        ----------
        path : str
        config : dict, optional
            If provided, used as fall-back for policy architecture settings.

        Returns
        -------
        DAPOTrainer
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        obs_dim    = ckpt["obs_dim"]
        action_dim = ckpt["action_dim"]
        seed       = ckpt.get("seed", 42)
        hparams    = ckpt.get("dapo_hparams", {})

        # Build a minimal config from the saved hparams
        base_config: dict = config if config is not None else {}
        trainer = cls(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=base_config,
            seed=seed,
        )

        # Override hparams from checkpoint
        trainer.K             = int(hparams.get("K",             trainer.K))
        trainer.eps_low       = float(hparams.get("eps_low",     trainer.eps_low))
        trainer.eps_high      = float(hparams.get("eps_high",    trainer.eps_high))
        trainer.max_weight    = float(hparams.get("max_weight",  trainer.max_weight))
        trainer.n_tickers     = int(hparams.get("n_tickers",     trainer.n_tickers))
        trainer._best_val_sharpe = float(ckpt.get("best_val_sharpe", float("-inf")))

        trainer.policy.load_state_dict(ckpt["policy_state_dict"])
        trainer.policy.eval()

        return trainer
