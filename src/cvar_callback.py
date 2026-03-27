"""
cvar_callback.py — Conditional Value-at-Risk (CVaR) Lagrangian callback for SB3.

The callback implements an online Lagrangian method that constrains the
expected tail loss of the portfolio:

    CVaR_α(π) ≤ τ

where α = 0.05 (5th-percentile tail), τ = 0.02 (2% max daily CVaR loss).

At each update step the callback:
  1. Collects raw portfolio returns from ``info["portfolio_return"]``
  2. Maintains a rolling 252-day return buffer
  3. Estimates CVaR_α from the buffer (empirical alpha-quantile + tail mean)
  4. Updates the Lagrangian multiplier via dual ascent:
       λ ← max(0, λ + η * (CVaR_loss - τ))
  5. Pushes the updated (λ, VaR_α) to every sub-environment via env_method

The step-level penalty in the environment reward is:
    cvar_penalty(t) = λ * max(0, −r(t) − (−VaR_α))

which penalises returns that land in the tail (below VaR_α).  When λ = 0
(the default at training start) the reward is unchanged.

Usage (in train_fold_seed or train.py)::

    from src.cvar_callback import CVaRLagrangianCallback
    cvar_cb = CVaRLagrangianCallback(config)
    callbacks = make_callbacks(val_env, config, model_dir) + [cvar_cb]
    model.learn(..., callback=callbacks)
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError as exc:
    raise ImportError(
        "stable-baselines3 is required.  Install with: pip install stable-baselines3"
    ) from exc


class CVaRLagrangianCallback(BaseCallback):
    """
    Online Lagrangian CVaR constraint for PPO portfolio training.

    Parameters
    ----------
    config : dict
        Loaded config.yaml.  Reads from ``config["env"]``:

        - ``cvar_alpha``       : float, default 0.05 — tail quantile (5th pct)
        - ``cvar_threshold``   : float, default 0.02 — max tolerated CVaR loss
        - ``cvar_lr``          : float, default 0.001 — dual ascent step size
        - ``lambda_cvar_init`` : float, default 0.0  — initial multiplier
        - ``var_alpha_init``   : float, default -100.0 — initial VaR (never fires)

    cvar_buffer_size : int
        Rolling window length for empirical CVaR estimation.
        Default 252 (≈ 1 trading year).
    update_freq : int
        Number of environment steps between dual-ascent updates.
        Default 252 (≈ 1 trading year, i.e., once per episode).
    verbose : int
        SB3 verbosity (0 = silent, 1 = log each update).
    """

    def __init__(
        self,
        config: dict,
        cvar_buffer_size: int = 252,
        update_freq: int = 252,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)

        env_cfg = config.get("env", {})
        self.cvar_alpha:       float = float(env_cfg.get("cvar_alpha",       0.05))
        self.cvar_threshold:   float = float(env_cfg.get("cvar_threshold",   0.02))
        self.cvar_lr:          float = float(env_cfg.get("cvar_lr",          0.001))
        self.lambda_cvar:      float = float(env_cfg.get("lambda_cvar_init", 0.0))
        self.var_alpha:        float = float(env_cfg.get("var_alpha_init",   -100.0))

        self.cvar_buffer_size: int = cvar_buffer_size
        self.update_freq:      int = update_freq

        # Rolling circular buffer of raw portfolio returns
        self._buffer: deque = deque(maxlen=cvar_buffer_size)
        self._steps_since_update: int = 0

    # -----------------------------------------------------------------------
    # BaseCallback interface
    # -----------------------------------------------------------------------

    def _on_step(self) -> bool:
        """
        Called after every environment step.

        Extracts ``portfolio_return`` from info dicts, fills the buffer, and
        triggers a dual-ascent update every ``update_freq`` steps.
        """
        # Extract raw portfolio returns from all parallel envs
        infos: List[dict] = self.locals.get("infos", [])
        for info in infos:
            if "portfolio_return" in info:
                self._buffer.append(float(info["portfolio_return"]))

        self._steps_since_update += len(infos)

        if (self._steps_since_update >= self.update_freq
                and len(self._buffer) >= max(10, int(self.cvar_buffer_size * 0.1))):
            self._dual_ascent_update()
            self._steps_since_update = 0

        return True  # continue training

    def _on_training_end(self) -> None:
        """Final dual-ascent step at training completion."""
        if len(self._buffer) >= max(10, int(self.cvar_buffer_size * 0.1)):
            self._dual_ascent_update()

    # -----------------------------------------------------------------------
    # CVaR estimation and dual ascent
    # -----------------------------------------------------------------------

    def _dual_ascent_update(self) -> None:
        """
        Compute empirical CVaR from the rolling buffer and perform one step
        of projected gradient dual ascent on the Lagrangian multiplier λ.

        CVaR_α (Conditional Value-at-Risk):
            VaR_α = α-quantile of the return distribution (negative = loss)
            CVaR_α = E[r | r ≤ VaR_α]  (expected return in the tail)
            CVaR_loss = max(0, -CVaR_α)  (tail loss as a positive number)

        Dual ascent:
            λ ← max(0, λ + η × (CVaR_loss − τ))

        When CVaR_loss ≤ τ, the constraint is satisfied and λ is reduced
        (or stays at 0).  When CVaR_loss > τ, λ increases, strengthening
        the tail penalty in the environment reward.
        """
        returns = np.array(self._buffer, dtype=np.float64)

        # Empirical VaR_α (α-percentile of the return distribution)
        var_alpha = float(np.percentile(returns, self.cvar_alpha * 100))

        # Empirical CVaR_α: mean of returns at or below VaR_α
        tail = returns[returns <= var_alpha]
        if len(tail) == 0:
            cvar_alpha_val = var_alpha
        else:
            cvar_alpha_val = float(tail.mean())

        cvar_loss = max(0.0, -cvar_alpha_val)  # loss = positive tail loss

        # Dual ascent: maximise the Lagrangian dual w.r.t. λ (projected to ≥ 0)
        self.lambda_cvar = max(
            0.0,
            self.lambda_cvar + self.cvar_lr * (cvar_loss - self.cvar_threshold),
        )
        self.var_alpha = var_alpha

        # Push updated parameters to all environments
        self._push_cvar_params()

        if self.verbose >= 1:
            print(
                f"[CVaRLagrangianCallback] "
                f"VaR_α={var_alpha:.4f}  CVaR_loss={cvar_loss:.4f}  "
                f"τ={self.cvar_threshold:.4f}  λ={self.lambda_cvar:.4f}  "
                f"n_buf={len(self._buffer)}"
            )

    def _push_cvar_params(self) -> None:
        """
        Push (lambda_cvar, var_alpha) to every sub-environment.

        Uses ``env_method`` to call ``PortfolioEnv.set_cvar_params`` on each
        sub-environment, which is compatible with both ``DummyVecEnv`` and
        ``SubprocVecEnv``.
        """
        try:
            self.training_env.env_method(
                "set_cvar_params",
                self.lambda_cvar,
                self.var_alpha,
            )
        except Exception as e:
            # Graceful degradation: if env_method fails (e.g., VecNormalize
            # wrapping without underlying PortfolioEnv), log and continue.
            if self.verbose >= 1:
                print(f"[CVaRLagrangianCallback] Warning: env_method failed — {e}")

    # -----------------------------------------------------------------------
    # Diagnostic helpers
    # -----------------------------------------------------------------------

    def get_cvar_params(self) -> dict:
        """Return current CVaR parameters as a dict."""
        return {
            "lambda_cvar": self.lambda_cvar,
            "var_alpha":   self.var_alpha,
            "buffer_size": len(self._buffer),
        }

    @staticmethod
    def compute_cvar(returns: np.ndarray, alpha: float = 0.05) -> tuple:
        """
        Compute empirical VaR_α and CVaR_α from a return array.

        Parameters
        ----------
        returns : np.ndarray
            Array of daily portfolio returns.
        alpha : float
            Tail quantile (default 0.05 → 5th percentile).

        Returns
        -------
        (var_alpha, cvar_alpha) : tuple of float
            Both values are negative for losses.
        """
        if len(returns) == 0:
            return 0.0, 0.0
        var_alpha = float(np.percentile(returns, alpha * 100))
        tail = returns[returns <= var_alpha]
        cvar_alpha = float(tail.mean()) if len(tail) > 0 else var_alpha
        return var_alpha, cvar_alpha
