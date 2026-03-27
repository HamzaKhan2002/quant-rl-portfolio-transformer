"""
portfolio_env.py — Custom Gymnasium environment for PPO/DAPO portfolio allocation.

Observation space : Box(shape=(obs_dim,))  — per-ticker features + shared regime
Action space      : Box(low=-20, high=20, shape=(action_dim,)) — raw logits
                    → projected to probability simplex via constrained softmax
                    → each weight ≤ max_weight_per_ticker (default 0.35)

Reward (per step t):
    r_portfolio(t)   = Σ_i  w_i(t) × simple_return_i(t+1)
    σ_rolling_21d(t) = rolling std of the last 21 portfolio returns
    turnover(t)      = Σ_i |w_i(t) − w_i(t−1)|
    drawdown(t)      = (peak_value − current_value) / peak_value
    entropy_w(t)     = −Σ_i w_i log(w_i + ε)   [Shannon weight entropy]
    cvar_pen(t)      = λ_cvar × max(0, −r(t) − VaR_α)  [tail loss penalty]

    reward(t) = r_portfolio(t) / (σ_rolling_21d(t) + ε)     [Sharpe term]
              − λ_tc  × turnover(t)                          [cost term]
              − λ_dd  × max(0, drawdown(t) − dd_threshold)   [drawdown term]
              + λ_div × entropy_w(t) / ln(action_dim)        [diversification bonus]
              − cvar_pen(t)                                  [CVaR Lagrangian]

    During vol buffer warm-up (buffer_count < vol_window), the Sharpe term
    degenerates to r_portfolio(t) to avoid reward spikes from near-zero vol.

Episode structure:
    - Starts at index 0 in the provided obs/return arrays.
    - Initial weights: equal allocation (1/n_tickers each, 0 cash).
    - Terminates when all time steps are consumed.
    - No early termination on drawdown.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import math

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "gymnasium is required.  Install with: pip install gymnasium"
    ) from exc


class PortfolioEnv(gym.Env):
    """
    Single-universe Gymnasium environment for PPO/DAPO portfolio allocation.

    Parameters
    ----------
    obs_array : np.ndarray, shape (T, obs_dim)
        Normalised observation matrix — one row per trading day.
    forward_return_array : np.ndarray, shape (T, n_tickers)
        Per-ticker simple forward returns.  ``forward_return_array[t, i]`` is
        the return of ticker i from day t to day t+1.
    config : dict
        Loaded config.yaml.  Reads keys from ``config["env"]``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        obs_array: np.ndarray,
        forward_return_array: np.ndarray,
        config: dict,
    ) -> None:
        super().__init__()

        assert obs_array.shape[0] == forward_return_array.shape[0], (
            "obs_array and forward_return_array must have the same number of rows "
            f"(got {obs_array.shape[0]} vs {forward_return_array.shape[0]})"
        )

        self.obs_array = obs_array.astype(np.float32)
        self.forward_return_array = forward_return_array.astype(np.float32)

        env_cfg = config["env"]
        self.n_tickers: int  = env_cfg["n_tickers"]
        self.action_dim: int = env_cfg["action_dim"]       # 6 (5 tickers + cash)
        self.obs_dim:    int = env_cfg["obs_dim"]           # 74 (or varies for ablations)
        self.lambda_tc:  float = float(env_cfg["transaction_cost"])
        self.vol_eps:    float = float(env_cfg["vol_eps"])
        self.vol_window: int   = int(env_cfg["vol_window"])

        # Drawdown penalty
        self.lambda_dd:    float = float(env_cfg.get("lambda_dd", 0.0))
        self.dd_threshold: float = float(env_cfg.get("dd_threshold", 0.05))

        # Concentration constraint: each weight ≤ max_weight_per_ticker
        # Default 1.0 = unconstrained (backward compatible)
        self.max_weight: float = float(env_cfg.get("max_weight_per_ticker", 1.0))

        # Diversification entropy bonus (Ng et al. compliant: additive shaping)
        self.lambda_div: float = float(env_cfg.get("lambda_div", 0.0))
        # log(action_dim) is the maximum entropy (uniform distribution)
        self._log_n: float = math.log(self.action_dim)

        # CVaR Lagrangian penalty (updated externally by CVaRLagrangianCallback)
        # lambda_cvar: Lagrangian multiplier (dual variable, non-negative)
        # var_alpha:   rolling VaR estimate at alpha level (negative = loss)
        self.lambda_cvar: float = float(env_cfg.get("lambda_cvar_init", 0.0))
        self.var_alpha:   float = float(env_cfg.get("var_alpha_init", -100.0))

        # Last row of forward_return_array is 0-filled (shift(-1).fillna(0));
        # stop one step early so every step has a real forward return.
        self._n_steps = obs_array.shape[0] - 1

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-20.0,
            high=20.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        # Episode state (initialised in reset())
        self._t: int = 0
        self._prev_weights = np.zeros(self.action_dim, dtype=np.float32)
        self._return_buffer = np.zeros(self.vol_window, dtype=np.float64)
        self._buffer_count: int = 0
        self._portfolio_value: float = 1.0   # running NAV for drawdown tracking
        self._peak_value:      float = 1.0   # running peak NAV

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self._t = 0
        # Equal-weight start: 1/n_tickers per ticker, 0 cash
        self._prev_weights = np.zeros(self.action_dim, dtype=np.float32)
        self._prev_weights[: self.n_tickers] = 1.0 / self.n_tickers
        self._return_buffer = np.zeros(self.vol_window, dtype=np.float64)
        self._buffer_count = 0
        self._portfolio_value = 1.0
        self._peak_value = 1.0

        return self.obs_array[0].copy(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step.

        Parameters
        ----------
        action : np.ndarray, shape (action_dim,)
            Raw agent output (logits in ℝ^n).  Projected to the probability
            simplex via constrained softmax before use.
        """
        # 1. Project action → valid portfolio weights (constrained simplex)
        weights = self._constrained_softmax(action, self.max_weight)

        # 2. Portfolio simple return
        ticker_weights = weights[: self.n_tickers]
        ticker_returns = self.forward_return_array[self._t]
        port_return    = float(np.dot(ticker_weights, ticker_returns))

        # 3. Update rolling return buffer (FIFO)
        self._return_buffer = np.roll(self._return_buffer, -1)
        self._return_buffer[-1] = port_return
        self._buffer_count = min(self._buffer_count + 1, self.vol_window)

        # 4. Rolling portfolio volatility (over filled portion)
        if self._buffer_count < 2:
            roll_vol = 0.0
        else:
            roll_vol = float(np.std(self._return_buffer[-self._buffer_count:]))

        # 5. Transaction cost (L1 weight turnover)
        turnover = float(np.sum(np.abs(weights - self._prev_weights)))

        # 6. Update NAV + drawdown
        self._portfolio_value *= (1.0 + port_return)
        self._peak_value = max(self._peak_value, self._portfolio_value)
        current_drawdown = (
            (self._peak_value - self._portfolio_value) / self._peak_value
        )

        # 7. Diversification entropy bonus (Ng et al.: additive, no optimal policy change)
        #    Normalised entropy ∈ [0, 1]: 0 = full concentration, 1 = uniform
        weight_entropy = float(
            -np.sum(weights * np.log(weights + 1e-8))
        )
        div_bonus = self.lambda_div * (weight_entropy / self._log_n)

        # 8. CVaR Lagrangian step-level penalty
        #    Penalises returns that land in the tail (below −VaR_α).
        #    var_alpha is updated externally by CVaRLagrangianCallback.
        #    When lambda_cvar = 0 (default), this term is zero.
        cvar_penalty = self.lambda_cvar * max(0.0, -port_return - (-self.var_alpha))

        # 9. Reward — warm-up aware Sharpe + all penalty/bonus terms
        dd_penalty = self.lambda_dd * max(0.0, current_drawdown - self.dd_threshold)
        if self._buffer_count < self.vol_window:
            sharpe_term = port_return
        else:
            sharpe_term = port_return / (roll_vol + self.vol_eps)

        reward = (
            sharpe_term
            - self.lambda_tc * turnover
            - dd_penalty
            + div_bonus
            - cvar_penalty
        )

        # 10. Advance state
        self._prev_weights = weights.copy()
        self._t += 1

        terminated = self._t >= self._n_steps
        truncated  = False

        next_obs = (
            self.obs_array[self._t].copy()
            if not terminated
            else np.zeros(self.obs_dim, dtype=np.float32)
        )

        info: Dict[str, Any] = {
            "portfolio_return":  port_return,
            "rolling_vol":       roll_vol,
            "turnover":          turnover,
            "weights":           weights.copy(),
            "step":              self._t - 1,
            "current_drawdown":  current_drawdown,
            "weight_entropy":    weight_entropy,
            "div_bonus":         div_bonus,
            "cvar_penalty":      cvar_penalty,
        }

        return next_obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # CVaR Lagrangian interface (called by CVaRLagrangianCallback)
    # ------------------------------------------------------------------

    def set_cvar_params(self, lambda_cvar: float, var_alpha: float) -> None:
        """
        Update CVaR Lagrangian parameters from an external callback.

        Parameters
        ----------
        lambda_cvar : float
            Current Lagrangian multiplier (≥ 0).  Larger values impose
            stronger constraints on tail losses.
        var_alpha : float
            Alpha-level VaR estimate from the rolling return buffer.
            Negative value represents a loss (e.g. −0.01 = 1% daily loss).
        """
        self.lambda_cvar = float(lambda_cvar)
        self.var_alpha   = float(var_alpha)

    # ------------------------------------------------------------------
    # Action projection
    # ------------------------------------------------------------------

    @staticmethod
    def _constrained_softmax(x: np.ndarray, max_weight: float = 1.0) -> np.ndarray:
        """
        Project logit vector onto the constrained probability simplex.

        When max_weight ≥ 1.0 this reduces to standard softmax.  For
        max_weight < 1.0 the output is the unique projection satisfying:
          - all weights ≥ 0
          - weights sum to 1.0
          - no single weight exceeds max_weight

        Algorithm: iterative clip-and-redistribute (converges in O(n) iterations).

        Parameters
        ----------
        x : np.ndarray, shape (n,)
            Raw logits.
        max_weight : float
            Maximum allowed weight per asset (0 < max_weight ≤ 1).

        Returns
        -------
        np.ndarray, shape (n,)   dtype float32, values in [0, max_weight], sum = 1.
        """
        x = np.clip(x, -20.0, 20.0)
        e = np.exp(x - x.max())
        weights = (e / e.sum()).astype(np.float64)

        if max_weight >= 1.0:
            return weights.astype(np.float32)

        # Iterative projection: clip above max_weight, redistribute excess
        # proportionally to strictly-uncapped assets.  Previously capped
        # elements (== max_weight) are excluded from redistribution to prevent
        # them from receiving additional weight and violating the constraint.
        # Converges in at most n iterations (each iteration caps ≥1 asset).
        for _ in range(len(weights) + 1):
            over = weights > max_weight
            if not over.any():
                break
            excess = (weights[over] - max_weight).sum()
            weights[over] = max_weight
            # Only redistribute to elements strictly below cap
            uncapped_mask = weights < max_weight
            uncapped_sum = weights[uncapped_mask].sum()
            if uncapped_sum > 1e-12:
                weights[uncapped_mask] += (
                    excess * weights[uncapped_mask] / uncapped_sum
                )

        # Numerical safety: renormalise to exactly 1.0
        total = weights.sum()
        if total > 1e-12:
            weights /= total

        return weights.astype(np.float32)

    # Keep legacy name for backward compatibility with ensemble code
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax (unconstrained)."""
        x = np.clip(x, -20.0, 20.0)
        e = np.exp(x - x.max())
        return (e / e.sum()).astype(np.float32)
