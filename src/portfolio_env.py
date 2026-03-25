"""
portfolio_env.py — Custom Gymnasium environment for PPO portfolio allocation.

Observation space : Box(shape=(74,))   — 5×14 per-ticker + 4 shared regime
Action space      : Box(low=0, high=1, shape=(6,)) — raw logits for softmax
                    → projected to probability simplex (5 tickers + 1 cash)

Reward (strictly from approved design — no additional shaping):
    r_portfolio(t)   = Σ_i  w_i(t) × simple_return_i(t+1)
    σ_rolling_21d(t) = rolling std of the last 21 portfolio returns
    turnover(t)      = Σ_i |w_i(t) − w_i(t−1)|
    reward(t)        = r_portfolio(t) / (σ_rolling_21d(t) + ε) − λ × turnover(t)

Episode structure:
    - Starts at index 0 (first date in the provided obs/return arrays).
    - Initial weights: equal allocation (1/n_tickers for each ticker, 0 cash).
    - Terminates when all time steps have been consumed.
    - No early termination on drawdown.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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
    Single-asset-universe Gymnasium environment for PPO portfolio allocation.

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
        self.n_tickers: int = env_cfg["n_tickers"]
        self.action_dim: int = env_cfg["action_dim"]       # 6 (5 tickers + cash)
        self.obs_dim:    int = env_cfg["obs_dim"]           # 74
        self.lambda_tc:  float = float(env_cfg["transaction_cost"])
        self.vol_eps:    float = float(env_cfg["vol_eps"])
        self.vol_window: int   = int(env_cfg["vol_window"])

        self._n_steps = obs_array.shape[0]

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        # Episode state (initialised in reset())
        self._t: int = 0
        self._prev_weights = np.zeros(self.action_dim, dtype=np.float32)
        self._return_buffer = np.zeros(self.vol_window, dtype=np.float64)

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

        return self.obs_array[0].copy(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step.

        Parameters
        ----------
        action : np.ndarray, shape (6,)
            Raw agent output in [0, 1]^6.  Projected to the probability
            simplex via softmax before use.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        # 1. Project action → valid portfolio weights
        weights = self._softmax(action)  # shape (6,), sums to 1.0

        # 2. Portfolio simple return (ticker weights × ticker forward returns)
        ticker_weights  = weights[: self.n_tickers]          # first 5 dimensions
        ticker_returns  = self.forward_return_array[self._t]  # shape (5,)
        port_return     = float(np.dot(ticker_weights, ticker_returns))

        # 3. Update rolling portfolio return buffer (FIFO)
        self._return_buffer = np.roll(self._return_buffer, -1)
        self._return_buffer[-1] = port_return

        # 4. Rolling portfolio volatility
        roll_vol = float(np.std(self._return_buffer))

        # 5. Transaction cost (L1 weight turnover)
        turnover = float(np.sum(np.abs(weights - self._prev_weights)))

        # 6. Reward — strictly Sharpe-normalised return minus turnover cost
        reward = port_return / (roll_vol + self.vol_eps) - self.lambda_tc * turnover

        # 7. Advance state
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
            "portfolio_return": port_return,
            "rolling_vol":      roll_vol,
            "turnover":         turnover,
            "weights":          weights.copy(),
            "step":             self._t - 1,
        }

        return next_obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax — projects any R^n vector to the simplex."""
        x = np.clip(x, -20.0, 20.0)     # guard against extreme values
        e = np.exp(x - x.max())
        return (e / e.sum()).astype(np.float32)
