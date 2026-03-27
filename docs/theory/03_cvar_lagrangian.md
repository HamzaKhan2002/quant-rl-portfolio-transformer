# CVaR Lagrangian Constraint

## 1. Problem Statement

The reward-shaping terms introduced in Phase 4 (diversification bonus, drawdown penalty) incentivise the agent toward better average-case behaviour, but they do not provide a formal guarantee on tail risk. The agent could still accept occasional large drawdowns if the expected Sharpe improvement outweighs the penalty.

A Conditional Value-at-Risk (CVaR) constraint directly bounds the expected loss in the worst-α fraction of outcomes:

$$\text{CVaR}_\alpha(\pi) \leq \tau$$

where:
- $\alpha = 0.05$: we constrain the expected loss in the worst 5% of trading days
- $\tau = 0.02$: maximum tolerated conditional tail loss (2% per day)

---

## 2. CVaR Definition

For a portfolio return distribution $r \sim \pi$:

**Value-at-Risk** (VaR) at level $\alpha$:
$$\text{VaR}_\alpha = \inf\{x : P(r \leq x) \geq \alpha\}$$
In practice: the $\alpha$-quantile of the empirical return distribution. Negative for losses.

**Conditional Value-at-Risk** (CVaR, also called Expected Shortfall):
$$\text{CVaR}_\alpha = \mathbb{E}[r \mid r \leq \text{VaR}_\alpha]$$
The mean return conditional on being in the tail. Also negative for losses.

**CVaR loss** (the constraint target):
$$\text{CVaR\_loss} = \max(0, -\text{CVaR}_\alpha)$$
Positive number representing the average tail loss magnitude.

---

## 3. Lagrangian Relaxation

The constrained optimisation problem:

$$\max_\pi \mathbb{E}[r(\pi)] \quad \text{s.t.} \quad \text{CVaR\_loss}(\pi) \leq \tau$$

is relaxed to an unconstrained Lagrangian:

$$\mathcal{L}(\pi, \lambda) = \mathbb{E}[r(\pi)] - \lambda \cdot (\text{CVaR\_loss}(\pi) - \tau)$$

The Lagrangian multiplier $\lambda \geq 0$ adaptively penalises constraint violations. The agent optimises $\pi$ for the augmented reward while the callback optimises $\lambda$ to enforce the constraint.

---

## 4. Dual Ascent Update

The callback performs online projected gradient ascent on the dual variable $\lambda$:

$$\lambda \leftarrow \max\left(0,\ \lambda + \eta \cdot (\text{CVaR\_loss} - \tau)\right)$$

where $\eta = 0.001$ is the step size (`cvar_lr`). Projection onto $\mathbb{R}_{\geq 0}$ ensures $\lambda$ remains non-negative (a zero $\lambda$ means no constraint is applied).

**Convergence intuition:**
- If $\text{CVaR\_loss} > \tau$ (constraint violated): $\lambda$ increases → stronger tail penalty → agent reduces tail losses
- If $\text{CVaR\_loss} \leq \tau$ (constraint satisfied): $\lambda$ decreases → relaxed penalty → agent has more freedom
- At optimality: $\lambda^* > 0$ iff the constraint is tight; $\lambda^* = 0$ iff it is slack

**Step-size choice**: $\eta = 0.001$ is intentionally small. CVaR estimation from a 252-day buffer is noisy; aggressive dual ascent would cause $\lambda$ oscillations that destabilise PPO training. The slow ascent allows PPO to adapt its policy between $\lambda$ updates.

---

## 5. Environment-Level Penalty

The step-level CVaR penalty injected into the environment reward:

$$r_{\text{CVaR}}(t) = -\lambda \cdot \max\left(0,\ -r_p(t) - (-\text{VaR}_\alpha)\right)$$

This penalises any single-step return that falls in the tail (below $\text{VaR}_\alpha$). The penalty amount is proportional to how far below the tail threshold the return falls.

**Why step-level?** Rather than penalising at episode end (which would create sparse rewards), the step-level penalty provides an immediate gradient signal for every tail event. This is analogous to the instantaneous Lagrangian form in continuous-time stochastic control.

**Initialisation**: $\lambda = 0$, $\text{VaR}_\alpha = -100$ (effectively never triggers). Training begins unconstrained, allowing the agent to first learn a reasonable portfolio strategy before the constraint is tightened.

---

## 6. Rolling Buffer

CVaR is estimated from a rolling 252-day (1 trading year) buffer of raw portfolio returns. The buffer is updated every environment step; the dual ascent update fires every 252 steps (≈ 1 episode).

**Empirical CVaR estimation**:
1. Sort the buffer returns
2. $\hat{\text{VaR}}_\alpha = \text{Percentile}(\{r_t\}, \alpha \times 100)$
3. $\hat{\text{CVaR}}_\alpha = \text{mean}(\{r_t : r_t \leq \hat{\text{VaR}}_\alpha\})$

A minimum buffer of 10% of `cvar_buffer_size` (≥ 25 returns) is required before the first update, preventing noisy early-training estimates.

---

## 7. Integration with SB3

`CVaRLagrangianCallback` inherits from `stable_baselines3.common.callbacks.BaseCallback`:

```python
from src.cvar_callback import CVaRLagrangianCallback
cvar_cb = CVaRLagrangianCallback(config)
callbacks = make_callbacks(val_env, config, model_dir) + [cvar_cb]
model.learn(total_timesteps=500_000, callback=callbacks)
```

The callback reads `info["portfolio_return"]` from `self.locals["infos"]` — the standard SB3 info dict. It pushes $(\lambda, \text{VaR}_\alpha)$ to all sub-environments via `env_method("set_cvar_params", ...)`, which is compatible with both `DummyVecEnv` and `SubprocVecEnv`.

---

## 8. Configuration

```yaml
env:
  cvar_alpha:       0.05    # tail quantile (5th percentile)
  cvar_threshold:   0.02    # max tolerated daily CVaR loss (2%)
  cvar_lr:          0.001   # dual ascent step size
  lambda_cvar_init: 0.0     # initial Lagrangian multiplier
  var_alpha_init:  -100.0   # initial VaR (never triggers at start)
```

---

## 9. Relationship to Other Reward Terms

| Term | Type | Effect |
|------|------|--------|
| Sharpe | Objective | Maximise risk-adjusted return |
| Transaction cost | Hard penalty | Discourage excessive turnover |
| Drawdown penalty | Soft penalty | Episode-level drawdown control |
| Diversification bonus | Soft incentive | Increase weight entropy |
| CVaR Lagrangian | Adaptive constraint | Bound tail risk, formally |

CVaR is the only term with a formal constraint semantics. All others are soft preferences. Theoretically, only CVaR guarantees the constraint is satisfied at convergence (under stationarity).

---

## 10. References

- Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2(3), 21–41.
- Chow, Y., Ghavamzadeh, M., Janson, L., & Pavone, M. (2017). Risk-constrained reinforcement learning with percentile risk criteria. *JMLR*.
- Borkar, V. S. (2005). An actor-critic algorithm for constrained Markov decision processes. *Systems & Control Letters*.
