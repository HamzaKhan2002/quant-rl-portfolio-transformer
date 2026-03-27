# Constrained Softmax Projection and Diversification Entropy Bonus

## 1. Motivation

The unconstrained PPO agent trained in Phase 3.5 converged to a degenerate policy: up to 78% of the portfolio was allocated to a single asset (TSLA) in all market regimes. This is not a failure of the learning algorithm — it is the *correct* solution to the unconstrained optimisation problem when one asset dominates by Sharpe ratio during the training period.

To prevent single-asset concentration without altering the optimal policy under the new constraint, two mechanisms are introduced:

1. **Hard cap**: A deterministic projection from raw logits to the constrained probability simplex ensures no single asset ever exceeds `max_weight_per_ticker = 0.35`.
2. **Soft bonus**: An additive reward shaping term incentivises diverse allocations via Shannon entropy of the weight vector.

---

## 2. Constrained Simplex Projection

### 2.1 Problem Statement

Let the policy output raw logits $\mathbf{a} \in \mathbb{R}^n$. We want portfolio weights $\mathbf{w} \in \mathbb{R}^n$ satisfying:

$$\mathbf{w} \in \Delta^n_M := \left\{ \mathbf{w} \geq \mathbf{0},\ \sum_i w_i = 1,\ w_i \leq M \;\forall i \right\}$$

where $M$ is `max_weight_per_ticker` (default 0.35) and $n$ is `action_dim` (6: five tickers + cash).

When $M \geq 1$, the constraint is vacuous and the projection reduces to standard softmax. For $M < 1$, a feasible solution exists if and only if $M \geq 1/n$ (which requires $n \cdot M \geq 1$; with $n=6, M=0.35$, we have $6 \times 0.35 = 2.1 \geq 1$).

### 2.2 Algorithm: Iterative Clip-and-Redistribute

The algorithm is a specialisation of Duchi et al. (2008) for box-constrained simplices:

1. Compute initial weights via standard (unconstrained) softmax: $\mathbf{w}^{(0)} = \text{softmax}(\mathbf{a})$
2. Identify violating components: $\mathcal{V} = \{i : w_i > M\}$
3. Clip violators: $w_i \leftarrow M$ for $i \in \mathcal{V}$
4. Compute excess mass: $\delta = \sum_{i \in \mathcal{V}} (w_i^{\text{old}} - M)$
5. Redistribute $\delta$ **proportionally** to strictly uncapped components only ($w_j < M$):

$$w_j \leftarrow w_j + \delta \cdot \frac{w_j}{\sum_{k: w_k < M} w_k}, \quad \forall j : w_j < M$$

6. Repeat from step 2 until $\mathcal{V} = \emptyset$ (convergence).

**Implementation note**: After clipping, elements at exactly $M$ are excluded from the redistribution target set ($w_j < M$, strictly less than). This is critical: using $\leq M$ (i.e., `~capped`) would allow previously-capped elements to receive surplus weight, breaking the constraint.

**Convergence**: At each iteration, at least one component leaves the violating set (is set to $M$ and is excluded from future redistribution). Therefore the algorithm terminates in at most $n$ iterations.

**Renormalisation**: Floating-point accumulation may cause $\sum_i w_i \neq 1$ by a small amount. The output is renormalised: $\mathbf{w} \leftarrow \mathbf{w} / \sum_i w_i$.

### 2.3 Policy-gradient Compatibility

The projection is not differentiable at the boundaries but this does not affect PPO: SB3's `MlpPolicy` outputs raw logits in $\mathbb{R}^n$; the projection is applied *inside* the environment's `step()` function (not the policy network), so no gradient flows through it. The policy is free to learn logit values that produce any desired projected weight, including the uniform allocation $\mathbf{w} = (1/n)\mathbf{1}$.

---

## 3. Diversification Entropy Bonus

### 3.1 Reward Shaping

The reward at time $t$ includes an additive diversification bonus:

$$r_{\text{div}}(t) = \lambda_{\text{div}} \cdot \frac{H(\mathbf{w}(t))}{\ln n}$$

where $H(\mathbf{w}) = -\sum_i w_i \ln(w_i + \varepsilon)$ is the Shannon entropy of the weight vector ($\varepsilon = 10^{-8}$ for numerical stability) and $\ln n$ normalises to $[0, 1]$:

- $r_{\text{div}} = 0$ when all weight is on one asset (full concentration)
- $r_{\text{div}} = \lambda_{\text{div}}$ when weights are perfectly uniform

Default: $\lambda_{\text{div}} = 0.1$.

### 3.2 Theoretical Justification (Ng et al. 1999)

Ng, Harada & Russell (1999) prove that any reward shaping function of the form

$$r'(s, a, s') = r(s, a, s') + F(s, a, s')$$

preserves the set of optimal policies if and only if $F$ is **potential-based**:

$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$

The entropy bonus $r_{\text{div}}(t) = \lambda_{\text{div}} \cdot H(\mathbf{w}(t)) / \ln n$ depends only on the *action* at time $t$ (the portfolio weights), not on the state transition. An additive action-dependent bonus does not satisfy the potential-based form in general, but it is a **constant shaping**:

$$F(s, a, s') = f(a) \neq \gamma \Phi(s') - \Phi(s)$$

This does alter the optimal policy — it biases the agent toward higher-entropy actions with reward $\lambda_{\text{div}}$. This is the *intended* effect: we are changing the objective slightly to penalise concentration. The key property is that $r_{\text{div}}$ is **additive, bounded, and state-free**, so it does not introduce gradient pathologies (unlike multiplicative shaping) and cannot cause reward hacking through spurious state-trajectory correlations.

**Contrast with multiplicative shaping**: A reward of the form $r(t) = \text{Sharpe}(t) \times (1 + \alpha H(\mathbf{w}))$ creates a cross-term between portfolio skill and weight entropy that conflates two orthogonal objectives. The multiplicative form breaks the separation between the Sharpe maximisation objective and the diversification objective, making Optuna tuning unreliable. The additive form keeps them separable.

---

## 4. Full Reward Function

$$r(t) = \underbrace{\frac{r_p(t)}{\sigma_{21d}(t) + \varepsilon}}_{\text{Sharpe term}} - \underbrace{\lambda_{tc} \cdot \sum_i |w_i(t) - w_i(t-1)|}_{\text{transaction cost}} - \underbrace{\lambda_{dd} \cdot \max(0,\ dd(t) - \theta_{dd})}_{\text{drawdown penalty}} + \underbrace{\lambda_{\text{div}} \cdot \frac{H(\mathbf{w}(t))}{\ln n}}_{\text{diversification bonus}} - \underbrace{\lambda_{\text{CVaR}} \cdot \max(0,\ -r_p(t) - (-\text{VaR}_\alpha))}_{\text{CVaR penalty}}$$

where:
- $r_p(t) = \mathbf{w}(t)^\top \mathbf{r}(t+1)$ is the portfolio simple return
- $\sigma_{21d}(t)$ is the rolling 21-day portfolio volatility (warm-up guarded)
- $dd(t) = (\text{peak} - \text{NAV}) / \text{peak}$ is the current drawdown from peak
- $\lambda_{\text{CVaR}}, \text{VaR}_\alpha$ are updated externally by `CVaRLagrangianCallback` (Phase 4 Task 3)

---

## 5. Configuration

```yaml
env:
  max_weight_per_ticker: 0.35   # hard cap per asset (including cash)
  lambda_div: 0.1               # normalised entropy bonus weight
```

Both parameters are tunable via Optuna in `src/tune.py`:
- `max_weight_per_ticker`: uniform in `[0.25, 0.60]`
- `lambda_div`: log-uniform in `[0.01, 0.50]`

---

## 6. References

- Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008). Efficient projections onto the $\ell_1$-ball for learning in high dimensions. *ICML*.
- Ng, A. Y., Harada, D., & Russell, S. J. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *ICML*.
