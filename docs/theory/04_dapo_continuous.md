# DAPO: Direct Advantage Policy Optimization for Continuous Portfolio Allocation

## 1. Motivation

PPO uses Generalised Advantage Estimation (GAE) to assign credit across a trajectory using a learned value function. This creates two problems for portfolio RL:

1. **Critic approximation error**: The value function is hard to learn for financial time series because returns are non-stationary and regime-dependent. A poorly calibrated critic degrades the advantage estimate, destabilising training.
2. **Single-action sample**: At each timestep, PPO observes only one action (the committed trajectory action). This gives a high-variance policy gradient estimate in action space.

DAPO (Direct Advantage Policy Optimization) eliminates the critic entirely, replacing GAE with **group-relative advantage estimation**: at each timestep, K actions are sampled from the current policy, and each action's advantage is computed relative to the group's mean and standard deviation.

For the continuous portfolio domain, this is particularly natural: since the portfolio return is a deterministic function of the weight vector and the realized asset returns, we can compute K rewards at each state without any environment interaction beyond the observed prices.

---

## 2. Algorithm

### 2.1 Group-Relative Advantage

At time $t$, with observation $o_t$ and realized asset returns $\mathbf{r}_t$:

1. Sample K actions from the current policy: $\{a_k\}_{k=1}^K \sim \pi(\cdot | o_t)$
2. Project each to portfolio weights: $\mathbf{w}_k = \text{constrained\_softmax}(a_k, M)$
3. Compute portfolio returns: $R_k = \mathbf{w}_k^{(1:n)} \cdot \mathbf{r}_t$
4. Group statistics: $\bar{R} = \frac{1}{K}\sum_k R_k$, $\hat{\sigma} = \text{std}(\{R_k\})$
5. Group-relative advantages: $A_k = \frac{R_k - \bar{R}}{\max(\hat{\sigma}, \sigma_{\min})}$

Properties:
- $\sum_k A_k = 0$ exactly (zero-mean by construction)
- $A_k > 0$ iff action $k$ outperforms the group mean
- $\sigma_{\min}$ prevents division by near-zero variance (degenerate groups)

### 2.2 Dynamic Group Filtering

Groups with $\hat{\sigma} < \sigma_{\min}$ ("no-signal groups") are discarded. These occur when all K weight vectors produce nearly identical returns — for example, when the cash allocation dominates and asset returns matter little. Including these groups would produce zero gradients with high computational cost.

### 2.3 Asymmetric Clipping

DAPO extends PPO's CLIP objective with asymmetric bounds:

$$\mathcal{L}_{\text{DAPO}} = -\mathbb{E}_{t,k}\left[\min\left(\rho_{t,k} A_k,\ \text{clip}\left(\rho_{t,k}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}\right) A_k\right)\right]$$

where $\rho_{t,k} = \frac{\pi_{\theta}(a_k | o_t)}{\pi_{\theta_{\text{old}}}(a_k | o_t)}$ is the probability ratio.

With $\varepsilon_{\text{low}} = 0.1 < \varepsilon_{\text{high}} = 0.2$:
- **Positive-advantage actions**: allowed up to 20% probability increase (more aggressive exploitation of good actions)
- **Negative-advantage actions**: prevented from more than 10% probability decrease (conservative when avoiding bad actions)

The asymmetry reflects that increasing the probability of a genuinely good action is less risky than decreasing the probability of a bad one (which might accidentally push the policy toward even worse alternatives).

### 2.4 Policy Architecture

The Gaussian policy outputs the mean and log-standard-deviation of a multivariate Gaussian distribution over action space (raw logits):

$$\pi_\theta(a | o) = \mathcal{N}(\mu_\theta(o), \text{diag}(\sigma_\theta(o)^2))$$

- **Shared trunk**: MLP with architecture [256, 256] (same as PPO)
- **Mean head**: Linear(256, action_dim), initialised with small weights
- **Log-std head**: Linear(256, action_dim), bias initialised to −1 (σ₀ ≈ 0.37)
- **Action**: $a \sim \mathcal{N}(\mu, \sigma)$, then $\mathbf{w} = \text{constrained\_softmax}(a, M)$

Log-std is clamped to [−5, 2] for numerical stability.

---

## 3. Comparison to Original DAPO (LLM Setting)

The DAPO paper (Shao et al. 2025) was designed for LLMs with discrete token actions and verifiable reward signals. The key adaptations for continuous portfolio allocation:

| Aspect | LLM DAPO | Portfolio DAPO |
|--------|----------|----------------|
| Action space | Discrete tokens | Continuous logits → constrained simplex |
| Reward | Binary (correct/incorrect) | Portfolio return (continuous) |
| K-sampling | K full response trajectories | K weight vectors at each state (no trajectory rollout) |
| Episode structure | Single question → answer | Sequential T-step trading episode |
| Critic | None (both) | None |
| Group relative adv | Yes (both) | Yes |
| Asymmetric clipping | Yes (both) | Yes |

The portfolio setting allows an even more efficient K-sampling: since $R_k = \mathbf{w}_k^{(1:n)} \cdot \mathbf{r}_t$ is a closed-form function, K rewards are computed from K weight vectors without any additional environment rollouts.

---

## 4. Full Training Objective

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{DAPO}}(\theta) - \lambda_{\text{ent}} \cdot H(\pi_\theta)$$

where $H(\pi_\theta) = \frac{1}{T \cdot K} \sum_{t,k} \text{entropy}(\pi_\theta(\cdot | o_t))$ is the mean policy entropy (encouraging exploration).

The CVaR Lagrangian from Phase 4 Task 3 is applied at the environment level during rollout collection, not explicitly in the loss. This is consistent: the step-level CVaR penalty modifies the reward used to compute group advantages, which flows through to the DAPO gradient automatically.

---

## 5. Implementation Notes

### K=8 groups

Each T-step episode generates T × K = T × 8 training samples. With T ≈ 252 trading days, this gives ~2,016 (action, advantage) pairs per epoch, compared to ~252 for PPO. The higher sample density per episode should improve gradient quality, especially for the diversification constraint.

### No value function bootstrap

DAPO advantages are fully on-policy (computed from current-policy samples at the current timestep's asset returns). There is no temporal discounting or bootstrapping from a value function estimate. This means:
- No discount factor γ (DAPO is undiscounted by design — each timestep is evaluated independently)
- No risk of value function divergence in non-stationary markets
- Higher variance per sample, but offset by K-fold oversampling

### No VecNormalize

DAPO does not use SB3's VecNormalize (which normalises rewards across parallel environments). Group-relative normalisation provides the equivalent scaling: advantages are always in a comparable scale regardless of the absolute return magnitude. This makes DAPO training more robust to regime changes that shift the return distribution.

---

## 6. Configuration

```yaml
dapo:
  group_size:      8      # K: actions per state per step
  eps_low:         0.1    # asymmetric clip lower bound
  eps_high:        0.2    # asymmetric clip upper bound
  filter_min_std:  0.01   # minimum group std to retain group
  ent_coef:        0.01   # entropy regularisation coefficient
  n_epochs:        4      # gradient passes per rollout
  batch_size:      256
  max_grad_norm:   0.5
  learning_rate:   3.0e-4
  total_timesteps: 500000
```

These parameters are not yet in the default `config.yaml` — DAPO defaults fall back to PPO settings where applicable.

---

## 7. References

- Shao, Z., et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476.
- DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.  *(Group-relative advantage estimation first introduced here)*
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.  *(PPO baseline for clipping)*
