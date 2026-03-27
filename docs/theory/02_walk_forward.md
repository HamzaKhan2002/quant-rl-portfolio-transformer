# Walk-Forward Cross-Validation

## 1. Motivation

Standard random-split cross-validation is ill-suited for financial time series because:

1. **Temporal leakage**: Shuffling breaks the time ordering, allowing the model to implicitly learn patterns from "future" periods during training.
2. **Non-stationarity**: Market regimes (volatility clusters, trend/mean-reversion cycles) shift over time. A model trained on one regime may fail in another.
3. **Single test period**: Evaluating on one held-out year conflates model quality with the particular characteristics of that year (e.g., a bull vs. bear market).

Walk-forward cross-validation addresses all three issues by training and evaluating on multiple non-overlapping test periods, always preserving the temporal ordering.

---

## 2. Fold Design

Three evaluation folds are defined, constrained by data availability:

| Fold | Model | obs_dim | Train | Val | Test | Market |
|------|-------|---------|-------|-----|------|--------|
| 1    | Full  | 74      | 2022  | 2023 | 2024 | Bull (+54% NASDAQ) |
| 2    | Full  | 74      | 2022–2023 | 2024 | 2025 | Bull-to-mixed |
| R    | Regime-only | 34 | 2019–2021 | 2022-H1 | 2022-H2 | Bear (–33% NASDAQ) |

**Data availability constraints (hard limits)**:

| Source | Available from | Available to |
|--------|---------------|--------------|
| Sentiment (8 features/ticker) | 2022-01-03 | 2025-12-31 |
| Regime probabilities | 2019-04-03 | 2026-02-27 |
| Price data (yfinance) | Unlimited | — |

These constraints eliminate any possibility of 5+ year walk-forward folds using the full 74-dim model. Only two non-overlapping full-model folds are feasible (test years 2024 and 2025).

**Why only 3 folds instead of 5?**

The original plan proposed a 5-fold walk-forward from 2014. This is infeasible because:
- Sentiment data (the second phase of the pipeline) begins 2022-01-03.
- An inner-join between sentiment and other data sources produces an empty DataFrame for any training window starting before 2022.
- Fold R compensates by testing the 2022 bear market using a reduced-dimensionality model that does not require sentiment.

---

## 3. Fold R: Regime-Only Observation

Fold R is the methodologically critical fold because the 2022 NASDAQ bear market (-33% drawdown, driven by rate hikes and tech multiple compression) is entirely inside the training window of both full-model folds and therefore cannot be evaluated out-of-sample by them.

### 3.1 Why the bear market matters

The thesis claim is not "beat a buy-and-hold benchmark in a bull market." It is "achieve better risk-adjusted returns in adverse market regimes." The 60/40 benchmark achieves Sharpe 1.30 with MDD 11% partly because bonds provided genuine diversification in 2025 — a characteristic that is regime-dependent. Fold R tests whether the regime signal actually helps the agent reduce drawdown in a bear market.

### 3.2 Observation space (34-dim)

Without sentiment data, the per-ticker feature set reduces to:

| Index (within ticker block) | Feature | Source |
|----|---------|--------|
| 0 | `log_return_1d` | Price |
| 1 | `log_return_5d` | Price |
| 2 | `log_return_21d` | Price |
| 3 | `realized_vol_21d` | Price |
| 4 | `days_since_filing` | Placeholder = 0 |
| 5 | `has_sentiment_data` | Placeholder = 0 |

6 features × 5 tickers = 30 per-ticker dimensions. Plus 4 shared regime probabilities: `prob_low`, `prob_med`, `prob_high`, `prob_crash`. Total: **34 dimensions**.

The two placeholder features at indices 4 and 5 are set to zero for all time steps. This is equivalent to telling the agent "no sentiment filings were observed" — consistent with the pre-2022 period when the sentiment pipeline did not exist.

### 3.3 Regime signal note

Analysis of the Phase 1 Transformer classifier across all available dates (2019–2026) reveals:
- `prob_crash` never exceeds 0.005 across 2,447 trading days
- The 2022 NASDAQ bear market is labelled `medium_vol` (not `high_vol` or `crash`)
- `high_vol` activations: only 27 days in COVID March–April 2020

**Consequence**: The regime signal provides a 2-state effective signal (low_vol vs. medium_vol), not 4-state. Fold R regime-conditional analysis must focus on cash allocation differences between these two soft states, not on `crash` → cash pivots.

---

## 4. Expanding vs. Rolling Window

Fold 1 trains on 1 year (2022), Fold 2 trains on 2 years (2022–2023). This is an **expanding window** design, which is standard for financial ML for two reasons:

1. **More data is better for PPO**: The policy gradient estimator benefits from more diverse episodes.
2. **Regime exposure**: Fold 2 sees both the 2022 bear (medium_vol) and 2023 bull (low_vol) during training, making it more robust.

A rolling window (fixed train length across folds) would be appropriate if there were concerns about concept drift making older data harmful. Given only 4 years of sentiment data, this is not yet a concern.

---

## 5. Evaluation Protocol

Each fold trains `K` seeds (default: 4 seeds — 42, 1, 7, 2026) and reports:

1. **Per-seed test Sharpe** — seed variance captures training instability
2. **Ensemble test metrics** — averaged logits across seeds, softmax applied once
3. **Cross-fold aggregate** — mean ± std across folds (excluding Fold R from full-model comparison)

### Metrics reported per fold

| Metric | Definition |
|--------|-----------|
| Sharpe | \((\bar{r} - r_f) / \sigma_r \times \sqrt{252}\) |
| Sortino | \((\bar{r} - r_f) / \sigma_{\text{down}} \times \sqrt{252}\) |
| Max Drawdown | \(\max_t (\text{peak}_t - \text{NAV}_t) / \text{peak}_t\) |
| Calmar | Annualised return / Max Drawdown |
| Mean Daily Turnover | \(\mathbb{E}[\sum_i \|w_i(t) - w_i(t-1)\|]\) |

---

## 6. Cross-Fold Significance

To assess whether the ensemble PPO outperforms equal-weight buy-and-hold across folds, a **paired block bootstrap** test is used:

1. For each fold, collect daily excess returns: `ppo_returns - ewbh_returns`
2. Apply block bootstrap (block size = 21 trading days, 10,000 resamples) to each fold's excess return series
3. A two-sided p-value is computed from the bootstrap distribution of the mean excess return
4. Folds 1 and 2 (full model) are pooled for a combined test; Fold R is analysed separately

---

## 7. CLI

```bash
# All folds, single seed
python -m src.walk_forward

# Specific fold
python -m src.walk_forward --fold R

# Ensemble of 4 seeds with tuned HPs
python -m src.walk_forward --seeds 42 1 7 2026 --use-tuned

# Output structure
output/walk_forward/
  fold_1/
    seed_42/best_model_seed_42.zip
    seed_1/ best_model_seed_1.zip
    ...
    normalizer.pkl
    fold_metrics.json
  fold_2/  ...
  fold_R/  ...
  summary.json        # cross-fold aggregate statistics
```
