# Walk-Forward Cross-Validation

## 1. Motivation

Standard random-split cross-validation is ill-suited for financial time series because:

1. **Temporal leakage**: Shuffling breaks the time ordering, allowing the model to implicitly learn patterns from "future" periods during training.
2. **Non-stationarity**: Market regimes (volatility clusters, trend/mean-reversion cycles) shift over time. A model trained on one regime may fail in another.
3. **Single test period**: Evaluating on one held-out year conflates model quality with the particular characteristics of that year (e.g., a bull vs. bear market).

Walk-forward cross-validation addresses all three issues by training and evaluating on multiple non-overlapping test periods, always preserving the temporal ordering.

---

## 2. Fold Design

Six evaluation folds are defined. Five full-model folds cover the 2014–2025 period
(including the COVID crash, 2022 bear market, and 2024 AI bull run). Fold R provides
an additional evaluation on the 2022 H2 bear market using the regime-only model.

| Fold | Model | obs_dim | Train | Val | Test | Market Regime |
|------|-------|---------|-------|-----|------|----------------|
| 1    | Full  | 79      | 2014–2018 | 2019 | 2020 | COVID crash (SPY -34%) |
| 2    | Full  | 79      | 2014–2020 | 2021 | 2022 | Bear market (-33% NASDAQ) |
| 3    | Full  | 79      | 2014–2021 | 2022 | 2023 | Recovery (+43% NASDAQ) |
| 4    | Full  | 79      | 2014–2022 | 2023 | 2024 | AI bull run (+54% NASDAQ) |
| 5    | Full  | 79      | 2014–2023 | 2024 | 2025 | Default split |
| R    | Regime-only | 34 | 2019–2021 | 2022-H1 | 2022-H2 | Bear (–33% NASDAQ) |

### 2.1 Data availability constraints

| Source | Available from | Available to | Note |
|--------|---------------|--------------|------|
| EDGAR 10-K sentiment (4 features/ticker) | 2014-01-03 | 2025-12-31 | Full history |
| X/Grok social (3 features/ticker) | 2022-01-01 | 2025-12-31 | `has_x_data=0` flag for pre-2022 |
| Regime probabilities | 2018-02-09 | 2026-02-27 | Effective start limited by ETH-USD history |
| Price data (yfinance) | Unlimited | — | |

> **Data constraint note**: The regime classifier uses SPY, QQQ, BTC-USD, and ETH-USD as
> inputs. ETH-USD first traded circa late 2017, so after the 60-day lookback warm-up, the
> first usable regime probability date is **2018-02-09**. This means Fold 1 effective
> training data starts 2018-02-09 (only ~10 months), and Fold 5 has the most data
> (~1,432 trading days from 2018-02-09 to 2023-12-31).

---

## 3. Regime Classifier Validation (STEP 2 Hard Requirements)

The Phase 1 Transformer classifier was retrained on the 2014–2025 period with
**inverse-frequency class weighting** (`nn.CrossEntropyLoss(weight=1/class_counts)`) to
prevent crash-state collapse.

### Key date validation pass/fail

| Date | Event | prob_crash | prob_high | PASS |
|------|-------|-----------|----------|------|
| 2020-03-16 | COVID selloff | **0.910** | 0.071 | YES (>0.3) |
| 2020-03-23 | COVID bottom | **0.802** | 0.142 | YES (>0.3) |
| 2022-06-13 | Bear market low | 0.058 | **0.567** | YES (>0.2) |
| 2022-10-13 | Bear market low | 0.010 | **0.567** | YES (>0.2) |

### Aggregate crash statistics (over 2018-02-09 to 2026-02-27)

| Metric | Value | Requirement | Status |
|--------|-------|-------------|--------|
| Days with prob_crash > 0.3 | 50 | >= 20 | PASS |
| Crash class fraction | 1.61% | >= 1.5% | PASS |
| High-vol class fraction | 15.3% | — | |
| Low-vol class fraction | 66.8% | — | |

---

## 4. Fold R: Regime-Only Observation

Fold R is the methodologically critical fold because the 2022 NASDAQ bear market
(-33% drawdown, driven by rate hikes and tech multiple compression) is entirely
inside the training window of full-model folds 1–3 and therefore cannot be evaluated
out-of-sample by them.

### 4.1 Why the bear market matters

The thesis claim is not "beat a buy-and-hold benchmark in a bull market." It is
"achieve better risk-adjusted returns in adverse market regimes." Fold R tests whether
the regime signal actually helps the agent reduce drawdown in a bear market.

### 4.2 Observation space (34-dim)

Without X/Grok social features, the per-ticker feature set reduces to:

| Index (within ticker block) | Feature | Source |
|----|---------|--------|
| 0 | `log_return_1d` | Price |
| 1 | `log_return_5d` | Price |
| 2 | `log_return_21d` | Price |
| 3 | `realized_vol_21d` | Price |
| 4 | `days_since_filing` | Placeholder = 0 |
| 5 | `has_sentiment_data` | Placeholder = 0 |

6 features × 5 tickers = 30 per-ticker dimensions. Plus 4 shared regime probabilities:
`prob_low`, `prob_med`, `prob_high`, `prob_crash`. Total: **34 dimensions**.

---

## 5. `has_x_data` Feature (15th per-ticker feature, obs_dim 79)

The 15th per-ticker feature `has_x_data` is a binary flag:
- **1**: Real-time X/Twitter social data was extracted via Grok for this ticker-year
- **0**: Pre-2022 row (Grok not queried), or Grok API returned None

This flag distinguishes genuine social signal from the zero-fill used for pre-2022 rows.
It enables the agent to weight the social features appropriately: when `has_x_data=0`,
the accompanying `x_sentiment`, `hype_velocity`, and `trending_score` are 0.0 placeholders.

---

## 6. Expanding vs. Rolling Window

The expanding window design is standard for financial ML:

1. **More data is better**: The policy gradient estimator benefits from more diverse episodes.
2. **Regime exposure**: Later folds include the 2022 bear AND 2023 bull in training, making
   the policy more robust.

---

## 7. Evaluation Protocol

Each fold trains `K` seeds (default: 4 seeds — 42, 1, 7, 2026) and reports:

1. **Per-seed test Sharpe** — seed variance captures training instability
2. **Ensemble test metrics** — averaged logits across seeds, softmax applied once
3. **Cross-fold aggregate** — mean ± std across folds

### Metrics reported per fold

| Metric | Definition |
|--------|-----------|
| Sharpe | \((\bar{r} - r_f) / \sigma_r \times \sqrt{252}\) |
| Sortino | \((\bar{r} - r_f) / \sigma_{\text{down}} \times \sqrt{252}\) |
| Max Drawdown | \(\max_t (\text{peak}_t - \text{NAV}_t) / \text{peak}_t\) |
| Calmar | Annualised return / Max Drawdown |
| Mean Daily Turnover | \(\mathbb{E}[\sum_i \|w_i(t) - w_i(t-1)\|]\) |

### Fold 1 smoke test (100k steps, seed 42, test year 2020 COVID crash)

| Metric | Value |
|--------|-------|
| Sharpe | 2.10 |
| Sortino | 2.61 |
| Max Drawdown | 0.345 |
| Annualised Return | 1.163 |
| Calmar | 3.37 |

---

## 8. Cross-Fold Significance

To assess whether the ensemble PPO outperforms equal-weight buy-and-hold across folds,
a **paired block bootstrap** test is used:

1. For each fold, collect daily excess returns: `ppo_returns - ewbh_returns`
2. Apply block bootstrap (block size = 21 trading days, 10,000 resamples)
3. A two-sided p-value is computed from the bootstrap distribution of the mean excess return
4. Full-model folds (1–5) are pooled for combined test; Fold R is analysed separately

---

## 9. CLI

```bash
# All folds (default), seed 42
python -m src.walk_forward

# Specific fold
python -m src.walk_forward --fold 1

# Multiple folds with ensemble seeds
python -m src.walk_forward --fold 1 2 3 4 5 R --seeds 42 1 7 2026 --use-tuned

# Output structure
output/walk_forward/
  fold_1/
    seed_42/best_model_seed_42.zip
    fold_metrics.json
  ...
  fold_R/  ...
  summary.json        # cross-fold aggregate statistics
```
