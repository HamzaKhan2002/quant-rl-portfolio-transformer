# Phase 3 Context — quant-rl-portfolio-transformer

**Source:** This file was generated from `quant-nlp-claude-grok-sentiment` (Phase 2) at the end of Phase 2.
Copy it to the root of `quant-rl-portfolio-transformer/` for a context-complete Cursor session.

---

## 1. Project Overview

This is Phase 3 of a 3-phase quantitative trading pipeline targeting a multi-asset equity portfolio (AAPL, MSFT, GOOGL, AMZN, TSLA). The pipeline was built in strict sequence: Phase 1 detects market regimes, Phase 2 scores NLP sentiment from SEC filings, and Phase 3 will train a PPO reinforcement-learning agent that consumes both.

**Phase 1 — `quant-multi-transformer-regimes`**
Market regime detection using a Transformer encoder trained on price/volatility features. Output: `regime_probs.csv` with columns `Date, prob_low, prob_med, prob_high, prob_crash, regime_label`. Date range: 2019-04-03 → 2026-02-27 (daily, business days only). Each row sums to 1.0 across the four probability columns.

**Phase 2 — `quant-nlp-claude-grok-sentiment`** *(this repo)*
NLP sentiment pipeline. Fetches SEC 10-K filings via `edgartools`, scores them with Claude (primary) or FinBERT (fallback), and fetches X/social sentiment via Grok. Scores are anchored at `report_date` (fiscal period end) and decayed exponentially with a 45-business-day half-life. Output: `sentiment_features.csv` (5215 rows × 13 cols, daily) and `sentiment_events.csv` (20 rows, one per ticker×year). Date range: 2022-01-03 → 2025-12-31.

**Phase 3 — `quant-rl-portfolio-transformer`** *(new repo — being built)*
PPO reinforcement-learning agent. Merges Phase 1 regime features + Phase 2 sentiment features + price data. Learns an optimal daily rebalancing policy for the 5-equity universe. The agent receives a ~17-column feature vector per (ticker, date) and outputs portfolio weights. Training should be restricted to rows where `has_sentiment_data == True` OR where `has_sentiment_data` is passed as a binary feature so the agent learns to act without sentiment signal too (see Section 4, item 3).

---

## 2. Phase 2 Output Schemas

### `sentiment_events.csv`

One row per (Ticker, year). Event-level anchor table before daily expansion.

| Column | Dtype | Description |
|--------|-------|-------------|
| `Date` | object (YYYY-MM-DD) | = `filing_date` = fiscal period end (`report_date`) |
| `Ticker` | object | AAPL, AMZN, GOOGL, MSFT, TSLA |
| `year` | int64 | Fiscal year (assigned by `report_date.year`) |
| `tone_score` | float64 | [-1, +1] bearish → bullish (Claude or FinBERT) |
| `risk_score` | float64 | [0, 1] minimal → severe risk language |
| `growth_outlook` | float64 | [-1, +1] contraction → expansion (Claude only, distinct from tone) |
| `x_sentiment` | float64 | [-1, +1] Grok social sentiment for ticker/year |
| `hype_velocity` | float64 | [-1, +1] panic → extreme hype (Grok) |
| `trending_score` | float64 | [0, +1] not trending → viral (Grok) |
| `sentiment_velocity` | float64 | YoY diff of `tone_score` per ticker |
| `risk_momentum` | float64 | YoY diff of `risk_score` per ticker |
| `data_source` | object | `"claude"`, `"finbert"`, or `"none"` |
| `filing_date` | object (YYYY-MM-DD) | Duplicate of `Date`; retained for pipeline internals |

**Shape:** (20, 13) — 5 tickers × 4 years (2022–2025)
**Date range:** 2022-06-30 → 2025-12-31
**Tickers:** `['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA']`

**Sample row (AAPL FY2024):**
```
Date                  2024-09-28      ← report_date, NOT 2024-11-01 (SEC filing_date)
Ticker                AAPL
year                  2024
tone_score            -0.2
risk_score             0.4
growth_outlook         0.1
x_sentiment            0.65
hype_velocity          0.73
trending_score         0.82
sentiment_velocity     0.0
risk_momentum          0.0
data_source            claude
filing_date            2024-09-28
```

---

### `sentiment_features.csv`

One row per (Ticker, business_day). Daily features after exponential decay.

| Column | Dtype | Description |
|--------|-------|-------------|
| `Date` | object (YYYY-MM-DD) | Business day |
| `Ticker` | object | AAPL, AMZN, GOOGL, MSFT, TSLA |
| `tone_score` | float64 | Decayed tone signal |
| `risk_score` | float64 | Decayed risk signal |
| `growth_outlook` | float64 | Decayed growth signal |
| `x_sentiment` | float64 | Decayed Grok social sentiment |
| `hype_velocity` | float64 | Decayed Grok hype velocity |
| `trending_score` | float64 | Decayed Grok trending score |
| `sentiment_velocity` | float64 | Decayed YoY tone delta |
| `risk_momentum` | float64 | Decayed YoY risk delta |
| `data_source` | object | Source of the last filing anchor (`"claude"`, `"finbert"`, `"none"`) |
| `days_since_filing` | int64 | Business days elapsed since last filing anchor |
| `has_sentiment_data` | bool | `True` = real scores active; `False` = pre-filing dates |

**Shape:** (5215, 13)
**Date range:** 2022-01-03 → 2025-12-31
**Tickers:** `['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA']`
**NaN count:** 0

**`has_sentiment_data` distribution:**
```
True  (active decay signal):  4,121 rows  (79.0%)
False (pre-filing-date rows): 1,094 rows  (21.0%)
```
The 1,094 `False` rows are **not dead data** — they are equity rows dated before each ticker's first fiscal-year-end filing (e.g., AAPL Jan 2022 – Sep 2022, before the FY2022 10-K anchor at `2022-09-24`). All sentiment scores are 0.0 for these rows. Phase 3 should pass `has_sentiment_data` as a binary input feature rather than filtering rows out.

**Sample row (AAPL, first day with sentiment):**
```
Date                  2022-09-23
Ticker                AAPL
tone_score            -0.2
risk_score             0.4
growth_outlook         0.2
x_sentiment            0.65
hype_velocity          0.47
trending_score         0.82
sentiment_velocity     0.0
risk_momentum          0.0
data_source            claude
days_since_filing      0
has_sentiment_data     True
```

---

## 3. Phase 3 Merge Recipe

```python
import pandas as pd

sentiment = pd.read_csv("../quant-nlp-claude-grok-sentiment/output/sentiment_features.csv")
regime    = pd.read_csv("../quant-multi-transformer-regimes/output/regime_probs.csv")

sentiment["Date"] = pd.to_datetime(sentiment["Date"])
regime["Date"]    = pd.to_datetime(regime["Date"])

# Inner merge — drops ~38 US holiday dates present in sentiment
# (freq="B" includes New Year/MLK/etc.) but absent from regime calendar.
# Left merge would introduce NaN rows in regime columns.
merged = sentiment.merge(regime, on="Date", how="inner")
# Result: ~5180 rows × 17 columns, 0 NaN
```

**Why inner, not left:** `sentiment_features.csv` is generated with `pd.date_range(..., freq="B")` which includes all business days including US bank holidays. Phase 1's regime model was trained only on trading days (NYSE calendar), so ~38 market holidays are absent from `regime_probs.csv`. An inner merge silently drops these holiday rows — this is correct behavior.

**Full merged schema (17 columns, in order):**

| # | Column | Source |
|---|--------|--------|
| 1 | `Date` | shared key |
| 2 | `Ticker` | Phase 2 |
| 3 | `tone_score` | Phase 2 |
| 4 | `risk_score` | Phase 2 |
| 5 | `growth_outlook` | Phase 2 |
| 6 | `x_sentiment` | Phase 2 |
| 7 | `hype_velocity` | Phase 2 |
| 8 | `trending_score` | Phase 2 |
| 9 | `sentiment_velocity` | Phase 2 |
| 10 | `risk_momentum` | Phase 2 |
| 11 | `data_source` | Phase 2 |
| 12 | `days_since_filing` | Phase 2 |
| 13 | `has_sentiment_data` | Phase 2 |
| 14 | `prob_low` | Phase 1 |
| 15 | `prob_med` | Phase 1 |
| 16 | `prob_high` | Phase 1 |
| 17 | `prob_crash` | Phase 1 |

Note: `regime_label` (string) from Phase 1 can be added as column 18 if needed for logging, but should not be passed as a raw PPO feature — use the four probability columns instead.

---

## 4. Key Architecture Decisions

**1. `report_date` vs `filing_date`**
All filing anchors use `report_date` (fiscal period end), not `filing_date` (SEC submission date). AAPL's fiscal year ends in late September; MSFT's ends in late June. Using `filing_date` would misalign fiscal years by 1–2 months and create spurious year assignments (e.g., AAPL FY2024 filed 2024-11-01 would be tagged as FY2024 by filing date but could fall in the wrong decay window). Correct anchors: `AAPL FY2024 = 2024-09-28`, `MSFT FY2024 = 2024-06-30`. The `edgar_fetcher.py` date filter spans `{year}-01-01` to `{year+1}-06-30` to catch late filers, then selects the candidate where `report_date.year == year`.

**2. Exponential decay, halflife=45 business days**
Grounded in PEAD (Post-Earnings Announcement Drift) literature (Bernard & Thomas 1989): approximately 60 calendar days ≈ 45 business days. The decay formula is: `score(t) = score_at_filing × exp(−ln(2)/45 × bdays_since)`. Day 0 = 100%, day 45 = 50%, day 90 = 25%. The `days_since_filing` column is exposed in `sentiment_features.csv`. **Phase 3 must pass `days_since_filing` as a PPO input feature** so the agent can learn its own effective discount weighting rather than having the discount hard-coded.

**3. `has_sentiment_data` flag — do NOT filter out False rows**
`True` = equity row with an active Claude/FinBERT decay signal. `False` = equity row before the first annual 10-K was scored (leading-edge artifact at the start of the time series). The agent must learn to act in both regimes. Filtering to `True`-only rows would remove the realistic scenario where no filing signal is available (e.g., the first months of any new ticker added to the universe). Pass `has_sentiment_data` as a binary input feature (0/1) and keep all rows.

**4. Crypto removed — equities only**
BTC-USD and ETH-USD were removed. No SEC filings exist for crypto, and their market dynamics (24/7 trading, 80–200% annual volatility, on-chain signal sources) differ fundamentally from equities. Phase 3 is AAPL, MSFT, GOOGL, AMZN, TSLA only. Adding crypto in future requires a **separate dedicated pipeline**, not a modification of this one.

**5. Claude chain-of-thought at `temperature=0`**
At `temperature=0` Claude collapses numeric scores to a single mode (observed: `risk_score=0.70` for every large-cap filing). The fix: a `risk_factors_noted` field in the JSON output forces the model to identify 3 company-specific risks before scoring. This breaks greedy-decoding symmetry and produces meaningful variance (`risk_score` range = 0.20). **Do not remove `risk_factors_noted` from the Claude prompt** if re-prompting in Phase 3.

**6. SDK + proxy conflict — always pass `base_url` explicitly**
The Anthropic SDK auto-reads `ANTHROPIC_BASE_URL` from the environment. In this repo, that env var points to a VS Code LM proxy (`http://localhost:10268`). Any module instantiating `anthropic.Anthropic()` must pass `base_url="https://api.anthropic.com"` explicitly when using a real `sk-ant-api03-...` key, or else requests will be routed to the proxy and rejected with 401. This applies to **any new module in Phase 3** that calls the Anthropic API.

```python
# CORRECT — explicit base_url bypasses ANTHROPIC_BASE_URL env var
client = anthropic.Anthropic(
    api_key=os.getenv("CLAUDE_API_KEY"),
    base_url="https://api.anthropic.com",
)
```

**7. `growth_outlook` is a genuinely distinct dimension (Claude only)**
For FinBERT, `growth_outlook == tone_score` (FinBERT has no dedicated growth label — it is a proxy). For Claude, `growth_outlook` is prompted separately with calibration anchors and empirically diverges from `tone_score`. Per-ticker Pearson r with 30-day forward returns (N=4 per ticker, directional only): TSLA r=+0.97, GOOGL r=−0.78, AAPL r=−0.68. Do not drop `growth_outlook` from the feature set — treat it as a separate signal from `tone_score`.

---

## 5. Bugs to Not Repeat

**1. `int64`/`int32` dtype mismatch on Windows**
`pd.merge_asof` requires identical dtypes on join key columns. On Windows, `numpy` defaults to `int32` for `np.arange()`. This silently corrupts the merge without raising an error. Always use explicit `dtype="int64"` for any integer index column used in `merge_asof`.
```python
bday_df = pd.DataFrame({"bday_idx": np.arange(len(bdays), dtype="int64")})
tkr_events["filing_bday_idx"] = tkr_events["filing_bday_idx"].astype("int64")
```

**2. `ANTHROPIC_BASE_URL` proxy hijacking (see Section 4, item 6)**
Even `anthropic.Anthropic(api_key=real_key)` — with no explicit `base_url` — will use the proxy if `ANTHROPIC_BASE_URL` is set in the environment. The result is a 401. Always pass `base_url` explicitly.

**3. `filing.text()` vs `filing.obj()`**
`filing.text()` returns the raw HTML from the SEC EDGAR cover page. The cover page contains near-zero sentiment signal and is mostly boilerplate (table of contents, signatures). Always use:
```python
obj = filing.obj()
risk_factors    = str(obj.risk_factors)          # Item 1A
mgmt_discussion = str(obj.management_discussion) # Item 7
```

**4. `amendments=True` in `get_filings` (edgartools default)**
`company.get_filings(form="10-K")` returns both 10-K and 10-K/A (amendments) by default. Amendments often have empty Item 1A / Item 7 sections (they only amend specific exhibits). Always use `amendments=False`:
```python
filings = company.get_filings(form="10-K", amendments=False)
```

**5. `ffill(limit=N)` cliff artifact**
`ffill(limit=90)` creates a hard cliff at day 90 where the score drops from `score_at_filing × (decayed small value)` to 0.0 in a single step. This daily jump of ~0.05–0.2 is large relative to the signal and creates a spike in `max_daily_change` that confuses gradient-based learners (PPO actor loss spike). The Phase 2 fix is exponential decay. Never reintroduce `ffill` with a finite limit for sentiment propagation in Phase 3.

---

## 6. Test Patterns to Follow

Phase 2 tests are in `tests/test_sentiment_pipeline.py`. Phase 3 tests should mirror these patterns:

**Mock all external calls in unit tests:**
```python
# No network, no API key required — use static fixtures
def test_mock_claude_parser():
    from src.claude_parser import parse_structured_response
    raw = '{"sentiment": "positive", "score": 0.8}'
    result = parse_structured_response(raw)
    assert result["score"] == 0.8
```

**Test mathematical invariants with exact values, not just "not NaN":**
```python
# Phase 2 example — test that halflife=45 gives exactly 50% at day 45
score_45 = df.loc[day_plus_45, "tone_score"]
assert 0.35 < score_45 < 0.55  # ~50% of 0.8
```

**Test the decay cliff is absent:**
```python
max_daily_jump = series.diff().abs().max()
assert max_daily_jump < 0.05, f"Cliff detected: {max_daily_jump:.4f}"
```

**Explicitly test edge cases:**
- Pre-filing rows: `score == 0.0` before first anchor
- Single-filing tickers: `sentiment_velocity == 0.0` for first year
- Windows dtype safety: index column dtypes after `merge_asof`

**Mark integration tests clearly** (require `.env` / network):
```python
def test_edgar_fetcher_sections():
    """Real EDGAR fetch — requires internet."""
    sections = fetch_filing_sections("AAPL", "10-K", 2023)
    if sections is None:
        print("WARN — EDGAR fetch returned None (check .env)")
        return
    assert len(sections["risk_factors"]) > 1000
    assert "<html" not in sections["risk_factors"].lower()
```

---

## 7. File Paths (relative to Phase 3 repo root)

```
Phase 2 daily features : ../quant-nlp-claude-grok-sentiment/output/sentiment_features.csv
Phase 2 event anchors  : ../quant-nlp-claude-grok-sentiment/output/sentiment_events.csv
Phase 1 regime probs   : ../quant-multi-transformer-regimes/output/regime_probs.csv
Phase 3 repo root      : ./  (quant-rl-portfolio-transformer/)
```

Assume all three repos are siblings under the same parent directory. If directory structure differs, update paths accordingly — do not hardcode absolute paths in Phase 3 source files.

**Phase 2 source modules (for reference, do not copy):**
```
../quant-nlp-claude-grok-sentiment/src/edgar_fetcher.py      # edgartools wrapper
../quant-nlp-claude-grok-sentiment/src/claude_parser.py      # Anthropic API + FinBERT fallback
../quant-nlp-claude-grok-sentiment/src/grok_x_fetcher.py     # xAI Grok social sentiment
../quant-nlp-claude-grok-sentiment/src/sentiment_pipeline.py # orchestrator + exponential decay
../quant-nlp-claude-grok-sentiment/config.yaml               # model names, tickers, years
```

---

## Ready to Build — Start with the Phase 3 prompt.
