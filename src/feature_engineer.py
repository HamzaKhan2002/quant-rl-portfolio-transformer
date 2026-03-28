"""
feature_engineer.py — Price feature computation, observation matrix
construction, and z-score normalization.

Responsibilities:
  1. fetch_price_data()      — download OHLCV data via yfinance
  2. compute_price_features() — log returns (1d / 5d / 21d), realised vol (21d),
                                trend features, simple returns (portfolio reward)
  3. build_observation_matrix() — join sentiment+regime + price features,
                                  pivot to (T, 94) obs array and (T, 5) forward
                                  return array ready for PortfolioEnv
  4. Normalizer               — z-score fitted on train split only

Observation layout (94 features, ordered):
  [AAPL×18, MSFT×18, GOOGL×18, AMZN×18, TSLA×18, prob_low, prob_med,
   prob_high, prob_crash]

Per-ticker features (18, in order):
  log_return_1d, log_return_5d, log_return_21d, realized_vol_21d,
  tone_score, risk_score, growth_outlook, x_sentiment, hype_velocity,
  trending_score, sentiment_velocity, risk_momentum,
  days_since_filing, has_sentiment_data, has_x_data,
  sma_ratio_50_200, sma_dist_200, drawdown_from_ath
"""

from __future__ import annotations

import warnings
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # allow import without yfinance for unit tests
    yf = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Feature name constants (canonical ordering)
# ---------------------------------------------------------------------------

TICKER_FEATURES: List[str] = [
    "log_return_1d",
    "log_return_5d",
    "log_return_21d",
    "realized_vol_21d",
    "tone_score",
    "risk_score",
    "growth_outlook",
    "x_sentiment",
    "hype_velocity",
    "trending_score",
    "sentiment_velocity",
    "risk_momentum",
    "days_since_filing",
    "has_sentiment_data",
    "has_x_data",           # 1 if real X/Twitter data exists for this year, 0 otherwise
    "sma_ratio_50_200",     # SMA(50)/SMA(200) - 1: trend direction (golden/death cross)
    "sma_dist_200",         # price/SMA(200) - 1: distance from 200-day moving average
    "drawdown_from_ath",    # (price - 252d rolling max) / 252d rolling max
]

SHARED_FEATURES: List[str] = ["prob_low", "prob_med", "prob_high", "prob_crash"]

# String columns that must be dropped before building the numeric obs matrix
_DROP_COLS = ["data_source", "regime_label"]


# ---------------------------------------------------------------------------
# 1. Price data fetch
# ---------------------------------------------------------------------------

def fetch_price_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download adjusted close prices for *tickers* via yfinance.

    Parameters
    ----------
    tickers : list of str
        E.g. ``["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]``.
    start_date, end_date : str
        ISO 8601 dates.  Extra buffer before the window start is added
        internally so that rolling features at the first date are valid.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame indexed by Date with one column per ticker (adjusted
        close price).  Dates are business-day aligned.
    """
    if yf is None:
        raise ImportError("yfinance is required for fetch_price_data()")

    # Download with a 260-day buffer so SMA(200) and 252-day rolling max are
    # valid at the first date (200 trading days + margin for holidays)
    buffer_start = (pd.Timestamp(start_date) - pd.offsets.BDay(260)).strftime("%Y-%m-%d")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers,
            start=buffer_start,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )

    # yfinance returns MultiIndex columns (price_type, ticker) when len(tickers) > 1
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][tickers]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"

    return prices


# ---------------------------------------------------------------------------
# 2. Price feature computation
# ---------------------------------------------------------------------------

def compute_price_features(
    prices: pd.DataFrame,
    tickers: List[str],
) -> pd.DataFrame:
    """
    Compute per-ticker price features from a wide price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide DataFrame (Date index, one column per ticker — adjusted close).
    tickers : list of str
        Canonical ticker order.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
          Date, Ticker, log_return_1d, log_return_5d, log_return_21d,
          realized_vol_21d, simple_return_1d

        Rows with NaN (first 21 days of rolling window warm-up) are **kept**
        but filled with 0.0 to avoid gaps in the observation matrix.
        Callers that need only fully-valid rows can filter by date.
    """
    records = []
    for ticker in tickers:
        s = prices[ticker].dropna()

        log_ret = np.log(s / s.shift(1))
        simple_ret = (s / s.shift(1)) - 1.0

        # Trend features
        sma_50  = s.rolling(50).mean()
        sma_200 = s.rolling(200).mean()
        rolling_max_252 = s.rolling(252).max()

        df_t = pd.DataFrame({
            "Date": s.index,
            "Ticker": ticker,
            "log_return_1d": log_ret.values,
            "log_return_5d": np.log(s / s.shift(5)).values,
            "log_return_21d": np.log(s / s.shift(21)).values,
            "realized_vol_21d": log_ret.rolling(21).std().values,
            "simple_return_1d": simple_ret.values,
            "sma_ratio_50_200": (sma_50 / sma_200 - 1.0).values,
            "sma_dist_200": (s / sma_200 - 1.0).values,
            "drawdown_from_ath": ((s - rolling_max_252) / rolling_max_252).values,
        })

        # Fill NaN from rolling warm-up with 0.0
        df_t = df_t.fillna(0.0)
        records.append(df_t)

    return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Observation matrix construction
# ---------------------------------------------------------------------------

def build_observation_matrix(
    merged_df: pd.DataFrame,
    price_features_df: pd.DataFrame,
    tickers: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine sentiment+regime features with price features and produce arrays
    ready for PortfolioEnv.

    The observation at time *t* uses only information observable at close of
    day *t* (backward-looking features).  The return array contains the
    forward simple return from day *t* to day *t+1*, used inside the env to
    compute the portfolio reward.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Output of ``data_loader.load_merged_features`` (or a date-filtered
        split thereof).  Long format: one row per (Date, Ticker).
    price_features_df : pd.DataFrame
        Output of ``compute_price_features``.  Long format: one row per
        (Date, Ticker).
    tickers : list of str
        Canonical ticker order (must match config["tickers"]).

    Returns
    -------
    obs_array : np.ndarray, shape (T, 94)
        Observation matrix.  Column layout:
          [AAPL×18 | MSFT×18 | GOOGL×18 | AMZN×18 | TSLA×18 | regime×4]
    forward_return_array : np.ndarray, shape (T, 5)
        Per-ticker simple forward returns aligned with obs_array.
        ``forward_return_array[t, i]`` = simple return of ticker i from
        day t to day t+1.
    dates : np.ndarray of np.datetime64
        Business days corresponding to each row in obs_array / forward_return_array.
    """
    # --- Join merged (sentiment + regime) with price features ---------------
    df = merged_df.merge(price_features_df, on=["Date", "Ticker"], how="inner")

    # Drop string / categorical columns not used as numeric features
    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns])

    # Cast bool → int for has_sentiment_data and has_x_data
    df["has_sentiment_data"] = df["has_sentiment_data"].astype(int)
    # has_x_data may be absent (older sentiment CSV without extended data)
    if "has_x_data" not in df.columns:
        df["has_x_data"] = 0
    df["has_x_data"] = df["has_x_data"].astype(int)

    # Ensure days_since_filing is int64 (Windows dtype safety — CONTEXT.md §5.1)
    df["days_since_filing"] = df["days_since_filing"].astype("int64")

    # --- Sort and align tickers ---------------------------------------------
    ticker_rank = {t: i for i, t in enumerate(tickers)}
    df = df.copy()
    df["_rank"] = df["Ticker"].map(ticker_rank)
    df = df.sort_values(["Date", "_rank"]).drop(columns=["_rank"])

    # --- Build wide per-ticker feature matrix --------------------------------
    # pivot_table returns MultiIndex columns: level-0 = feature, level-1 = Ticker
    ticker_pivot = df.pivot_table(
        index="Date",
        columns="Ticker",
        values=TICKER_FEATURES,
        aggfunc="first",
    )
    # Swap levels so we get (Ticker, feature) and then select canonical order:
    # [AAPL_feat0, AAPL_feat1, ..., MSFT_feat0, ..., TSLA_feat13]
    ticker_pivot = ticker_pivot.reorder_levels([1, 0], axis=1)  # now (Ticker, feature)
    ordered_cols = [(ticker, feat) for ticker in tickers for feat in TICKER_FEATURES]
    ticker_pivot = ticker_pivot[ordered_cols]  # enforce canonical block order

    # --- Shared regime features (same for all tickers on a given date) ------
    shared_pivot = df[df["Ticker"] == tickers[0]].set_index("Date")[SHARED_FEATURES]
    shared_pivot = shared_pivot.reindex(ticker_pivot.index)

    # --- Build forward return matrix ----------------------------------------
    return_pivot = df.pivot_table(
        index="Date",
        columns="Ticker",
        values="simple_return_1d",
        aggfunc="first",
    )[tickers]  # enforce canonical ticker order

    # Forward return at day t = backward simple return recorded at day t+1
    # Shift return_matrix up by one row: forward_return[t] = return_matrix[t+1]
    forward_returns = return_pivot.shift(-1).fillna(0.0)

    # --- Align all matrices on the same date index ---------------------------
    common_dates = (
        ticker_pivot.index
        .intersection(shared_pivot.index)
        .intersection(forward_returns.index)
    )
    common_dates = common_dates.sort_values()

    ticker_matrix = ticker_pivot.loc[common_dates].values.astype(np.float32)    # (T, 90)
    shared_matrix = shared_pivot.loc[common_dates].values.astype(np.float32)    # (T, 4)
    return_matrix  = forward_returns.loc[common_dates].values.astype(np.float32) # (T, 5)

    obs_array = np.hstack([ticker_matrix, shared_matrix])  # (T, 94)

    # Replace any remaining NaN / Inf with 0.0 (safety guard)
    obs_array     = np.nan_to_num(obs_array,     nan=0.0, posinf=0.0, neginf=0.0)
    return_matrix = np.nan_to_num(return_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return obs_array, return_matrix, common_dates.values


# ---------------------------------------------------------------------------
# 4. Normalizer
# ---------------------------------------------------------------------------

class Normalizer:
    """
    Column-wise z-score normalizer fitted on training data only.

    Attributes
    ----------
    mean_ : np.ndarray, shape (n_features,)
    std_  : np.ndarray, shape (n_features,)
    eps   : float — added to std to prevent division by zero
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_:  np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Normalizer":
        """Compute mean and std from training array *X* (shape: T × F)."""
        self.mean_ = X.mean(axis=0).astype(np.float32)
        self.std_  = X.std(axis=0).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply z-score normalisation. Returns float32 array of same shape."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer has not been fitted. Call fit() first.")
        return ((X - self.mean_) / (self.std_ + self.eps)).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on *X* then return the normalised array."""
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        """Persist the fitted normalizer to *path* (joblib format)."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        """Load a previously saved normalizer from *path*."""
        return joblib.load(path)
