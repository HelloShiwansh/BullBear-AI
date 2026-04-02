"""
feature_engineering.py
-----------------------
Adds technical indicators and calendar/seasonality features to cleaned OHLCV data.

Indicators implemented:
  - Moving Averages (MA10, MA30) — trend direction
  - RSI (14-period)              — momentum / overbought-oversold
  - MACD (12/26/9)               — momentum crossover signals
  - Bollinger Bands (20-day, 2σ) — volatility + mean-reversion signals
  - Daily return + Rolling vol   — raw price behaviour
  - Log volume                   — activity proxy

Total features: 24 (21 technical+calendar + 3 sentiment placeholders)

Sentiment columns (sentiment_score, sentiment_magnitude, sentiment_article_count)
are always present in FEATURE_COLUMNS. When pipeline runs without sentiment
(use_sentiment=False), pipeline.py fills these with 0.0 so the schema is
always identical regardless of whether sentiment is enabled.

Design principles:
- Every computation uses only past or present data (no future leakage).
- Rolling windows use min_periods so warm-up rows produce NaN, not silent zeros.
- All features are scale-invariant ratios or bounded indicators — comparable
  across tickers with very different price levels (₹200 Wipro vs ₹3500 TCS).
- Cyclical encoding for calendar features so Dec→Jan is a small step, not 11.

Scaling note:
  daily_return and volatility are unbounded — StandardScale them inside
  the ML pipeline AFTER the train/test split to prevent leakage.
"""

import numpy as np
import pandas as pd

try:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD as TAmacd
    from ta.volatility import BollingerBands as TAbb
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("[WARN] `ta` library not found. RSI / MACD / BB computed manually.")


# ── Constants ─────────────────────────────────────────────────────────────────

MA_SHORT         = 10
MA_LONG          = 30
RSI_PERIOD       = 14
VOLATILITY_WINDOW = 10
LOG_VOLUME       = True

MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9

BB_WINDOW        = 20
BB_STD           = 2.0


# ── Manual fallback calculations ──────────────────────────────────────────────

def _compute_rsi_manual(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's smoothed RSI — matches ta library output exactly."""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_macd_manual(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD line, signal line, histogram.
    min_periods=slow ensures both EMAs have enough history before producing values.
    Returns: (macd_line, signal_line, histogram)
    """
    ema_fast    = close.ewm(span=fast,   min_periods=slow, adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   min_periods=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _compute_bb_manual(
    close: pd.Series,
    window: int = BB_WINDOW,
    num_std: float = BB_STD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands: upper, mid, lower.
    Uses population std (ddof=0) and min_periods=window.
    Returns: (upper, mid, lower)
    """
    mid   = close.rolling(window, min_periods=window).mean()
    std   = close.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


# ── A. Technical Indicators ───────────────────────────────────────────────────

def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add MACD features normalised by Close for cross-ticker comparability.

    macd_line_ratio   = (EMA12 - EMA26) / Close
    macd_signal_ratio = EMA9(macd_line) / Close
    macd_hist_ratio   = (line - signal) / Close  ← most ML-useful (momentum acceleration)
    """
    df    = df.copy()
    close = df["Close"]

    if TA_AVAILABLE:
        obj         = TAmacd(close=close, window_fast=MACD_FAST,
                             window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
        macd_line   = obj.macd()
        signal_line = obj.macd_signal()
        histogram   = obj.macd_diff()
    else:
        macd_line, signal_line, histogram = _compute_macd_manual(close)

    df["macd_line_ratio"]   = macd_line   / close
    df["macd_signal_ratio"] = signal_line / close
    df["macd_hist_ratio"]   = histogram   / close
    return df


def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Bollinger Band features — both scale-invariant.

    bb_width = (upper - lower) / mid   — volatility regime
    bb_pct   = (close - lower) / (upper - lower)  — position in band
    """
    df    = df.copy()
    close = df["Close"]

    if TA_AVAILABLE:
        obj   = TAbb(close=close, window=BB_WINDOW, window_dev=BB_STD)
        upper = obj.bollinger_hband()
        mid   = obj.bollinger_mavg()
        lower = obj.bollinger_lband()
    else:
        upper, mid, lower = _compute_bb_manual(close)

    band_range      = upper - lower
    df["bb_width"]  = band_range / mid.replace(0, np.nan)
    df["bb_pct"]    = (close - lower) / band_range.replace(0, np.nan)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to the OHLCV DataFrame.

    All rolling windows use min_periods = window_size so the warm-up period
    produces NaN rather than misleading partial-window values. These NaN rows
    are dropped once in pipeline.py after target creation.
    """
    df    = df.copy()
    close = df["Close"]

    # Moving averages + ratios
    df["ma10"]       = close.rolling(MA_SHORT, min_periods=MA_SHORT).mean()
    df["ma30"]       = close.rolling(MA_LONG,  min_periods=MA_LONG).mean()
    df["ma10_ratio"] = close / df["ma10"]
    df["ma30_ratio"] = close / df["ma30"]

    # RSI
    if TA_AVAILABLE:
        df["rsi"] = RSIIndicator(close=close, window=RSI_PERIOD).rsi()
    else:
        df["rsi"] = _compute_rsi_manual(close, period=RSI_PERIOD)

    # MACD
    df = add_macd(df)

    # Bollinger Bands
    df = add_bollinger_bands(df)

    # Daily return + rolling volatility
    df["daily_return"] = close.pct_change()
    df["volatility"]   = (
        df["daily_return"]
        .rolling(VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW)
        .std()
    )

    # Volume
    df["log_volume"] = np.log1p(df["Volume"]) if LOG_VOLUME else df["Volume"]

    return df


# ── B. Calendar / Seasonality Features ───────────────────────────────────────

def _cyclical_encode(
    series: pd.Series, max_val: int
) -> tuple[pd.Series, pd.Series]:
    """
    Encode a cyclical integer as (sin, cos) pair.
    Ensures Dec→Jan is a small step, not 11 units apart.
    Returns: (sin_series, cos_series)
    """
    sin = np.sin(2 * np.pi * series / max_val)
    cos = np.cos(2 * np.pi * series / max_val)
    return sin, cos


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add seasonality features from the Date index.

    Raw integers kept for tree models (can split on month <= 3).
    Cyclical encodings added for linear models and LSTM.
    Year is deliberately excluded — it leaks temporal ordering.
    """
    df  = df.copy()
    idx = df.index

    df["month"]       = idx.month
    df["day_of_week"] = idx.dayofweek
    df["quarter"]     = idx.quarter

    df["is_monsoon"] = idx.month.isin([6, 7, 8, 9]).astype(int)
    df["is_winter"]  = idx.month.isin([11, 12, 1]).astype(int)
    df["is_summer"]  = idx.month.isin([3, 4, 5]).astype(int)

    df["month_sin"],       df["month_cos"]       = _cyclical_encode(
        pd.Series(idx.month,      index=idx), max_val=12
    )
    df["day_of_week_sin"], df["day_of_week_cos"] = _cyclical_encode(
        pd.Series(idx.dayofweek,  index=idx), max_val=5
    )
    return df


# ── C. Feature Column List ────────────────────────────────────────────────────

# Single source of truth for what goes into the model.
# Raw OHLCV columns (Open, High, Low, Close, Volume) are excluded —
# they are price-unit and need scaling inside the ML pipeline.
#
# Sentiment columns are always present:
#   - use_sentiment=True  → real FinBERT scores from pipeline.py
#   - use_sentiment=False → filled with 0.0 by pipeline.py
# Either way the schema is identical and model code never needs to change.

FEATURE_COLUMNS = [
    # ── Technical: trend / momentum ───────────────────────────────────────
    "ma10_ratio",
    "ma30_ratio",
    "rsi",
    # ── Technical: MACD ───────────────────────────────────────────────────
    "macd_line_ratio",
    "macd_signal_ratio",
    "macd_hist_ratio",
    # ── Technical: Bollinger Bands ────────────────────────────────────────
    "bb_width",
    "bb_pct",
    # ── Technical: return / volatility / volume ───────────────────────────
    "daily_return",
    "volatility",
    "log_volume",
    # ── Calendar / Seasonality ────────────────────────────────────────────
    "month",
    "day_of_week",
    "quarter",
    "is_monsoon",
    "is_winter",
    "is_summer",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    # ── Sentiment (FinBERT) ───────────────────────────────────────────────
    "sentiment_score",           # weighted (pos - neg),   range [-1, +1]
    "sentiment_magnitude",       # weighted (1 - neutral), range [0,  1]
    "sentiment_article_count",   # log(1 + n_articles) for that day
    "has_sentiment",
]


# ── D. Master Function ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline on a cleaned OHLCV DataFrame.

    NaN rows are NOT dropped here — that happens in pipeline.py after
    target creation so both feature NaNs and target NaN are dropped together.
    """
    df = add_technical_indicators(df)
    df = add_calendar_features(df)
    return df


# ── Entry point (standalone test) ─────────────────────────────────────────────

if __name__ == "__main__":
    from data_fetch import fetch_all_tickers
    from preprocess import clean_all

    raw     = fetch_all_tickers()
    cleaned = clean_all(raw)

    sample  = list(cleaned.keys())[0]
    df_feat = engineer_features(cleaned[sample])

    # Exclude sentiment cols from NaN check — they're added later by pipeline
    tech_cal_cols = [c for c in FEATURE_COLUMNS
                     if c not in ("sentiment_score", "sentiment_magnitude",
                                  "sentiment_article_count")]

    print(f"\nFeature columns ({len(FEATURE_COLUMNS)} total):")
    for c in FEATURE_COLUMNS:
        print(f"  {c}")

    print(f"\nSample — {sample} (last 5 rows):")
    print(df_feat[tech_cal_cols].tail(5).to_string())

    print(f"\nNaN counts (technical + calendar):")
    print(df_feat[tech_cal_cols].isna().sum().to_string())
