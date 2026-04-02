"""
preprocess.py
-------------
Cleans raw OHLCV DataFrames before feature engineering.

Rules enforced here:
1. Sort by Date ascending (critical for time-series correctness).
2. Drop rows where OHLCV values are fully missing.
3. Forward-fill small gaps (e.g., exchange holidays with stale prices).
4. Drop any remaining NaN rows.
5. Validate that Volume is non-negative.
6. Cast all price/volume columns to float64 for consistent dtype.

What we deliberately do NOT do here:
- No feature scaling  (that belongs in the ML pipeline to prevent leakage)
- No target creation  (handled in target.py)
- No shuffling        (never — this is time-series data)
"""

from typing import Dict

import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
# Maximum consecutive NaN rows we allow before forward-fill gives up
MAX_FILL_LIMIT = 3


# ── Core Functions ────────────────────────────────────────────────────────────

def clean_ohlcv(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    Clean a single ticker's OHLCV DataFrame.

    Steps:
    - Ensure Date index is a sorted DatetimeIndex.
    - Keep only OHLCV columns.
    - Forward-fill gaps up to MAX_FILL_LIMIT consecutive days.
    - Drop remaining rows with any NaN.
    - Cast to float64.
    - Validate non-negative volume and prices.

    Returns a cleaned copy (never mutates the input).
    """
    label = f"[{ticker}]" if ticker else ""
    df = df.copy()

    # ── 1. Index validation ────────────────────────────────────────────────
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # ── 2. Column subset ───────────────────────────────────────────────────
    missing_cols = [c for c in OHLCV_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{label} Missing columns: {missing_cols}")
    df = df[OHLCV_COLS].copy()

    # ── 3. Drop rows that are entirely NaN ────────────────────────────────
    before = len(df)
    df.dropna(how="all", inplace=True)
    dropped_all = before - len(df)
    if dropped_all:
        print(f"{label} Dropped {dropped_all} fully-NaN rows.")

    # ── 4. Forward-fill small gaps (e.g., holiday NaNs) ───────────────────
    df.ffill(limit=MAX_FILL_LIMIT, inplace=True)

    # ── 5. Drop any remaining rows with NaN ───────────────────────────────
    before = len(df)
    df.dropna(inplace=True)
    dropped_partial = before - len(df)
    if dropped_partial:
        print(f"{label} Dropped {dropped_partial} partially-NaN rows after ffill.")

    # ── 6. Cast to float64 ────────────────────────────────────────────────
    df = df.astype(float)

    # ── 7. Sanity checks ──────────────────────────────────────────────────
    if (df["Volume"] < 0).any():
        raise ValueError(f"{label} Negative Volume values detected.")
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError(f"{label} Non-positive price values detected.")

    return df


def clean_all(
    raw_data: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Run clean_ohlcv() over every ticker.
    Tickers that fail validation are dropped with a warning.
    Returns a dict of {ticker: cleaned_DataFrame}.
    """
    cleaned: Dict[str, pd.DataFrame] = {}

    for ticker, df in raw_data.items():
        try:
            cleaned[ticker] = clean_ohlcv(df, ticker=ticker)
            print(
                f"[{ticker}] Clean — {len(cleaned[ticker])} rows | "
                f"{cleaned[ticker].index.min().date()} → "
                f"{cleaned[ticker].index.max().date()}"
            )
        except Exception as e:
            print(f"[WARN] Dropping {ticker} due to error: {e}")

    print(f"\n✓ {len(cleaned)}/{len(raw_data)} tickers passed cleaning.")
    return cleaned


def log_data_quality(df: pd.DataFrame, ticker: str = "") -> None:
    """Print a quick data-quality summary for a single ticker."""
    label = f"[{ticker}]" if ticker else ""
    print(f"\n{label} Data Quality Report")
    print(f"  Rows        : {len(df)}")
    print(f"  Date range  : {df.index.min().date()} → {df.index.max().date()}")
    print(f"  NaN count   : {df.isna().sum().sum()}")
    print(f"  Dtypes      :\n{df.dtypes.to_string()}")


# ── Entry point (for standalone testing) ──────────────────────────────────────

if __name__ == "__main__":
    from data_fetch import fetch_all_tickers

    raw = fetch_all_tickers()
    cleaned = clean_all(raw)

    sample = list(cleaned.keys())[0]
    log_data_quality(cleaned[sample], ticker=sample)
    print(cleaned[sample].tail(3))
