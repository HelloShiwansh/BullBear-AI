"""
data_fetch.py
-------------
Downloads daily OHLCV data for the top 20 NIFTY stocks using yfinance.
Raw data is stored as individual CSVs under data/raw/.

Design notes:
- Each ticker gets its own CSV so partial re-runs are cheap.
- We only pull the last 5 years to stay time-bound and reduce noise.
- The .NS suffix is required by Yahoo Finance for NSE-listed stocks.
"""

import os
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────

# Top 20 NIFTY 50 constituents (as of 2024) with NSE suffix
NIFTY_TOP20 = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "BAJFINANCE.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "TITAN.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "SUNPHARMA.NS",
    "ULTRACEMCO.NS",
]

RAW_DATA_DIR = "data/raw"
LOOKBACK_YEARS = 9


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_date_range() -> tuple[str, str]:
    """Return (start_date, end_date) strings for the last N years."""
    end = datetime.today()
    start = end - timedelta(days=LOOKBACK_YEARS * 365)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance sometimes returns MultiIndex columns when auto_adjust=True.
    Flatten them and keep only OHLCV.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df


# ── Core Functions ────────────────────────────────────────────────────────────

def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker.
    Returns a cleaned DataFrame indexed by Date.
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker symbol.")
    df = _clean_columns(raw)
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def fetch_all_tickers(
    tickers: list[str] = NIFTY_TOP20,
    save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for all tickers.
    Saves each ticker's raw data to data/raw/<TICKER>.csv if save=True.
    Returns a dict of {ticker: DataFrame}.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    start, end = _get_date_range()

    data: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        csv_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")

        # Skip network call if cached CSV already exists
        if os.path.exists(csv_path):
            print(f"[CACHE] Loading {ticker} from {csv_path}")
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        else:
            print(f"[FETCH] Downloading {ticker} ({start} → {end})")
            try:
                df = fetch_ticker(ticker, start, end)
                if save:
                    df.to_csv(csv_path)
            except ValueError as e:
                print(f"[WARN] Skipping {ticker}: {e}")
                continue

        data[ticker] = df

    print(f"\n✓ Loaded {len(data)} tickers.")
    return data


# ── Entry point (for standalone testing) ──────────────────────────────────────

if __name__ == "__main__":
    data = fetch_all_tickers()
    sample_ticker = list(data.keys())[0]
    print(f"\nSample — {sample_ticker}:")
    print(data[sample_ticker].tail(3))
