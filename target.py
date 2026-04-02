"""
target.py
---------
Creates the classification target (Buy / Sell / Hold) from future returns.

Leakage prevention:
  The target for row t is: (Close[t+1] - Close[t]) / Close[t]
  This is computed using .shift(-1) which looks ONE step into the future.

  This means:
  - Row t's FEATURES should only use data up to time t.  ← enforced in feature_engineering.py
  - Row t's TARGET is computed from Close[t+1].          ← that's fine; it's the label
  - The last row always has NaN target and is dropped.

  What would be a leakage bug:
  - Using Close[t+1] as an INPUT feature (we never do this).
  - Computing future returns with wrong alignment (off-by-one in shift).

Class labels:
  0 = SELL   (next-day return < -1%)
  1 = HOLD   (next-day return between -1% and +1%)
  2 = BUY    (next-day return > +1%)

The ±1% threshold is a common industry convention.
It is configurable via BUY_THRESHOLD / SELL_THRESHOLD below.
"""

import pandas as pd


# ── Config ────────────────────────────────────────────────────────────────────

BUY_THRESHOLD = 0.01    # +1% → BUY
SELL_THRESHOLD = -0.01  # -1% → SELL

# Human-readable label map (useful for confusion matrices, reports)
LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}
INT_MAP = {"SELL": 0, "HOLD": 1, "BUY": 2}


# ── Core Functions ────────────────────────────────────────────────────────────

def compute_next_day_return(df: pd.DataFrame) -> pd.Series:
    """
    Compute the next-day percentage return for each row.

    return_t = (Close[t+1] - Close[t]) / Close[t]

    The last row will be NaN because there is no t+1 for it.
    """
    return df["Close"].pct_change().shift(-1)  # shift(-1) aligns t+1 return to row t


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'next_return' (float) and 'signal' (int: 0/1/2) columns.

    Steps:
    1. Compute next-day return via shift(-1).
    2. Apply thresholds to classify into BUY / HOLD / SELL.
    3. Encode as integer labels (0 = SELL, 1 = HOLD, 2 = BUY).

    The last row will have NaN target and should be dropped
    in pipeline.py after joining features and target.
    """
    df = df.copy()
    df["next_return"] = compute_next_day_return(df)

    def _classify(ret: float) -> int:
        if pd.isna(ret):
            return pd.NA
        if ret > BUY_THRESHOLD:
            return 2  # BUY
        if ret < SELL_THRESHOLD:
            return 0  # SELL
        return 1  # HOLD

    df["signal"] = df["next_return"].map(_classify).astype("Int64")  # nullable int
    return df


def get_label_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Return the class distribution of the signal column.
    Useful to check for class imbalance before model training.
    """
    if "signal" not in df.columns:
        raise ValueError("No 'signal' column found. Run create_target() first.")
    counts = df["signal"].value_counts().sort_index()
    counts.index = [LABEL_MAP.get(i, i) for i in counts.index]
    return counts


# ── Entry point (for standalone testing) ──────────────────────────────────────

if __name__ == "__main__":
    from data_fetch import fetch_all_tickers
    from preprocess import clean_all

    raw = fetch_all_tickers()
    cleaned = clean_all(raw)

    sample_ticker = list(cleaned.keys())[0]
    df_with_target = create_target(cleaned[sample_ticker])

    print(f"\nTarget preview — {sample_ticker}:")
    print(df_with_target[["Close", "next_return", "signal"]].tail(8).to_string())

    print(f"\nClass distribution:")
    print(get_label_distribution(df_with_target))
