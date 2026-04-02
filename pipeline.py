"""
pipeline.py
-----------
Orchestrates the full BullBear AI data preparation pipeline.

Execution order (must not be changed — each step depends on the previous):

  1. data_fetch  → Download / load raw OHLCV CSVs
  2. preprocess  → Clean, validate, sort
  3. feature_eng → Add technical + calendar features
  4. target      → Add next-day return + Buy/Sell/Hold label
  5. sentiment   → Join daily FinBERT sentiment scores (optional)
  6. assemble    → Stack tickers, drop NaN rows, validate, save

Output:
  data/processed/bullbear_dataset.csv
  Columns: [FEATURE_COLUMNS] + signal + ticker + Date (index)

Usage:
  # Without sentiment (price + calendar features only)
  python pipeline.py

  # With sentiment (requires data/news/india_financial_news.csv)
  # Change the run() call at the bottom of this file to:
  #   df = run(use_sentiment=True)
  python pipeline.py

Loading in model training scripts:
  from pipeline import load_dataset
  from feature_engineering import FEATURE_COLUMNS
  df = load_dataset()
  X  = df[FEATURE_COLUMNS]
  y  = df["signal"]
"""

import os
import pandas as pd

from data_fetch import fetch_all_tickers, NIFTY_TOP20
from preprocess import clean_all
from feature_engineering import engineer_features, FEATURE_COLUMNS
from target import create_target, get_label_distribution
from sentiment import build_sentiment_features, SENTIMENT_COLUMNS

OUTPUT_DIR  = "data/processed"
OUTPUT_FILE = "bullbear_dataset.csv"


# ── Per-ticker assembly ───────────────────────────────────────────────────────

def _assemble_ticker(
    df: pd.DataFrame,
    ticker: str,
    sentiment_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    For a single ticker:
    1. Engineer all technical + calendar features.
    2. Create target via shift(-1) — no leakage.
    3. Join daily sentiment scores if provided.
       - Left join on Date so every price row is preserved.
       - Days with no news → sentiment columns filled with 0.0 (neutral).
    4. Drop rows where any feature OR target is NaN (warm-up rows + last row).
    5. Add ticker column.
    """
    df = engineer_features(df)
    df = create_target(df)

    # ── Sentiment join ─────────────────────────────────────────────────────
    if sentiment_df is not None:
        ticker_sent = (
            sentiment_df[sentiment_df["ticker"] == ticker]
            .set_index("Date")[SENTIMENT_COLUMNS]
        )
        ticker_sent.index = pd.to_datetime(ticker_sent.index)
        df = df.join(ticker_sent, how="left")
        df[SENTIMENT_COLUMNS] = df[SENTIMENT_COLUMNS].fillna(0.0)
    else:
        # Sentiment not used — add zero columns to keep schema consistent
        for col in SENTIMENT_COLUMNS:
            df[col] = 0.0

    df["has_sentiment"] = (df["sentiment_score"] != 0).astype(int)

    # ── Drop NaN rows ──────────────────────────────────────────────────────
    df = df.dropna(subset=FEATURE_COLUMNS + ["signal"])
    df["ticker"] = ticker
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(
    tickers: list[str] = NIFTY_TOP20,
    save: bool = True,
    use_sentiment: bool = False,
) -> pd.DataFrame:
    """
    Run the full pipeline and return the ML-ready dataset.

    Parameters
    ----------
    tickers       : list of NSE ticker symbols (default: top 20 NIFTY)
    save          : save final CSV to data/processed/bullbear_dataset.csv
    use_sentiment : join FinBERT sentiment features (requires Kaggle news CSV)
                    see sentiment.py for setup instructions

    Returns
    -------
    pd.DataFrame — all features + signal + ticker, Date as index
    """
    print("=" * 60)
    print("BullBear AI — Data Preparation Pipeline")
    print("=" * 60)

    # ── 1. Fetch ───────────────────────────────────────────────────────────
    print("\n[1/5] Fetching raw OHLCV data...")
    raw_data = fetch_all_tickers(tickers=tickers, save=True)

    # ── 2. Clean ───────────────────────────────────────────────────────────
    print("\n[2/5] Cleaning & validating...")
    cleaned_data = clean_all(raw_data)

    # ── 3. Sentiment (optional) ────────────────────────────────────────────
    sentiment_df = None
    if use_sentiment:
        print("\n[3/5] Building sentiment features (FinBERT)...")
        try:
            sentiment_df = build_sentiment_features(tickers=tickers)
            print(f"  Sentiment ready: {len(sentiment_df):,} (date, ticker) rows")
        except FileNotFoundError as e:
            print(f"\n  [WARN] Sentiment skipped — {e}")
            print("  Download the Kaggle CSV and place at data/news/india_financial_news.csv")
    else:
        print("\n[3/5] Sentiment skipped (use_sentiment=False).")

    # ── 4 & 5. Feature engineering + target + assemble ────────────────────
    print("\n[4/5] Engineering features, creating targets, joining sentiment...")
    all_frames = []

    for ticker, df in cleaned_data.items():
        try:
            df_processed = _assemble_ticker(df, ticker, sentiment_df)
            all_frames.append(df_processed)
            print(f"  [{ticker}] → {len(df_processed):,} rows")
        except Exception as e:
            print(f"  [WARN] Skipping {ticker}: {e}")

    if not all_frames:
        raise RuntimeError("No tickers survived the pipeline. Check raw data.")

    # ── 6. Final assembly ──────────────────────────────────────────────────
    print("\n[5/5] Assembling final dataset...")
    final_df = pd.concat(all_frames, axis=0).sort_index()

    # Validation — zero NaNs must pass before saving
    assert final_df[FEATURE_COLUMNS].isna().sum().sum() == 0, \
        "NaNs remain in features — check pipeline!"
    assert final_df["signal"].isna().sum() == 0, \
        "NaNs remain in target — check pipeline!"

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✓ Pipeline complete.")
    print(f"  Total rows      : {len(final_df):,}")
    print(f"  Tickers         : {final_df['ticker'].nunique()}")
    print(f"  Feature columns : {len(FEATURE_COLUMNS)}")
    print(f"  Sentiment active: {use_sentiment and sentiment_df is not None}")
    print(f"  Date range      : {final_df.index.min().date()} → {final_df.index.max().date()}")

    print("\n  Signal distribution:")
    dist  = get_label_distribution(final_df)
    total = dist.sum()
    for label, count in dist.items():
        print(f"    {label:>4} : {count:>7,}  ({100*count/total:.1f}%)")

    # ── Save ───────────────────────────────────────────────────────────────
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
        final_df.to_csv(out_path)
        print(f"\n  Saved → {out_path}")

    return final_df


# ── Load helper ───────────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    """
    Load the pre-built dataset from disk.
    Use this in train_classical.py, train_lstm.py, evaluate.py.
    Skips the full pipeline — fast.
    """
    path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Run pipeline.py first."
        )
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    print(f"Loaded: {len(df):,} rows | {df['ticker'].nunique()} tickers | "
          f"{df.index.min().date()} → {df.index.max().date()}")
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Change use_sentiment=True when Kaggle CSV is in data/news/ ──
    df = run(use_sentiment=True)

    print("\n── First 5 rows ──")
    print(df[FEATURE_COLUMNS + ["signal", "ticker"]].head(5).to_string())

    print("\n── Last 5 rows ──")
    print(df[FEATURE_COLUMNS + ["signal", "ticker"]].tail(5).to_string())

    print("\n── Feature dtypes ──")
    print(df[FEATURE_COLUMNS].dtypes.to_string())
