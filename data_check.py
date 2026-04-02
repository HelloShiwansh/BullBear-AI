"""
data_check.py
-------------
Quick pre-training validation. Run this once before building any models.
Answers three specific questions:
  1. Class distribution — per ticker and overall
  2. Feature distributions — outliers in unbounded features
  3. Sentiment coverage — how many rows have real scores vs zero

Usage:
  python data_check.py

No arguments needed. Reads from data/processed/bullbear_dataset.csv.
Takes under 60 seconds to run.
"""

import os
import pandas as pd
import numpy as np
from feature_engineering import FEATURE_COLUMNS

CSV_PATH    = "data/processed/bullbear_dataset.csv"
LABEL_MAP   = {0: "SELL", 1: "HOLD", 2: "BUY"}
SCALE_COLS  = ["daily_return", "volatility", "log_volume"]
SENT_COLS   = ["sentiment_score", "sentiment_magnitude", "sentiment_article_count"]
DIVIDER     = "─" * 60


def load() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {CSV_PATH}.\n"
            "Run pipeline.py first."
        )
    df = pd.read_csv(CSV_PATH, index_col="Date", parse_dates=True)
    print(f"Loaded: {len(df):,} rows | {df['ticker'].nunique()} tickers | "
          f"{df.index.min().date()} → {df.index.max().date()}")
    return df


# ── Check 1: Class distribution ───────────────────────────────────────────────

def check_class_distribution(df: pd.DataFrame) -> None:
    print(f"\n{DIVIDER}")
    print("CHECK 1 — Class distribution")
    print(DIVIDER)

    # Overall
    overall = df["signal"].value_counts().sort_index()
    total   = len(df)
    print("\nOverall:")
    for code, count in overall.items():
        label = LABEL_MAP[code]
        bar   = "█" * int(40 * count / total)
        print(f"  {label:<4} {count:>7,}  ({100*count/total:5.1f}%)  {bar}")

    # Per ticker
    print("\nPer ticker (HOLD %):")
    ticker_dist = (
        df.groupby("ticker")["signal"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .rename(columns=LABEL_MAP)
    )

    # Flag tickers with extreme HOLD (>75%) or extreme SELL/BUY (>35%)
    flags = []
    for ticker in ticker_dist.index:
        hold_pct  = ticker_dist.loc[ticker, "HOLD"] * 100
        sell_pct  = ticker_dist.loc[ticker, "SELL"] * 100
        buy_pct   = ticker_dist.loc[ticker, "BUY"]  * 100
        flag      = " ← WARNING: extreme HOLD" if hold_pct > 75 else ""
        flag      = flag or (" ← WARNING: extreme SELL" if sell_pct > 35 else "")
        print(f"  {ticker:<20} SELL:{sell_pct:4.1f}%  HOLD:{hold_pct:4.1f}%  BUY:{buy_pct:4.1f}%{flag}")
        if flag:
            flags.append(ticker)

    if flags:
        print(f"\n  Flagged tickers: {flags}")
        print("  Consider: class_weight='balanced' will handle this automatically.")
    else:
        print("\n  No extreme distributions detected — class_weight='balanced' still recommended.")


# ── Check 2: Feature distributions ───────────────────────────────────────────

def check_feature_distributions(df: pd.DataFrame) -> None:
    print(f"\n{DIVIDER}")
    print("CHECK 2 — Feature distributions (unbounded features)")
    print(DIVIDER)

    print("\nFeatures needing StandardScaler (daily_return, volatility, log_volume):")
    stats = df[SCALE_COLS].describe().loc[["mean", "std", "min", "max"]]
    print(stats.round(5).to_string())

    # Outlier detection — flag values beyond 5 std deviations
    print("\nOutlier check (values beyond 5 standard deviations):")
    found_outliers = False
    for col in SCALE_COLS:
        mean   = df[col].mean()
        std    = df[col].std()
        cutoff = 5 * std
        extreme = df[col][np.abs(df[col] - mean) > cutoff]
        if len(extreme) > 0:
            found_outliers = True
            print(f"  {col}: {len(extreme)} extreme rows | "
                  f"max={extreme.max():.4f} | min={extreme.min():.4f}")
            # Show which tickers they belong to
            extreme_tickers = df.loc[extreme.index, "ticker"].value_counts()
            print(f"    Tickers: {extreme_tickers.to_dict()}")
        else:
            print(f"  {col}: clean (no values beyond 5σ)")

    if not found_outliers:
        print("\n  No extreme outliers. StandardScaler will work cleanly.")
    else:
        print("\n  Outliers found. StandardScaler will compress these — that's correct behaviour.")
        print("  No action needed before training.")

    # Bounded feature sanity checks
    print("\nBounded feature sanity checks:")
    checks = {
        "rsi":        (0, 100),
        "bb_pct":     (-0.5, 1.5),   # can slightly exceed [0,1] during breakouts
        "ma10_ratio": (0.5, 2.0),
        "ma30_ratio": (0.5, 2.0),
    }
    for col, (lo, hi) in checks.items():
        out_of_range = df[(df[col] < lo) | (df[col] > hi)]
        if len(out_of_range) > 0:
            print(f"  {col}: {len(out_of_range)} rows outside [{lo}, {hi}] — check these")
        else:
            print(f"  {col}: within expected range [{lo}, {hi}] ✓")


# ── Check 3: Sentiment coverage ───────────────────────────────────────────────

def check_sentiment_coverage(df: pd.DataFrame) -> None:
    print(f"\n{DIVIDER}")
    print("CHECK 3 — Sentiment coverage")
    print(DIVIDER)

    total = len(df)

    # Overall coverage
    nonzero = (df["sentiment_score"] != 0).sum()
    zero    = total - nonzero
    pct     = 100 * nonzero / total
    print(f"\nOverall:")
    print(f"  Rows with real sentiment : {nonzero:>7,}  ({pct:.1f}%)")
    print(f"  Rows with zero (no news) : {zero:>7,}  ({100-pct:.1f}%)")

    if pct < 20:
        print(f"\n  WARNING: Only {pct:.1f}% sentiment coverage.")
        print("  The sentiment features will have limited impact on model performance.")
        print("  This is expected — Kaggle news only covers 2017–2021.")
        print("  The two-model comparison (with vs without sentiment) is especially")
        print("  important here to measure the isolated contribution of sentiment.")
    elif pct < 40:
        print(f"\n  Moderate coverage ({pct:.1f}%). Sentiment will have measurable but")
        print("  limited impact. Two-model comparison will be informative.")
    else:
        print(f"\n  Good coverage ({pct:.1f}%). Sentiment should contribute meaningfully.")

    # Coverage by year
    print("\nCoverage by year:")
    df_temp = df.copy()
    df_temp["year"]         = df_temp.index.year
    df_temp["has_sentiment"] = (df_temp["sentiment_score"] != 0).astype(int)
    yearly = df_temp.groupby("year")["has_sentiment"].agg(["sum", "count"])
    yearly["pct"] = 100 * yearly["sum"] / yearly["count"]
    for year, row in yearly.iterrows():
        bar  = "█" * int(row["pct"] / 5)
        flag = " ← Kaggle news ends here" if year == 2021 else ""
        print(f"  {year}: {row['pct']:5.1f}%  {bar}{flag}")

    # Coverage per ticker
    print("\nCoverage per ticker (% rows with real sentiment):")
    ticker_cov = (
        df.groupby("ticker")
        .apply(lambda g: 100 * (g["sentiment_score"] != 0).mean())
        .sort_values(ascending=False)
    )
    for ticker, pct in ticker_cov.items():
        bar = "█" * int(pct / 5)
        print(f"  {ticker:<20} {pct:5.1f}%  {bar}")

    # Sentiment score distribution (non-zero rows only)
    print("\nSentiment score distribution (non-zero rows only):")
    nonzero_scores = df[df["sentiment_score"] != 0]["sentiment_score"]
    print(nonzero_scores.describe().round(4).to_string())
    positive_pct = 100 * (nonzero_scores > 0).mean()
    negative_pct = 100 * (nonzero_scores < 0).mean()
    print(f"\n  Positive articles: {positive_pct:.1f}%")
    print(f"  Negative articles: {negative_pct:.1f}%")


# ── Check 4: Multicollinearity (quick) ───────────────────────────────────────

def check_multicollinearity(df: pd.DataFrame) -> None:
    print(f"\n{DIVIDER}")
    print("CHECK 4 — Multicollinearity (high correlation pairs)")
    print(DIVIDER)

    # Only check numeric feature columns present in FEATURE_COLUMNS
    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    corr      = df[feat_cols].corr().abs()

    # Find pairs with |correlation| > 0.85
    print("\nFeature pairs with |correlation| > 0.85:")
    found = False
    for i in range(len(feat_cols)):
        for j in range(i + 1, len(feat_cols)):
            val = corr.iloc[i, j]
            if val > 0.85:
                found = True
                f1, f2 = feat_cols[i], feat_cols[j]
                print(f"  {f1:<25} ←→  {f2:<25}  r={val:.3f}")

    if not found:
        print("  No pairs above 0.85 threshold.")
        print("  Note: tree models handle correlated features naturally.")
        print("  LSTM is slightly more sensitive — monitor training loss if issues arise.")
    else:
        print("\n  Note: high correlation is fine for tree models.")
        print("  For LSTM — monitor whether removing one of each pair improves validation loss.")
        print("  Do NOT remove features before first training run.")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{DIVIDER}")
    print("SUMMARY — Readiness for model training")
    print(DIVIDER)

    nonzero_pct = 100 * (df["sentiment_score"] != 0).mean()
    n_rows      = len(df)
    n_tickers   = df["ticker"].nunique()
    n_features  = len([c for c in FEATURE_COLUMNS if c in df.columns])
    date_range  = f"{df.index.min().date()} → {df.index.max().date()}"

    print(f"""
  Dataset rows       : {n_rows:,}
  Tickers            : {n_tickers}
  Features available : {n_features} / {len(FEATURE_COLUMNS)} expected
  Date range         : {date_range}
  Sentiment coverage : {nonzero_pct:.1f}%

  Next steps:
  1. Run: pip install xgboost lightgbm
  2. Build evaluate.py  (shared split + scaler + metrics)
  3. Build train_classical.py  (RF, XGBoost, LightGBM — baseline)
  4. Build train_lstm.py  (30-day window, PyTorch)
  5. Run each model TWICE: with all 24 features, then with 21 (no sentiment)
  6. Compare F1 scores across models and sentiment conditions
    """)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load()
    check_class_distribution(df)
    check_feature_distributions(df)
    check_sentiment_coverage(df)
    check_multicollinearity(df)
    print_summary(df)
