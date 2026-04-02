"""
sentiment.py
------------
Builds daily sentiment features for each (Date, ticker) pair using:
  - Source      : Kaggle "India financial news headlines sentiments" dataset
                  kaggle.com/datasets/harshrkh/india-financial-news-headlines-sentiments
  - Scoring     : FinBERT (ProsusAI/finbert) — finance-specific transformer
  - Aggregation : Recency-weighted by publish time relative to NSE market hours

How to get the Kaggle dataset:
  1. Go to: kaggle.com/datasets/harshrkh/india-financial-news-headlines-sentiments
  2. Click Download → extract CSV
  3. Place it at: data/news/india_financial_news.csv
  4. Run: python sentiment.py   (builds + caches scored CSV)

Kaggle CSV format (confirmed):
  Columns   : Date, Title, URL, sentiment, confidence
  Date format: DD/MM/YY  e.g. "05/01/17" = January 5, 2017
  Coverage  : Jan 2017 to Apr 2021

Output:
  data/news/sentiment_daily.csv — one row per (Date, ticker)
  Columns: sentiment_score, sentiment_magnitude, sentiment_article_count

Design decisions:
  - FinBERT is run once and results cached — re-runs load from cache instantly.
  - sentiment_score     = positive_prob - negative_prob  range [-1, +1]
  - sentiment_magnitude = 1 - neutral_prob               signal strength
  - The pre-labelled sentiment/confidence columns in the CSV are IGNORED.
    FinBERT produces better, consistent scores so we re-score everything.
  - Articles after NSE close (15:30 IST) shift to the NEXT trading day.
  - Date-only rows (no time info) treated as pre-market — full weight, same day.
  - Missing days/tickers filled with 0 (neutral) in pipeline.py.
"""

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────

NEWS_DIR          = "data/news"
RAW_NEWS_PATH     = os.path.join(NEWS_DIR, "india_financial_news.csv")
SCORED_CACHE_PATH = os.path.join(NEWS_DIR, "finbert_scored.csv")
DAILY_OUTPUT_PATH = os.path.join(NEWS_DIR, "sentiment_daily.csv")

# ── NSE market hours (IST = UTC+5:30) ─────────────────────────────────────────

NSE_OPEN_HOUR  = 9
NSE_OPEN_MIN   = 15
NSE_CLOSE_HOUR = 15
NSE_CLOSE_MIN  = 30

# ── FinBERT config ─────────────────────────────────────────────────────────────

FINBERT_MODEL = "ProsusAI/finbert"
BATCH_SIZE    = 32   # reduce to 16 if you hit memory errors on CPU
MAX_LENGTH    = 512

# ── Ticker keyword map ─────────────────────────────────────────────────────────
# Each ticker maps to a list of strings matched case-insensitively in headlines.
# Multiple keywords handle abbreviations and name variants.

# TICKER_KEYWORDS: dict[str, list[str]] = {
#     "RELIANCE.NS":   ["reliance", "ril", "mukesh ambani", "jio"],
#     "TCS.NS":        ["tcs", "tata consultancy", "tata consulting"],
#     "HDFCBANK.NS":   ["hdfc bank", "hdfcbank", "hdfc"],
#     "INFY.NS":       ["infosys", "infy"],
#     "ICICIBANK.NS":  ["icici bank", "icicibank", "icici"],
#     "HINDUNILVR.NS": ["hindustan unilever", "hul", "unilever india"],
#     "ITC.NS":        ["itc limited", "itc ltd", " itc "],
#     "SBIN.NS":       ["sbi", "state bank of india", "state bank india"],
#     "BHARTIARTL.NS": ["airtel", "bharti airtel", "bharti"],
#     "KOTAKBANK.NS":  ["kotak mahindra", "kotak bank", "kotak"],
#     "LT.NS":         ["larsen", "toubro", "l&t", "l & t"],
#     "AXISBANK.NS":   ["axis bank", "axisbank"],
#     "BAJFINANCE.NS": ["bajaj finance", "bajaj fin"],
#     "ASIANPAINT.NS": ["asian paint", "asian paints"],
#     "MARUTI.NS":     ["maruti", "maruti suzuki", "msil"],
#     "TITAN.NS":      ["titan company", "titan watch", "tanishq"],
#     "WIPRO.NS":      ["wipro"],
#     "HCLTECH.NS":    ["hcl tech", "hcltech", "hcl technologies"],
#     "SUNPHARMA.NS":  ["sun pharma", "sunpharma", "sun pharmaceutical"],
#     "ULTRACEMCO.NS": ["ultratech cement", "ultratech", "ultracemco"],
# }

TICKER_KEYWORDS: dict[str, list[str]] = {
    "RELIANCE.NS":   ["reliance industries", "reliance jio", "mukesh ambani",
                      " ril ", "ril's", "(ril)"],
    "TCS.NS":        ["tata consultancy", " tcs ", "tcs's", "(tcs)"],
    "HDFCBANK.NS":   ["hdfc bank", "hdfcbank"],
    "INFY.NS":       ["infosys", " infy "],
    "ICICIBANK.NS":  ["icici bank", "icicibank"],
    "HINDUNILVR.NS": ["hindustan unilever", " hul ", "hul's", "(hul)"],
    "ITC.NS":        ["itc limited", "itc ltd", " itc ", "itc's"],
    "SBIN.NS":       ["state bank of india", " sbi ", "sbi's", "(sbi)"],
    "BHARTIARTL.NS": ["bharti airtel", "airtel"],
    "KOTAKBANK.NS":  ["kotak mahindra", "kotak bank"],
    "LT.NS":         ["larsen & toubro", "larsen and toubro", " l&t ", " l & t "],
    "AXISBANK.NS":   ["axis bank"],
    "BAJFINANCE.NS": ["bajaj finance"],
    "ASIANPAINT.NS": ["asian paints", "asian paint"],
    "MARUTI.NS":     ["maruti suzuki", "maruti"],
    "TITAN.NS":      ["titan company", "titan industries", "tanishq",
                      "titan watch", " titan "],
    "WIPRO.NS":      ["wipro"],
    "HCLTECH.NS":    ["hcl technologies", "hcl tech", "hcltech"],
    "SUNPHARMA.NS":  ["sun pharma", "sun pharmaceutical", "sunpharma"],
    "ULTRACEMCO.NS": ["ultratech cement", "ultratech"],
}
# ── A. Data Loading ───────────────────────────────────────────────────────────

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Kaggle CSV columns → internal schema.

    Kaggle columns : Date, Title, URL, sentiment, confidence
    Internal schema: published_at, headline
    """
    col_map    = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}

    for candidate in ["date", "published_at", "publish_date",
                       "published", "time", "datetime", "timestamp"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "published_at"
            break

    for candidate in ["title", "headline", "news", "text",
                       "article", "description", "content"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "headline"
            break

    df = df.rename(columns=col_map)

    if "published_at" not in df.columns:
        raise ValueError(
            f"No date column found. Columns: {list(df.columns)}"
        )
    if "headline" not in df.columns:
        raise ValueError(
            f"No text column found. Columns: {list(df.columns)}"
        )
    return df


def load_news_data(path: str = RAW_NEWS_PATH) -> pd.DataFrame:
    """
    Load and clean the Kaggle news CSV.

    Key fix: the Kaggle CSV uses DD/MM/YY dates ("05/01/17" = Jan 5 2017).
    Standard pandas auto-detection would misread this as May 1 2017.
    We force format="%d/%m/%y" to get the correct dates.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nNews CSV not found at: {path}\n"
            "Download from:\n"
            "  kaggle.com/datasets/harshrkh/india-financial-news-headlines-sentiments\n"
            "Place the CSV at: data/news/india_financial_news.csv"
        )

    print(f"[NEWS] Loading {path}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"[NEWS] Raw rows  : {len(df):,}")
    print(f"[NEWS] Columns   : {list(df.columns)}")

    df = _normalise_columns(df)

    # ── Parse DD/MM/YY dates ──────────────────────────────────────────────
    df["published_at"] = pd.to_datetime(
        df["published_at"],
        format="%d/%m/%y",
        errors="coerce",
    )

    # Sanity check — if >50% nulls the format guess was wrong
    null_pct = df["published_at"].isna().mean()
    if null_pct > 0.5:
        print(f"[NEWS] Warning: {null_pct:.0%} dates null with DD/MM/YY — retrying with dayfirst")
        raw_dates = pd.read_csv(path, low_memory=False).pipe(_normalise_columns)["published_at"]
        df["published_at"] = pd.to_datetime(raw_dates, dayfirst=True, errors="coerce")

    # ── Clean ─────────────────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["published_at", "headline"])
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df[df["headline"].str.len() > 10]
    df = df.sort_values("published_at").reset_index(drop=True)

    print(f"[NEWS] Clean rows: {len(df):,} (dropped {before - len(df):,})")
    print(f"[NEWS] Date range: {df['published_at'].min().date()} → {df['published_at'].max().date()}")
    return df


# ── B. Ticker Filtering ───────────────────────────────────────────────────────

def filter_relevant_articles(
    df: pd.DataFrame,
    tickers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Keep only articles mentioning at least one of our 20 tickers.
    Adds 'ticker' column. Articles matching multiple tickers are duplicated
    — each stock gets its own row with the same headline.
    """
    if tickers is None:
        tickers = list(TICKER_KEYWORDS.keys())

    print(f"\n[NEWS] Matching headlines to {len(tickers)} tickers...")
    matched_rows = []

    for ticker in tickers:
        keywords = TICKER_KEYWORDS.get(ticker, [])
        if not keywords:
            continue
        mask    = df["headline"].str.lower().apply(
            lambda h: any(kw in h for kw in keywords)
        )
        matched = df[mask].copy()
        matched["ticker"] = ticker
        matched_rows.append(matched)
        print(f"  [{ticker:<20}] {mask.sum():>5,} articles")

    if not matched_rows:
        raise ValueError(
            "Zero articles matched any ticker keyword.\n"
            "Run the standalone diagnostic to debug:\n"
            "  python -c \"from sentiment import load_news_data; df=load_news_data(); print(df.head())\""
        )

    result = pd.concat(matched_rows, ignore_index=True)
    print(f"\n[NEWS] Total matched rows: {len(result):,}")
    return result


# ── C. FinBERT Scoring ────────────────────────────────────────────────────────

def _load_finbert():
    """Load FinBERT tokenizer + model. Downloads ~400MB on first call."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except ImportError:
        raise ImportError("Run: pip install transformers torch")

    print(f"\n[FINBERT] Loading '{FINBERT_MODEL}'...")
    print("  First run: downloads ~400MB. All runs after: loads from local cache.")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.eval()
    return tokenizer, model


def _score_batch(texts: list[str], tokenizer, model) -> list[dict]:
    """
    Score one batch with FinBERT.
    ProsusAI/finbert label order: positive=0, negative=1, neutral=2
    """
    import torch
    inputs = tokenizer(
        texts, padding=True, truncation=True,
        max_length=MAX_LENGTH, return_tensors="pt",
    )
    with torch.no_grad():
        probs = torch.nn.functional.softmax(
            model(**inputs).logits, dim=-1
        ).numpy()

    return [
        {"fin_positive": float(r[0]),
         "fin_negative": float(r[1]),
         "fin_neutral":  float(r[2])}
        for r in probs
    ]


def score_with_finbert(
    df: pd.DataFrame,
    cache_path: str = SCORED_CACHE_PATH,
) -> pd.DataFrame:
    """
    Score all headlines with FinBERT. Caches to CSV on first run.
    Re-running this function loads from cache — no model inference needed.
    """
    if os.path.exists(cache_path):
        print(f"\n[FINBERT] Cache hit — loading from {cache_path}")
        scored = pd.read_csv(cache_path)
        scored["published_at"] = pd.to_datetime(scored["published_at"])
        print(f"[FINBERT] {len(scored):,} rows loaded from cache.")
        return scored

    print(f"\n[FINBERT] Scoring {len(df):,} headlines (CPU, ~10-15 min, runs once)...")
    tokenizer, model = _load_finbert()
    texts      = df["headline"].tolist()
    all_scores = []

    for i in range(0, len(texts), BATCH_SIZE):
        all_scores.extend(_score_batch(texts[i:i + BATCH_SIZE], tokenizer, model))
        if i % (BATCH_SIZE * 10) == 0:
            print(f"  {100*i/len(texts):5.1f}%  ({i:,} / {len(texts):,})")

    print(f"  100.0%  ({len(texts):,} / {len(texts):,}) — done.")

    result = pd.concat([df.reset_index(drop=True), pd.DataFrame(all_scores)], axis=1)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    result.to_csv(cache_path, index=False)
    print(f"[FINBERT] Cached → {cache_path}")
    return result


# ── D. Trading Date Assignment ────────────────────────────────────────────────

def _get_trading_date(published_at: pd.Timestamp) -> pd.Timestamp:
    """
    Map article timestamp to correct trading day.
    Post-close (> 15:30 IST) → next calendar day.
    Date-only rows (hour=0) → same day (pre-market assumption).
    """
    try:
        ist = (published_at.tz_convert("Asia/Kolkata")
               if published_at.tzinfo else published_at)
        if ist.hour == 0 and ist.minute == 0:
            return ist.normalize()  # date-only → same day
        after_close = (ist.hour > NSE_CLOSE_HOUR) or (
            ist.hour == NSE_CLOSE_HOUR and ist.minute >= NSE_CLOSE_MIN
        )
        return (ist + pd.Timedelta(days=1)).normalize() if after_close else ist.normalize()
    except Exception:
        return pd.Timestamp(published_at.date())


def _recency_weight(published_at: pd.Timestamp) -> float:
    """
    Exponential decay from market open: exp(-hours_since_open / 4).
    Date-only rows (hour=0) → weight = 1.0 (pre-market, full influence).
    """
    try:
        ist = (published_at.tz_convert("Asia/Kolkata")
               if published_at.tzinfo else published_at)
        if ist.hour == 0 and ist.minute == 0:
            return 1.0
        minutes_after_open = max(
            0, ist.hour * 60 + ist.minute - (NSE_OPEN_HOUR * 60 + NSE_OPEN_MIN)
        )
        return float(np.exp(-minutes_after_open / 240))  # 240 min = 4 hours
    except Exception:
        return 1.0


# ── E. Daily Aggregation ──────────────────────────────────────────────────────

def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse article rows into one row per (trading_date, ticker).

    Features:
      sentiment_score         — recency-weighted mean(pos - neg), range [-1, +1]
      sentiment_magnitude     — recency-weighted mean(1 - neutral), range [0, 1]
      sentiment_article_count — log(1 + n_articles) — coverage volume
    """
    print("\n[SENTIMENT] Assigning trading dates and weights...")
    df = df.copy()
    df["sentiment_score"]     = df["fin_positive"] - df["fin_negative"]
    df["sentiment_magnitude"] = 1.0 - df["fin_neutral"]
    df["trading_date"]        = df["published_at"].apply(_get_trading_date)
    df["recency_weight"]      = df["published_at"].apply(_recency_weight)

    print("[SENTIMENT] Aggregating to (date, ticker) level...")

    def _agg(group):
        w = group["recency_weight"].values
        if w.sum() == 0:
            return pd.Series({
                "sentiment_score": 0.0,
                "sentiment_magnitude": 0.0,
                "sentiment_article_count": 0.0,
            })
        return pd.Series({
            "sentiment_score":         round(float(np.average(group["sentiment_score"],     weights=w)), 6),
            "sentiment_magnitude":     round(float(np.average(group["sentiment_magnitude"], weights=w)), 6),
            "sentiment_article_count": round(float(np.log1p(len(group))),                                6),
        })

    daily = (
        df.groupby(["trading_date", "ticker"])
          .apply(_agg, include_groups=False)
          .reset_index()
          .rename(columns={"trading_date": "Date"})
    )
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.sort_values(["Date", "ticker"]).reset_index(drop=True)

    print(f"[SENTIMENT] {len(daily):,} rows | "
          f"{daily['Date'].min().date()} → {daily['Date'].max().date()}")
    return daily


# ── F. Master Functions ───────────────────────────────────────────────────────

def build_sentiment_features(
    news_path: str = RAW_NEWS_PATH,
    tickers: Optional[list[str]] = None,
    force_rescore: bool = False,
) -> pd.DataFrame:
    """
    Full sentiment pipeline. Called by pipeline.py when use_sentiment=True.

    On first run : loads CSV → filters → scores with FinBERT → aggregates → caches
    On re-runs   : loads from data/news/sentiment_daily.csv instantly

    Parameters
    ----------
    news_path     : path to Kaggle CSV
    tickers       : defaults to all 20 NIFTY tickers
    force_rescore : delete caches and re-run FinBERT from scratch
    """
    os.makedirs(NEWS_DIR, exist_ok=True)

    if force_rescore:
        for p in [SCORED_CACHE_PATH, DAILY_OUTPUT_PATH]:
            if os.path.exists(p):
                os.remove(p)
        print("[SENTIMENT] Caches cleared.")

    if os.path.exists(DAILY_OUTPUT_PATH):
        print(f"[SENTIMENT] Loading cached output from {DAILY_OUTPUT_PATH}")
        daily = pd.read_csv(DAILY_OUTPUT_PATH, parse_dates=["Date"])
        print(f"[SENTIMENT] {len(daily):,} rows.")
        return daily

    raw      = load_news_data(news_path)
    filtered = filter_relevant_articles(raw, tickers)
    scored   = score_with_finbert(filtered)
    daily    = aggregate_daily_sentiment(scored)

    daily.to_csv(DAILY_OUTPUT_PATH, index=False)
    print(f"\n[SENTIMENT] Saved → {DAILY_OUTPUT_PATH}")
    return daily


def load_sentiment_features() -> pd.DataFrame:
    """Load pre-built daily sentiment. Use in model training scripts."""
    if not os.path.exists(DAILY_OUTPUT_PATH):
        raise FileNotFoundError(
            f"No cache at {DAILY_OUTPUT_PATH}.\n"
            "Run: python pipeline.py (with use_sentiment=True)"
        )
    df = pd.read_csv(DAILY_OUTPUT_PATH, parse_dates=["Date"])
    print(f"[SENTIMENT] {len(df):,} rows loaded.")
    return df


# ── Exported column list ───────────────────────────────────────────────────────

SENTIMENT_COLUMNS = [
    "sentiment_score",
    "sentiment_magnitude",
    "sentiment_article_count",
]


# ── Standalone diagnostic ──────────────────────────────────────────────────────

if __name__ == "__main__":
    daily = build_sentiment_features()
    print("\n── First 5 rows ──")
    print(daily.head(5).to_string())
    print("\n── sentiment_score stats ──")
    print(daily["sentiment_score"].describe().round(4))
    print("\n── Coverage per ticker ──")
    print(daily.groupby("ticker")["Date"].count().sort_values(ascending=False).to_string())
