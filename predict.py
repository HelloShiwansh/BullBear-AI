"""
predict.py — BullBear AI Live Prediction Engine
================================================
Loads the production Random Forest model and generates Buy/Hold/Sell
signals for all 20 NIFTY tickers using live yfinance data.

Usage (standalone):
    python predict.py                  # prints signals for all 20 tickers
    python predict.py --ticker TCS.NS  # single ticker

Usage (as module):
    from predict import predict_all, load_model
    signals, histories = predict_all(history_days=30)
"""

import argparse
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Import feature engineering (must be run from project root) ───────────────
from feature_engineering import engineer_features, FEATURE_COLUMNS

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH = Path("data/models/random_forest_without_sentiment.pkl")
SCALE_COLS = ["daily_return", "volatility", "log_volume"]

SIGNAL_MAP   = {0: "SELL", 1: "HOLD", 2: "BUY"}
SIGNAL_EMOJI = {0: "🔴", 1: "⚪", 2: "🟢"}
SIGNAL_COLOR = {0: "#DC2626", 1: "#64748B", 2: "#16A34A"}

# Minimum rows needed: MACD warmup (35) + buffer (25)
MIN_ROWS_REQUIRED = 60
# Days to fetch from yfinance (calendar days — accounts for weekends/holidays)
FETCH_CALENDAR_DAYS = 180

NIFTY_TOP20 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "TITAN.NS", "WIPRO.NS", "HCLTECH.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS",
]


# ── Model loader ─────────────────────────────────────────────────────────────
def load_model() -> tuple:
    """
    Load the production Random Forest model from disk.
    Returns: (model, feature_cols, scaler)
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model not found at {MODEL_PATH}. "
            "Run train_classical.py first."
        )
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["feature_cols"], data["scaler"]


# ── Raw data fetcher ─────────────────────────────────────────────────────────
def _fetch_ohlcv(ticker: str) -> pd.DataFrame | None:
    """
    Download OHLCV from yfinance and normalise column names.
    Returns None if data is insufficient.
    """
    end   = datetime.today()
    start = end - timedelta(days=FETCH_CALENDAR_DAYS)

    def _download() -> pd.DataFrame:
        return yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

    raw: pd.DataFrame | None = None
    # yfinance occasionally throws internal errors for a single ticker; retry + fallback.
    try:
        raw = _download()
    except Exception:
        raw = None

    if raw is None or raw.empty:
        try:
            # Fallback path tends to be more reliable for intermittent download issues.
            raw = yf.Ticker(ticker).history(period="9mo", interval="1d", auto_adjust=True)
        except Exception:
            raw = None

    if raw is None or raw.empty:
        return None

    # Flatten MultiIndex columns produced by some yfinance versions
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # history() sometimes uses lowercase names; normalise for rename-map below.
    raw_cols = {c: c.title() for c in raw.columns}
    raw = raw.rename(columns=raw_cols)

    raw = (
        raw.reset_index()
        .rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
    )
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").reset_index(drop=True)

    # Keep only the columns we need
    raw = raw[["date", "open", "high", "low", "close", "volume"]].copy()
    raw[["open", "high", "low", "close"]] = raw[["open", "high", "low", "close"]].astype(float)
    raw["volume"] = raw["volume"].astype(float)

    if len(raw) < MIN_ROWS_REQUIRED:
        return None

    return raw


# ── Single ticker prediction ─────────────────────────────────────────────────
def fetch_and_predict(
    ticker: str,
    model,
    feature_cols: list,
    scaler,
    history_days: int = 30,
) -> tuple[dict | None, pd.DataFrame | None]:
    """
    Fetch live OHLCV for one ticker, engineer features, run the model.

    Returns
    -------
    today_pred : dict
        Signal, probabilities, confidence, and latest feature values.
    history_df : pd.DataFrame
        Last `history_days` rows with columns: date, open, high, low, close,
        volume, signal, signal_name, confidence, prob_buy, prob_hold, prob_sell.
    Both return None on failure.
    """
    # ── 1. Fetch raw OHLCV ──────────────────────────────────────────────────
    raw = _fetch_ohlcv(ticker)
    if raw is None:
        return None, None

    # ── 2. Match training schema for feature engineering ────────────────────
    # feature_engineering expects DateTime index + uppercase OHLCV columns.
    fe_df = raw.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    ).copy()
    fe_df["Date"] = pd.to_datetime(fe_df["Date"])
    fe_df = fe_df.set_index("Date").sort_index()

    # Add sentinel sentiment cols (zero-fill — no live news source)
    # engineer_features does NOT add sentiment cols; those come from pipeline.
    for col in ["sentiment_score", "sentiment_magnitude", "sentiment_article_count"]:
        fe_df[col] = 0.0

    # ── 3. Feature engineering ───────────────────────────────────────────────
    df = engineer_features(fe_df)

    # has_sentiment is derived on-the-fly (not in CSV, per evaluate.py design)
    df["has_sentiment"] = 0

    # Convert OHLCV back to dashboard-friendly lowercase schema.
    df = df.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    # Drop NaN rows from MACD/BB warmup
    valid_feature_cols = [c for c in feature_cols if c in df.columns]
    df = df.dropna(subset=valid_feature_cols).reset_index(drop=True)

    if df.empty or len(df) < 2:
        return None, None

    # ── 4. Scale the 3 unbounded features ───────────────────────────────────
    X = df[feature_cols].copy()
    X[SCALE_COLS] = scaler.transform(X[SCALE_COLS])

    # ── 5. Predict all rows (needed for history) ─────────────────────────────
    preds = model.predict(X)
    probs = model.predict_proba(X)   # shape (n, 3) — columns: [SELL, HOLD, BUY]

    df = df.copy()
    df["signal"]      = preds
    df["signal_name"] = df["signal"].map(SIGNAL_MAP)
    df["prob_sell"]   = probs[:, 0]
    df["prob_hold"]   = probs[:, 1]
    df["prob_buy"]    = probs[:, 2]
    df["confidence"]  = probs.max(axis=1)

    # ── 6. Build today's prediction dict ─────────────────────────────────────
    last = df.iloc[-1]
    today_pred = {
        "ticker":       ticker,
        "ticker_short": ticker.replace(".NS", ""),
        "date":         pd.Timestamp(last["date"]).to_pydatetime(),
        "signal":       int(last["signal"]),
        "signal_name":  str(last["signal_name"]),
        "confidence":   float(last["confidence"]),
        "prob_buy":     float(last["prob_buy"]),
        "prob_hold":    float(last["prob_hold"]),
        "prob_sell":    float(last["prob_sell"]),
        "close":        float(last["close"]),
        "open":         float(last["open"]),
        "high":         float(last["high"]),
        "low":          float(last["low"]),
        "volume":       float(last["volume"]),
        "features":     {k: float(last[k]) for k in feature_cols if k in df.columns},
    }

    # ── 7. History slice ─────────────────────────────────────────────────────
    history_cols = [
        "date", "open", "high", "low", "close", "volume",
        "signal", "signal_name", "confidence",
        "prob_buy", "prob_hold", "prob_sell",
    ]
    history_df = df[history_cols].tail(history_days).copy().reset_index(drop=True)

    return today_pred, history_df


# ── Predict all 20 tickers ───────────────────────────────────────────────────
def predict_all(history_days: int = 30) -> tuple[list, dict]:
    """
    Run predictions for all 20 NIFTY tickers.

    Returns
    -------
    signals   : list of today_pred dicts (one per ticker, failures excluded)
    histories : dict of {ticker: history_df}
    """
    model, feature_cols, scaler = load_model()
    signals:   list = []
    histories: dict = {}

    for ticker in NIFTY_TOP20:
        try:
            pred, hist = fetch_and_predict(
                ticker, model, feature_cols, scaler, history_days
            )
            if pred is not None:
                signals.append(pred)
                histories[ticker] = hist
            else:
                print(f"[WARN] {ticker}: insufficient data or feature NaN — skipped")
        except Exception as exc:
            print(f"[ERROR] {ticker}: {exc}")

    return signals, histories


# ── Standalone CLI ───────────────────────────────────────────────────────────
def _cli() -> None:
    parser = argparse.ArgumentParser(description="BullBear AI — live signal printer")
    parser.add_argument("--ticker", default=None, help="Single ticker (e.g. TCS.NS)")
    args = parser.parse_args()

    model, feature_cols, scaler = load_model()
    tickers = [args.ticker] if args.ticker else NIFTY_TOP20

    print(f"\n{'─'*60}")
    print(f"  BullBear AI — Live Signals  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    print(f"{'─'*60}")
    print(f"  {'Ticker':<18} {'Signal':<8} {'Conf':>6}  {'BUY':>6}  {'SELL':>6}  {'Close':>10}")
    print(f"{'─'*60}")

    for ticker in tickers:
        pred, _ = fetch_and_predict(ticker, model, feature_cols, scaler)
        if pred is None:
            print(f"  {ticker:<18} {'ERROR':<8}")
            continue
        emoji = SIGNAL_EMOJI[pred["signal"]]
        print(
            f"  {pred['ticker_short']:<18} "
            f"{emoji} {pred['signal_name']:<6} "
            f"{pred['confidence']*100:>5.1f}%  "
            f"{pred['prob_buy']*100:>5.1f}%  "
            f"{pred['prob_sell']*100:>5.1f}%  "
            f"₹{pred['close']:>9,.1f}"
        )

    print(f"{'─'*60}\n")


if __name__ == "__main__":
    _cli()
