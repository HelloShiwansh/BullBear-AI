"""
evaluate.py
-----------
Shared foundation used by train_classical.py and train_lstm.py.

Responsibilities:
  1. Time-based train/test split (no random splitting — time-series rule)
  2. StandardScaler fitted on train only, applied to both splits
  3. Evaluation metrics: F1, confusion matrix, classification report
  4. Backtest: simulate trading signals on test period vs buy-and-hold
  5. Side-by-side comparison table across all 4 model runs

Usage (from training scripts):
  from evaluate import make_split, fit_scaler, apply_scaler, evaluate_model, run_backtest

Usage (standalone — prints comparison of all saved results):
  python evaluate.py
"""

import os
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from pipeline import load_dataset
from feature_engineering import FEATURE_COLUMNS

# ── Config ────────────────────────────────────────────────────────────────────

MODELS_DIR  = "data/models"
RESULTS_DIR = "data/results"

# Time-based split boundary
# ~80% train (2017–2023), ~20% test (2024–2026)
TRAIN_END_DATE = "2023-12-31"

# Features that need StandardScaler
SCALE_COLS = ["daily_return", "volatility", "log_volume"]

# Label mapping for display
LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}
LABELS    = [0, 1, 2]
DIVIDER   = "─" * 60

# Feature sets for the two experiment conditions
FEATURES_WITH_SENTIMENT    = FEATURE_COLUMNS  # all 25
FEATURES_WITHOUT_SENTIMENT = [
    c for c in FEATURE_COLUMNS
    if c not in (
        "sentiment_score",
        "sentiment_magnitude",
        "sentiment_article_count",
        "has_sentiment",
    )
]  # 21 features


# ── 1. Train / Test Split ─────────────────────────────────────────────────────

def make_split(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    train_end: str = TRAIN_END_DATE,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-based train/test split.

    Split is done per ticker first, then combined. This ensures that for
    every ticker, all training rows come before all test rows chronologically.
    Stacking tickers before splitting could interleave dates across tickers.

    Parameters
    ----------
    df           : full dataset from load_dataset()
    feature_cols : which features to use (default: FEATURE_COLUMNS / all 25)
    train_end    : last date of training period (inclusive)
    verbose      : print split summary

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    # Derive has_sentiment on the fly if missing from the CSV.
    # This happens when the dataset was built before has_sentiment was added
    # to FEATURE_COLUMNS — avoids needing to re-run the full pipeline.
    if "has_sentiment" not in df.columns and "sentiment_score" in df.columns:
        df = df.copy()
        df["has_sentiment"] = (df["sentiment_score"] != 0).astype(int)

    # Validate all requested features exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Features missing from dataset: {missing}")

    split_date = pd.Timestamp(train_end)
    train_rows, test_rows = [], []

    for ticker in df["ticker"].unique():
        ticker_df = df[df["ticker"] == ticker].sort_index()
        train_rows.append(ticker_df[ticker_df.index <= split_date])
        test_rows.append(ticker_df[ticker_df.index >  split_date])

    train_df = pd.concat(train_rows).sort_index()
    test_df  = pd.concat(test_rows).sort_index()

    X_train = train_df[feature_cols]
    X_test  = test_df[feature_cols]
    y_train = train_df["signal"]
    y_test  = test_df["signal"]

    if verbose:
        print(f"\nTrain/test split at {train_end}:")
        print(f"  Train : {len(X_train):>7,} rows | "
              f"{train_df.index.min().date()} → {train_df.index.max().date()}")
        print(f"  Test  : {len(X_test):>7,} rows | "
              f"{test_df.index.min().date()} → {test_df.index.max().date()}")
        print(f"  Features: {len(feature_cols)}")

        # Class distribution in each split
        for name, y in [("Train", y_train), ("Test", y_test)]:
            dist = y.value_counts(normalize=True).sort_index()
            parts = [f"{LABEL_MAP[k]}={v*100:.1f}%" for k, v in dist.items()]
            print(f"  {name} distribution: {' | '.join(parts)}")

    return X_train, X_test, y_train, y_test


# ── 2. Scaling ────────────────────────────────────────────────────────────────

def fit_scaler(
    X_train: pd.DataFrame,
    save: bool = True,
) -> StandardScaler:
    """
    Fit StandardScaler on training data only.
    Only scales SCALE_COLS — other features are already bounded.

    Saves scaler to data/models/scaler.pkl for use at prediction time.
    """
    scaler = StandardScaler()

    # Only fit on columns that exist in this feature set
    cols_to_scale = [c for c in SCALE_COLS if c in X_train.columns]
    scaler.fit(X_train[cols_to_scale])

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
            pickle.dump({"scaler": scaler, "cols": cols_to_scale}, f)
        print(f"  Scaler saved → {MODELS_DIR}/scaler.pkl")

    return scaler


def apply_scaler(
    X: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Apply a fitted scaler to a feature DataFrame.
    Only transforms SCALE_COLS — leaves all other features untouched.
    Returns a copy — never mutates the input.
    """
    X = X.copy()
    cols_to_scale = [c for c in SCALE_COLS if c in X.columns]
    if cols_to_scale:
        X[cols_to_scale] = scaler.transform(X[cols_to_scale])
    return X


def load_scaler() -> tuple[StandardScaler, list[str]]:
    """Load saved scaler from disk. Used at prediction time."""
    path = os.path.join(MODELS_DIR, "scaler.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Scaler not found at {path}.\n"
            "Run train_classical.py or train_lstm.py first."
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["scaler"], data["cols"]


# ── 3. Evaluation Metrics ─────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    condition: str,
    save: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Evaluate any trained model and save results.

    Parameters
    ----------
    model      : fitted model with .predict() method
    X_test     : scaled test features
    y_test     : true labels
    model_name : e.g. "random_forest", "xgboost", "lstm"
    condition  : "with_sentiment" or "without_sentiment"
    save       : save results to data/results/
    verbose    : print full report

    Returns
    -------
    dict with all metrics — used by comparison table
    """
    y_pred = model.predict(X_test)

    # Core metrics
    acc     = accuracy_score(y_test, y_pred)
    f1_mac  = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1_buy  = f1_score(y_test, y_pred, labels=[2], average="macro", zero_division=0)
    f1_sell = f1_score(y_test, y_pred, labels=[0], average="macro", zero_division=0)
    f1_hold = f1_score(y_test, y_pred, labels=[1], average="macro", zero_division=0)
    cm      = confusion_matrix(y_test, y_pred, labels=LABELS)
    report  = classification_report(
        y_test, y_pred,
        labels=LABELS,
        target_names=["SELL", "HOLD", "BUY"],
        zero_division=0,
    )

    results = {
        "model":            model_name,
        "condition":        condition,
        "accuracy":         round(acc,     4),
        "f1_macro":         round(f1_mac,  4),
        "f1_buy":           round(f1_buy,  4),
        "f1_sell":          round(f1_sell, 4),
        "f1_hold":          round(f1_hold, 4),
        "confusion_matrix": cm.tolist(),
        "timestamp":        datetime.now().isoformat(),
    }

    if verbose:
        label = f"{model_name} [{condition}]"
        print(f"\n{DIVIDER}")
        print(f"Evaluation — {label}")
        print(DIVIDER)
        print(f"\n  Accuracy : {acc:.4f}")
        print(f"  F1 macro : {f1_mac:.4f}")
        print(f"\n  Per-class F1:")
        print(f"    BUY  : {f1_buy:.4f}")
        print(f"    HOLD : {f1_hold:.4f}")
        print(f"    SELL : {f1_sell:.4f}")
        print(f"\n  Classification report:")
        print(report)
        print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
        _print_confusion_matrix(cm)

    if save:
        _save_results(results, model_name, condition)

    return results


def _print_confusion_matrix(cm: np.ndarray) -> None:
    """Print a readable confusion matrix with labels."""
    names = ["SELL", "HOLD", "BUY"]
    header = "           " + "  ".join(f"{n:>6}" for n in names)
    print(f"  {header}")
    for i, name in enumerate(names):
        row = "  ".join(f"{cm[i,j]:>6,}" for j in range(3))
        print(f"  {name:>8} : {row}")


def _save_results(results: dict, model_name: str, condition: str) -> None:
    """Save evaluation results as JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = f"{model_name}_{condition}.json"
    path     = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {path}")


# ── 4. Backtest ───────────────────────────────────────────────────────────────

def _build_return_lookup(df_full: pd.DataFrame) -> pd.Series:
    """
    Build a (date_str, ticker) → next_return lookup from the raw dataset.

    This is the correct way to align returns with predictions when the
    dataset has 20 tickers sharing the same dates. Looking up by date alone
    returns 20 rows; looking up by (date, ticker) returns exactly one.

    Uses next_return (the actual next-day return the signal is predicting).
    Falls back to daily_return if next_return is not in the dataset.
    """
    return_col = "next_return" if "next_return" in df_full.columns else "daily_return"
    df = df_full.copy()
    df["_key"] = df.index.astype(str) + "||" + df["ticker"].astype(str)
    lookup = df.set_index("_key")[return_col].fillna(0.0)
    return lookup


def run_backtest(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_full: pd.DataFrame,
    model_name: str,
    condition: str,
    ticker_series: pd.Series | None = None,
    verbose: bool = True,
) -> dict:
    """
    Simulate trading on the test period using model signals.

    Strategy:
      BUY  signal → long: gain actual next-day return
      SELL signal → flat: return = 0 (avoiding the loss)
      HOLD signal → flat: return = 0

    Fix for multi-ticker date ambiguity:
      Returns are looked up by (date, ticker) key — unique across the dataset.
      X_test.index gives dates. ticker_series gives the corresponding ticker
      for each row. Together they form a unique key per row.

    Buy-and-hold benchmark:
      Average next-day return across all tickers in the test period.
      Represents investing equally in all 20 stocks and holding.

    Parameters
    ----------
    model         : fitted model with .predict()
    X_test        : scaled test features
    y_test        : true labels
    df_full       : full unscaled dataset (for return lookup)
    model_name    : e.g. "random_forest", "lstm"
    condition     : "with_sentiment" or "without_sentiment"
    ticker_series : pd.Series of ticker names aligned with X_test rows.
                    If None, uses df_full filtered to test dates.
    verbose       : print results

    Returns
    -------
    dict with backtest metrics
    """
    y_pred = model.predict(X_test)

    # ── Build (date, ticker) → return lookup ──────────────────────────────
    return_lookup = _build_return_lookup(df_full)

    # ── Get ticker per test row ────────────────────────────────────────────
    if ticker_series is not None:
        # Classical path: ticker_series passed in, aligned with X_test
        tickers = ticker_series.values
        dates   = X_test.index.astype(str)
    else:
        # Fallback: derive from df_full filtered to test dates
        test_dates = X_test.index
        df_test    = df_full[df_full.index.isin(test_dates)].sort_index()
        tickers    = df_test["ticker"].values
        dates      = df_test.index.astype(str)

    # ── Look up actual next-day returns per (date, ticker) ────────────────
    min_len = min(len(y_pred), len(tickers), len(dates))
    y_pred  = y_pred[:min_len]

    actual_returns = np.array([
        return_lookup.get(f"{d}||{t}", 0.0)
        for d, t in zip(dates[:min_len], tickers[:min_len])
    ])

    # ── Strategy returns ───────────────────────────────────────────────────
    # BUY  → take the actual next-day return
    # SELL → stay flat (0) — we avoid the loss
    # HOLD → stay flat (0)
    strategy_returns = np.where(y_pred == 2, actual_returns, 0.0)

    # ── Buy-and-hold benchmark ─────────────────────────────────────────────
    # Mean daily return across all tickers — equal weight portfolio held flat
    return_col = "next_return" if "next_return" in df_full.columns else "daily_return"
    bah_daily  = (
        df_full[df_full.index > pd.Timestamp(TRAIN_END_DATE)]
        .groupby(df_full[df_full.index > pd.Timestamp(TRAIN_END_DATE)].index)[return_col]
        .mean()
        .fillna(0.0)
        .values
    )

    # ── Cumulative returns ─────────────────────────────────────────────────
    # Guard against extreme values (scaled data leak) — clip at ±50% per day
    strategy_returns = np.clip(strategy_returns, -0.5, 0.5)
    bah_daily        = np.clip(bah_daily,        -0.5, 0.5)

    strategy_cumret = float(np.prod(1 + strategy_returns) - 1)
    bah_cumret      = float(np.prod(1 + bah_daily) - 1)
    outperformance  = strategy_cumret - bah_cumret

    # ── Signal distribution ────────────────────────────────────────────────
    pred_series  = pd.Series(y_pred)
    total        = len(y_pred)
    buy_signals  = int((pred_series == 2).sum())
    sell_signals = int((pred_series == 0).sum())
    hold_signals = int((pred_series == 1).sum())

    backtest_results = {
        "model":               model_name,
        "condition":           condition,
        "strategy_return_pct": round(strategy_cumret * 100, 2),
        "buyhold_return_pct":  round(bah_cumret * 100, 2),
        "outperformance_pct":  round(outperformance * 100, 2),
        "buy_signals_pct":     round(100 * buy_signals  / total, 1),
        "sell_signals_pct":    round(100 * sell_signals / total, 1),
        "hold_signals_pct":    round(100 * hold_signals / total, 1),
    }

    if verbose:
        print(f"\n  Backtest — {model_name} [{condition}]")
        print(f"  Strategy return  : {strategy_cumret*100:+.2f}%")
        print(f"  Buy-and-hold     : {bah_cumret*100:+.2f}%")
        print(f"  Outperformance   : {outperformance*100:+.2f}%")
        print(f"  Signal mix       : BUY={buy_signals/total*100:.1f}%  "
              f"SELL={sell_signals/total*100:.1f}%  "
              f"HOLD={hold_signals/total*100:.1f}%")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"backtest_{model_name}_{condition}.json")
    with open(path, "w") as f:
        json.dump(backtest_results, f, indent=2)

    return backtest_results


# ── 5. Comparison Table ───────────────────────────────────────────────────────

def print_comparison_table() -> None:
    """
    Load all saved results and print a side-by-side comparison.
    Run this after all 4 models have been trained and evaluated.
    """
    if not os.path.exists(RESULTS_DIR):
        print("No results found. Train models first.")
        return

    result_files = [
        f for f in os.listdir(RESULTS_DIR)
        if f.endswith(".json") and not f.startswith("backtest_")
    ]

    if not result_files:
        print("No evaluation results found yet.")
        return

    records = []
    for fname in sorted(result_files):
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            records.append(json.load(f))

    # Load backtest results
    backtest_files = [
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith("backtest_") and f.endswith(".json")
    ]
    backtest_map = {}
    for fname in backtest_files:
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            bt = json.load(f)
            backtest_map[f"{bt['model']}_{bt['condition']}"] = bt

    print(f"\n{DIVIDER}")
    print("MODEL COMPARISON TABLE")
    print(DIVIDER)

    header = (
        f"{'Model':<20} {'Condition':<22} "
        f"{'F1 macro':>8} {'F1 BUY':>8} {'F1 SELL':>8} "
        f"{'F1 HOLD':>8} {'Accuracy':>9} {'Backtest':>9}"
    )
    print(f"\n{header}")
    print("─" * len(header))

    for r in records:
        key    = f"{r['model']}_{r['condition']}"
        bt_ret = backtest_map.get(key, {}).get("strategy_return_pct", "N/A")
        bt_str = f"{bt_ret:+.1f}%" if isinstance(bt_ret, (int, float)) else "N/A"

        print(
            f"  {r['model']:<18} {r['condition']:<22} "
            f"{r['f1_macro']:>8.4f} {r['f1_buy']:>8.4f} "
            f"{r['f1_sell']:>8.4f} {r['f1_hold']:>8.4f} "
            f"{r['accuracy']:>9.4f} {bt_str:>9}"
        )

    print(f"\n  Buy-and-hold benchmark: see individual backtest files in {RESULTS_DIR}/")
    print(f"  Best model: highest F1 macro wins for production deployment.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading dataset...")
    df = load_dataset()

    print("\nDemonstrating split with full feature set:")
    X_train, X_test, y_train, y_test = make_split(df)

    print("\nFitting scaler on training data...")
    scaler  = fit_scaler(X_train)
    X_train = apply_scaler(X_train, scaler)
    X_test  = apply_scaler(X_test,  scaler)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test  shape: {X_test.shape}")

    print("\nScaled feature stats (train):")
    print(X_train[SCALE_COLS].describe().round(4).to_string())

    print("\n\nPrinting any saved comparison results:")
    print_comparison_table()
