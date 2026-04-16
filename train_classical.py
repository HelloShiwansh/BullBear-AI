"""
train_classical.py
------------------
Trains three classical ML models on the BullBear AI dataset:
  - Random Forest   (scikit-learn)
  - XGBoost
  - LightGBM

Each model is trained TWICE:
  1. With sentiment    (25 features — full feature set)
  2. Without sentiment (21 features — technical + calendar only)

This produces 6 training runs total and enables the mentor-requested
ablation study: does sentiment actually improve predictions?

Cross-validation uses TimeSeriesSplit (walk-forward) — never standard k-fold,
which would leak future data into training folds.

The best model (highest F1 macro across all 6 runs) is saved as the
production classical model.

Usage:
  python train_classical.py

Outputs (saved to data/models/ and data/results/):
  best_model_classical.pkl       — best model object
  best_model_classical_info.json — which model + condition won
  scaler.pkl                     — fitted StandardScaler
  feature_importances_<name>_<condition>.csv
  <name>_<condition>.json        — evaluation metrics
  backtest_<name>_<condition>.json
"""

import os
import json
import pickle
import warnings
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[WARN] xgboost not installed. Run: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("[WARN] lightgbm not installed. Run: pip install lightgbm")

from pipeline import load_dataset
from evaluate import (
    make_split,
    fit_scaler,
    apply_scaler,
    evaluate_model,
    run_backtest,
    print_comparison_table,
    FEATURES_WITH_SENTIMENT,
    FEATURES_WITHOUT_SENTIMENT,
    MODELS_DIR,
    RESULTS_DIR,
    DIVIDER,
)

# ── Config ────────────────────────────────────────────────────────────────────

CV_FOLDS    = 5     # TimeSeriesSplit folds for cross-validation
N_JOBS      = -1    # use all CPU cores
RANDOM_SEED = 42

# Conditions to train under
CONDITIONS = {
    "with_sentiment":    FEATURES_WITH_SENTIMENT,
    "without_sentiment": FEATURES_WITHOUT_SENTIMENT,
}


# ── Model definitions ─────────────────────────────────────────────────────────

def get_models() -> dict:
    """
    Return all models to train.
    class_weight='balanced' handles the HOLD-heavy class distribution.
    Parameters are sensible defaults — not hypertuned yet.
    Hyperparameter tuning is a future improvement after baseline is established.
    """
    # Note: class_weight is NOT set here because make_sample_weights() already
    # applies balanced class weighting via compute_sample_weight("balanced").
    # Setting class_weight="balanced" AND passing sample_weight would
    # double-penalise the majority class (HOLD), causing the model to
    # refuse to predict it — which is exactly the bug we saw.
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=10,
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS,
        ),
    }

    if XGB_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS,
            verbosity=0,
        )

    if LGBM_AVAILABLE:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS,
            verbose=-1,
        )

    return models


# ── Sample weighting ──────────────────────────────────────────────────────────

def make_sample_weights(y_train: pd.Series) -> np.ndarray:
    """
    Combine two weighting strategies:
    1. Recency weighting  — recent rows count more (linspace 0.5 → 1.0)
    2. Class weighting    — balanced class weights
    Multiply them together for a single weight per sample.

    Rationale: recent market behaviour is more predictive of near-future
    behaviour. 2024 patterns matter more than 2017 patterns.
    """
    n = len(y_train)

    # Recency weights — linearly increasing from 0.5 to 1.0
    recency_w = np.linspace(0.5, 1.0, n)

    # Class balance weights
    class_w = compute_sample_weight("balanced", y_train)

    # Combined
    combined = recency_w * class_w

    # Normalise so weights sum to n (keeps loss scale stable)
    combined = combined * n / combined.sum()
    return combined


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    condition: str,
) -> float:
    """
    Walk-forward cross-validation using TimeSeriesSplit.

    TimeSeriesSplit always trains on past data and validates on future data
    within the training set — correct for time-series, unlike standard k-fold.

    Returns mean F1 macro across all folds.
    """
    tscv   = TimeSeriesSplit(n_splits=CV_FOLDS)
    scores = []

    print(f"\n  Cross-validating {model_name} [{condition}] "
          f"({CV_FOLDS}-fold walk-forward)...")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr = X_train.iloc[train_idx]
        X_vl = X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_vl = y_train.iloc[val_idx]

        sw = make_sample_weights(y_tr)

        # Fit with sample weights where supported
        try:
            model.fit(X_tr, y_tr, sample_weight=sw)
        except TypeError:
            model.fit(X_tr, y_tr)

        from sklearn.metrics import f1_score
        y_pred = model.predict(X_vl)
        f1     = f1_score(y_vl, y_pred, average="macro", zero_division=0)
        scores.append(f1)
        print(f"    Fold {fold}/{CV_FOLDS}: F1 macro = {f1:.4f}")

    mean_f1 = float(np.mean(scores))
    std_f1  = float(np.std(scores))
    print(f"  CV result: {mean_f1:.4f} ± {std_f1:.4f}")
    return mean_f1


# ── Feature importance ────────────────────────────────────────────────────────

def save_feature_importances(
    model,
    feature_cols: list[str],
    model_name: str,
    condition: str,
) -> None:
    """
    Extract and save feature importances.
    All three model types expose .feature_importances_ so this works uniformly.
    """
    if not hasattr(model, "feature_importances_"):
        return

    importances = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    importances["rank"] = range(1, len(importances) + 1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(
        RESULTS_DIR, f"feature_importances_{model_name}_{condition}.csv"
    )
    importances.to_csv(path, index=False)

    print(f"\n  Top 10 features — {model_name} [{condition}]:")
    print(f"  {'Rank':<5} {'Feature':<30} {'Importance':>10}")
    print(f"  {'─'*5} {'─'*30} {'─'*10}")
    for _, row in importances.head(10).iterrows():
        print(f"  {int(row['rank']):<5} {row['feature']:<30} {row['importance']:>10.4f}")

    print(f"\n  Full importances saved → {path}")


# ── Single training run ───────────────────────────────────────────────────────

def train_one(
    model,
    model_name: str,
    condition: str,
    feature_cols: list[str],
    df: pd.DataFrame,
) -> dict:
    """
    Full training pipeline for one (model, condition) combination.

    Steps:
    1. Split data using the condition's feature set
    2. Fit + apply scaler
    3. Cross-validate (walk-forward)
    4. Final fit on full training set
    5. Evaluate on held-out test set
    6. Save feature importances
    7. Run backtest
    8. Save model

    Returns evaluation results dict.
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name.upper()} [{condition}]")
    print(f"Features: {len(feature_cols)}")
    print(f"{'='*60}")

    # ── Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = make_split(
        df, feature_cols=feature_cols, verbose=True
    )

    # Capture ticker series aligned with X_test for backtest return lookup.
    # Must be done before scaling (scaling changes values but not alignment).
    df_sorted    = df.sort_index()
    test_mask    = df_sorted.index > pd.Timestamp("2023-12-31")
    ticker_test  = df_sorted.loc[test_mask, "ticker"].reset_index(drop=True)

    # ── Scale ──────────────────────────────────────────────────────────────
    scaler  = fit_scaler(X_train, save=(condition == "with_sentiment"))
    X_train = apply_scaler(X_train, scaler)
    X_test  = apply_scaler(X_test,  scaler)

    # ── Cross-validate ─────────────────────────────────────────────────────
    cv_f1 = cross_validate_model(model, X_train, y_train, model_name, condition)

    # ── Final fit on full training set ─────────────────────────────────────
    print(f"\n  Final fit on full training set ({len(X_train):,} rows)...")
    t0 = time.time()
    sw = make_sample_weights(y_train)
    try:
        model.fit(X_train, y_train, sample_weight=sw)
    except TypeError:
        model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Fit complete in {elapsed:.1f}s")

    # ── Evaluate ───────────────────────────────────────────────────────────
    results = evaluate_model(
        model, X_test, y_test,
        model_name=model_name,
        condition=condition,
        save=True,
        verbose=True,
    )
    results["cv_f1_macro"] = round(cv_f1, 4)

    # ── Feature importances ────────────────────────────────────────────────
    save_feature_importances(model, feature_cols, model_name, condition)

    # ── Backtest ───────────────────────────────────────────────────────────
    print(f"\n  Running backtest...")
    run_backtest(
        model, X_test, y_test, df,
        model_name=model_name,
        condition=condition,
        ticker_series=ticker_test,
        verbose=True,
    )

    # ── Save model ─────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(
        MODELS_DIR, f"{model_name}_{condition}.pkl"
    )
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":        model,
            "feature_cols": feature_cols,
            "scaler":       scaler,
            "results":      results,
        }, f)
    print(f"\n  Model saved → {model_path}")

    return results


# ── Main training loop ────────────────────────────────────────────────────────

def run_all() -> None:
    """
    Train all models under both conditions.
    Picks the best model overall and saves it as best_model_classical.pkl.
    """
    print("=" * 60)
    print("BullBear AI — Classical ML Training")
    print("=" * 60)

    # Load dataset once — shared across all runs
    print("\nLoading dataset...")
    df = load_dataset()
    print(f"Dataset: {len(df):,} rows | {df['ticker'].nunique()} tickers")

    all_results = []
    best_f1     = -1
    best_info   = {}
    best_model  = None

    models = get_models()
    print(f"\nModels to train : {list(models.keys())}")
    print(f"Conditions      : {list(CONDITIONS.keys())}")
    print(f"Total runs      : {len(models) * len(CONDITIONS)}")

    for model_name, model_template in models.items():
        for condition, feature_cols in CONDITIONS.items():

            # Fresh model instance for each run — don't reuse fitted state
            import copy
            model = copy.deepcopy(model_template)

            results = train_one(
                model       = model,
                model_name  = model_name,
                condition   = condition,
                feature_cols= feature_cols,
                df          = df,
            )
            all_results.append(results)

            # Track best model
            if results["f1_macro"] > best_f1:
                best_f1    = results["f1_macro"]
                best_info  = {
                    "model_name": model_name,
                    "condition":  condition,
                    "f1_macro":   results["f1_macro"],
                    "accuracy":   results["accuracy"],
                }
                best_model = model

    # ── Save best model ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"BEST CLASSICAL MODEL")
    print(f"{'='*60}")
    print(f"  Model     : {best_info['model_name']}")
    print(f"  Condition : {best_info['condition']}")
    print(f"  F1 macro  : {best_info['f1_macro']:.4f}")
    print(f"  Accuracy  : {best_info['accuracy']:.4f}")

    best_path = os.path.join(MODELS_DIR, "best_model_classical.pkl")
    with open(best_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\n  Best model saved → {best_path}")

    info_path = os.path.join(MODELS_DIR, "best_model_classical_info.json")
    with open(info_path, "w") as f:
        json.dump(best_info, f, indent=2)
    print(f"  Best model info → {info_path}")

    # ── Final comparison table ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print_comparison_table()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_all()
