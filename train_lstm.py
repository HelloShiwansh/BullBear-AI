"""
train_lstm.py
-------------
Trains an LSTM model on the BullBear AI dataset using PyTorch.

Key difference from classical ML:
  Each training sample is a SEQUENCE of 30 consecutive trading days,
  not a single row. The LSTM learns temporal patterns across the window —
  what happened over the past 30 days predicts tomorrow's signal.

Architecture:
  Input  : (batch, 30, n_features)   30-day window of features
  LSTM   : 2 layers, hidden=128, dropout=0.3
  FC     : 128 → 64 → 3
  Output : softmax probabilities for BUY / HOLD / SELL

Trained TWICE (same as classical):
  1. With sentiment    (25 features)
  2. Without sentiment (21 features)

Training uses early stopping — stops automatically when validation loss
stops improving, saving the best checkpoint.

Usage:
  python train_lstm.py

Outputs (saved to data/models/ and data/results/):
  lstm_with_sentiment.pt
  lstm_without_sentiment.pt
  lstm_config_<condition>.json
  training_curves_<condition>.png  (loss + accuracy per epoch)
  lstm_<condition>.json            (evaluation metrics)
  backtest_lstm_<condition>.json
"""

import os
import json
import time
import warnings
import copy

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[ERROR] PyTorch not installed. Run: pip install torch")
    exit(1)

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

# ── Config ─────────────────────────────────────────────────────────────────────

WINDOW_SIZE   = 30     # trading days per sequence
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE    = 64
MAX_EPOCHS    = 100
PATIENCE      = 10     # early stopping patience
RANDOM_SEED   = 42
N_CLASSES     = 3      # BUY, HOLD, SELL

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONDITIONS = {
    "with_sentiment":    FEATURES_WITH_SENTIMENT,
    "without_sentiment": FEATURES_WITHOUT_SENTIMENT,
}


# ── A. Sequence Dataset ────────────────────────────────────────────────────────

class StockSequenceDataset(Dataset):
    """
    PyTorch Dataset that builds sliding windows from the feature matrix.

    Each sample:
      X : (WINDOW_SIZE, n_features)  — 30 consecutive days of features
      y : int                        — signal label for the day AFTER the window

    Critical rule — windows must not cross ticker boundaries.
    RELIANCE's day 30 must not be followed by TCS's day 1.
    We build windows per-ticker then concatenate.

    The first (WINDOW_SIZE - 1) rows of each ticker cannot form a complete
    window — they are dropped, adding ~29 more warm-up rows per ticker on
    top of the pipeline's existing 34-row warm-up.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tickers: pd.Series,
        window_size: int = WINDOW_SIZE,
    ):
        self.sequences: list[np.ndarray] = []
        self.labels:    list[int]        = []

        # Build windows per ticker to prevent cross-ticker contamination
        unique_tickers = tickers.unique()

        for ticker in unique_tickers:
            mask    = (tickers == ticker).values
            X_tick  = X.values[mask]
            y_tick  = y.values[mask]

            n = len(X_tick)
            if n <= window_size:
                continue  # ticker too short to form any window

            for i in range(window_size, n):
                seq   = X_tick[i - window_size : i]   # shape (window_size, n_features)
                label = int(y_tick[i])
                self.sequences.append(seq)
                self.labels.append(label)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels    = np.array(self.labels,    dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx],    dtype=torch.long),
        )


# ── B. LSTM Model Architecture ─────────────────────────────────────────────────

class BullBearLSTM(nn.Module):
    """
    Two-layer LSTM followed by a two-layer fully connected classifier.

    Architecture:
      LSTM (input → hidden=128, layers=2, dropout=0.3, batch_first=True)
      → take only the last timestep output
      → FC: 128 → 64 → ReLU → Dropout(0.3) → 3
      → softmax (applied externally by loss function)

    dropout only applied between LSTM layers (not after the last layer)
    so we set dropout=0.3 only when num_layers > 1.
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers:  int = NUM_LAYERS,
        dropout:     float = DROPOUT,
        n_classes:   int = N_CLASSES,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            dropout      = dropout if num_layers > 1 else 0.0,
            batch_first  = True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        returns: (batch, n_classes) — raw logits
        """
        lstm_out, _ = self.lstm(x)
        last_step   = lstm_out[:, -1, :]   # take only the final timestep
        logits      = self.classifier(last_step)
        return logits


# ── C. Class weights for loss function ────────────────────────────────────────

def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.
    Handles the HOLD-heavy imbalance the same way classical ML does.
    weight[c] = total_samples / (n_classes * count[c])
    """
    counts  = np.bincount(y, minlength=N_CLASSES).astype(float)
    weights = len(y) / (N_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


# ── D. Training loop ───────────────────────────────────────────────────────────

def train_epoch(
    model:      BullBearLSTM,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
) -> tuple[float, float]:
    """One training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in LSTM
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


def validate_epoch(
    model:     BullBearLSTM,
    loader:    DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    """One validation epoch. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits      = model(X_batch)
            loss        = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)

    return total_loss / total, correct / total


# ── E. Prediction wrapper ──────────────────────────────────────────────────────

class LSTMPredictor:
    """
    Thin wrapper around a trained LSTM so it exposes .predict()
    compatible with evaluate_model() and run_backtest().
    """

    def __init__(self, model: BullBearLSTM, dataset: StockSequenceDataset):
        self.model   = model
        self.dataset = dataset

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for the full test dataset.
        X_test is ignored — we use the pre-built sequence dataset.
        Predictions are in the same order as the dataset's labels.
        """
        self.model.eval()
        all_preds = []

        loader = DataLoader(
            self.dataset,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
        )

        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(DEVICE)
                logits  = self.model(X_batch)
                preds   = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)

        return np.array(all_preds)


# ── F. Save training curves ────────────────────────────────────────────────────

def save_training_curves(
    history: dict,
    condition: str,
) -> None:
    """
    Save training curves as a simple text file.
    Avoids matplotlib dependency — can be plotted externally if needed.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"training_curves_lstm_{condition}.csv")

    rows = []
    for epoch in range(len(history["train_loss"])):
        rows.append({
            "epoch":      epoch + 1,
            "train_loss": round(history["train_loss"][epoch], 6),
            "val_loss":   round(history["val_loss"][epoch],   6),
            "train_acc":  round(history["train_acc"][epoch],  6),
            "val_acc":    round(history["val_acc"][epoch],    6),
        })

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\n  Training curves saved → {path}")


# ── G. Full training run ───────────────────────────────────────────────────────

def train_one(
    condition:    str,
    feature_cols: list[str],
    df:           pd.DataFrame,
) -> dict:
    """
    Full LSTM training pipeline for one condition.

    Steps:
    1. Split data (same boundaries as classical models)
    2. Fit + apply scaler
    3. Build sequence datasets (per-ticker sliding windows)
    4. Train with early stopping
    5. Evaluate on test sequences
    6. Backtest
    7. Save model + config

    Returns evaluation results dict.
    """
    n_features = len(feature_cols)

    print(f"\n{'='*60}")
    print(f"Training: LSTM [{condition}]")
    print(f"Features: {n_features} | Window: {WINDOW_SIZE} days | Device: {DEVICE}")
    print(f"{'='*60}")

    # ── Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = make_split(
        df, feature_cols=feature_cols, verbose=True
    )

    # Need ticker labels for window building — retrieve from df
    df_sorted  = df.sort_index()
    train_mask = df_sorted.index <= pd.Timestamp("2023-12-31")
    test_mask  = df_sorted.index >  pd.Timestamp("2023-12-31")

    tickers_train = df_sorted.loc[train_mask, "ticker"].reset_index(drop=True)
    tickers_test  = df_sorted.loc[test_mask,  "ticker"].reset_index(drop=True)

    # Reset indices for alignment
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    # ── Scale ──────────────────────────────────────────────────────────────
    scaler  = fit_scaler(
        X_train,
        save=(condition == "with_sentiment")
    )
    X_train_sc = apply_scaler(X_train, scaler)
    X_test_sc  = apply_scaler(X_test,  scaler)

    # ── Build sequence datasets ────────────────────────────────────────────
    print(f"\n  Building {WINDOW_SIZE}-day sequence windows...")

    # Use 80% of train for training, 20% for validation (walk-forward)
    n_train    = len(X_train_sc)
    val_start  = int(n_train * 0.8)

    X_tr  = X_train_sc.iloc[:val_start]
    X_val = X_train_sc.iloc[val_start:]
    y_tr  = y_train.iloc[:val_start]
    y_val = y_train.iloc[val_start:]
    t_tr  = tickers_train.iloc[:val_start]
    t_val = tickers_train.iloc[val_start:]

    train_dataset = StockSequenceDataset(X_tr,       y_tr,   t_tr,         WINDOW_SIZE)
    val_dataset   = StockSequenceDataset(X_val,      y_val,  t_val,        WINDOW_SIZE)
    test_dataset  = StockSequenceDataset(X_test_sc,  y_test, tickers_test, WINDOW_SIZE)

    print(f"  Train sequences : {len(train_dataset):,}")
    print(f"  Val   sequences : {len(val_dataset):,}")
    print(f"  Test  sequences : {len(test_dataset):,}")

    if len(train_dataset) == 0:
        raise RuntimeError("No training sequences built — check window size vs data length.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Model, loss, optimiser ────────────────────────────────────────────
    model     = BullBearLSTM(input_size=n_features).to(DEVICE)
    class_w   = compute_class_weights(train_dataset.labels)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {total_params:,}")

    # ── Training loop with early stopping ─────────────────────────────────
    print(f"\n  Training (max {MAX_EPOCHS} epochs, patience={PATIENCE})...")
    print(f"  {'Epoch':<8} {'Train Loss':>10} {'Val Loss':>10} "
          f"{'Train Acc':>10} {'Val Acc':>10} {'':>6}")

    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_weights  = None
    patience_ctr  = 0

    t_start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc = validate_epoch(model, val_loader, criterion)

        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        improved = vl_loss < best_val_loss
        marker   = " *" if improved else ""

        if epoch % 5 == 0 or epoch == 1 or improved:
            print(f"  {epoch:<8} {tr_loss:>10.4f} {vl_loss:>10.4f} "
                  f"{tr_acc:>10.4f} {vl_acc:>10.4f} {marker:>6}")

        if improved:
            best_val_loss = vl_loss
            best_weights  = copy.deepcopy(model.state_dict())
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    elapsed = time.time() - t_start
    best_epoch = len(history["val_loss"]) - patience_ctr
    print(f"\n  Training complete: {elapsed:.1f}s | Best epoch: {best_epoch}")

    # Restore best weights
    model.load_state_dict(best_weights)

    # ── Save training curves ───────────────────────────────────────────────
    save_training_curves(history, condition)

    # ── Evaluate on test sequences ─────────────────────────────────────────
    print(f"\n  Evaluating on test sequences ({len(test_dataset):,} samples)...")

    # Get test labels aligned with sequences
    y_test_seq = pd.Series(test_dataset.labels)

    # Wrap model in predictor for compatibility with evaluate_model
    predictor = LSTMPredictor(model, test_dataset)

    # evaluate_model calls model.predict(X_test) — we pass a dummy X_test
    # since LSTMPredictor.predict() uses the pre-built dataset internally
    dummy_X = pd.DataFrame(
        np.zeros((len(test_dataset), n_features)),
        columns=feature_cols,
    )

    results = evaluate_model(
        model       = predictor,
        X_test      = dummy_X,
        y_test      = y_test_seq,
        model_name  = "lstm",
        condition   = condition,
        save        = True,
        verbose     = True,
    )

    # ── Backtest ───────────────────────────────────────────────────────────
    # Use the shared run_backtest from evaluate.py which looks up returns
    # by (date, ticker) key from the raw unscaled dataset — correct alignment.
    # LSTM sequences are built from scaled data so we cannot extract returns
    # from the sequence array directly (they would be standardised values,
    # not actual percentages). The raw dataset is the source of truth.
    print(f"\n  Running backtest...")

    # Build ticker series aligned with test sequences.
    # Each sequence corresponds to the ticker of its last day.
    # test_dataset was built per-ticker so we can reconstruct the mapping.
    test_ticker_list = []
    for ticker in tickers_test.unique():
        mask   = (tickers_test == ticker).values
        n_tick = mask.sum()
        if n_tick > WINDOW_SIZE:
            test_ticker_list.extend([ticker] * (n_tick - WINDOW_SIZE))
    ticker_series_test = pd.Series(test_ticker_list[:len(test_dataset)])

    # Date series: last date of each sequence window
    # Reconstruct from df test slice sorted by ticker then date
    df_test_raw = df[df.index > pd.Timestamp("2023-12-31")].sort_values(
        ["ticker", df.index.name if df.index.name else "Date"]
        if df.index.name else ["ticker"]
    )
    date_list = []
    for ticker in tickers_test.unique():
        t_dates = df_test_raw[df_test_raw["ticker"] == ticker].index
        if len(t_dates) > WINDOW_SIZE:
            date_list.extend(t_dates[WINDOW_SIZE:])
    date_series_test = pd.DatetimeIndex(date_list[:len(test_dataset)])

    # Build a dummy X_test with the correct date index for run_backtest
    dummy_X_indexed = pd.DataFrame(
        np.zeros((len(test_dataset), n_features)),
        index=date_series_test,
        columns=feature_cols,
    )

    run_backtest(
        model         = predictor,
        X_test        = dummy_X_indexed,
        y_test        = y_test_seq,
        df_full       = df,
        model_name    = "lstm",
        condition     = condition,
        ticker_series = ticker_series_test,
        verbose       = True,
    )

    # ── Save model ─────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"lstm_{condition}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model saved → {model_path}")

    # Save config for loading later
    config = {
        "input_size":    n_features,
        "hidden_size":   HIDDEN_SIZE,
        "num_layers":    NUM_LAYERS,
        "dropout":       DROPOUT,
        "n_classes":     N_CLASSES,
        "window_size":   WINDOW_SIZE,
        "feature_cols":  feature_cols,
        "condition":     condition,
        "best_epoch":    best_epoch,
        "best_val_loss": round(best_val_loss, 6),
    }
    config_path = os.path.join(MODELS_DIR, f"lstm_config_{condition}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved  → {config_path}")

    return results


# ── H. Main ───────────────────────────────────────────────────────────────────

def run_all() -> None:
    """Train LSTM under both conditions. Print final comparison."""
    print("=" * 60)
    print("BullBear AI — LSTM Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Window: {WINDOW_SIZE} days | Hidden: {HIDDEN_SIZE} | "
          f"Layers: {NUM_LAYERS} | Dropout: {DROPOUT}")

    print("\nLoading dataset...")
    df = load_dataset()
    print(f"Dataset: {len(df):,} rows | {df['ticker'].nunique()} tickers")

    all_results = []
    best_f1     = -1
    best_info   = {}

    for condition, feature_cols in CONDITIONS.items():
        results = train_one(
            condition    = condition,
            feature_cols = feature_cols,
            df           = df,
        )
        all_results.append(results)

        if results["f1_macro"] > best_f1:
            best_f1   = results["f1_macro"]
            best_info = {
                "model_name": "lstm",
                "condition":  condition,
                "f1_macro":   results["f1_macro"],
                "accuracy":   results["accuracy"],
            }

    # ── Best LSTM summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BEST LSTM")
    print(f"{'='*60}")
    print(f"  Condition : {best_info['condition']}")
    print(f"  F1 macro  : {best_info['f1_macro']:.4f}")
    print(f"  Accuracy  : {best_info['accuracy']:.4f}")

    info_path = os.path.join(MODELS_DIR, "best_model_lstm_info.json")
    with open(info_path, "w") as f:
        json.dump(best_info, f, indent=2)

    # ── Full comparison (classical + LSTM) ─────────────────────────────────
    print(f"\n{'='*60}")
    print("FULL MODEL COMPARISON (Classical + LSTM)")
    print_comparison_table()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_all()
