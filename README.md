# BullBear AI — Stock Trend Prediction System

BullBear AI is an end-to-end machine learning system that predicts Buy/Sell/Hold signals for NIFTY-50 stocks using technical indicators, calendar seasonality, and news sentiment analysis. Features include automated data pipelines, multiple ML models (Random Forest, XGBoost, LSTM), backtesting capabilities, and an interactive Streamlit dashboard. Achieves 33% outperformance over buy-and-hold strategies.


---

## Results

| Model | Condition | F1 Macro | Backtest Return |
|---|---|---|---|
| **Random Forest** | **w/o sentiment** | **0.3769 ★** | **+38.1% ★** |
| Random Forest | with sentiment | 0.3743 | +32.1% |
| XGBoost | w/o sentiment | 0.3745 | +21.6% |
| XGBoost | with sentiment | 0.3692 | −24.8% |
| LightGBM | w/o sentiment | 0.3679 | −2.5% |
| LightGBM | with sentiment | 0.3676 | −10.3% |
| LSTM | with sentiment | 0.3294 | +69.5% |
| LSTM | w/o sentiment | 0.3197 | −40.7% |

**Buy-and-hold benchmark: +4.98%** · Test period: 2024–2026 · Production model: Random Forest w/o sentiment

---

## Stack

- **Data** — `yfinance` · 20 NIFTY tickers · 9 years (2017–2026) · 43,760 rows
- **Features** — 25 total: technical indicators, calendar/seasonality, FinBERT sentiment
- **Models** — Random Forest, XGBoost, LightGBM, LSTM (8 runs total)
- **Dashboard** — Streamlit + Plotly

---

## Project Structure

```
├── data_fetch.py            # Download OHLCV from yfinance
├── preprocess.py            # Clean, validate, forward-fill
├── feature_engineering.py   # Engineer all 25 features
├── target.py                # Create Buy/Sell/Hold labels
├── pipeline.py              # Orchestrator (use_sentiment=True/False)
├── sentiment.py             # FinBERT scoring + daily aggregation
├── evaluate.py              # Train/test split, scaler, metrics, backtest
├── train_classical.py       # RF / XGBoost / LightGBM — 6 runs
├── train_lstm.py            # LSTM — 2 runs
├── data_check.py            # Pre-training data validation
├── predict.py               # Live prediction engine
├── dashboard.py             # Streamlit dashboard
│
└── data/
    ├── raw/                 # 20 ticker CSVs
    ├── news/                # Kaggle news + FinBERT cache
    ├── processed/           # bullbear_dataset.csv
    ├── models/              # Trained models + scaler.pkl
    └── results/             # Metrics, backtest, feature importances
```

---

## Quickstart

```bash
python -m venv bb_venv
bb_venv\Scripts\activate          # Windows
# source bb_venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### Run the full pipeline

```bash
python pipeline.py          # Build dataset
python train_classical.py   # Train RF, XGBoost, LightGBM
python train_lstm.py        # Train LSTM
python evaluate.py          # Compare all models
```

### Launch the dashboard

```bash
streamlit run dashboard.py
```

### Run live predictions from the terminal

```bash
python predict.py                  # All 20 tickers
python predict.py --ticker TCS.NS  # Single ticker
```

---

## Features

### Technical (11)
`ma10_ratio` · `ma30_ratio` · `rsi` · `macd_line_ratio` · `macd_signal_ratio` · `macd_hist_ratio` · `bb_width` · `bb_pct` · `daily_return` · `volatility` · `log_volume`

### Calendar / Seasonality (10)
`month` · `quarter` · `day_of_week` · `is_monsoon` · `is_winter` · `is_summer` · `month_sin` · `month_cos` · `day_of_week_sin` · `day_of_week_cos`

### Sentiment — FinBERT (4)
`sentiment_score` · `sentiment_magnitude` · `sentiment_article_count` · `has_sentiment`

Source: [Kaggle — India Financial News](https://www.kaggle.com/datasets/harshrkh/india-financial-news-headlines-sentiments) · Coverage: 2017–2021 (10.7% of rows)

---

## Target

```
BUY  (2)  next-day return > +1%
HOLD (1)  −1% ≤ next-day return ≤ +1%
SELL (0)  next-day return < −1%
```

---

## Key Design Decisions

- **Time-based split only** — no shuffling, train ≤ 2023-12-31, test ≥ 2024-01-01
- **Scaler fitted on train only** — `StandardScaler` saved to `data/models/scaler.pkl`, applied separately to test
- **TimeSeriesSplit(n_splits=5)** for cross-validation — regular k-fold is invalid for time-series
- **MA / MACD / BB ratios** — normalised by price so features are comparable across all 20 tickers
- **Post-close news → next trading day** — articles after 15:30 IST are attributed to the next day to prevent leakage
- **Sentiment = 0 for missing days** — neutral fill, not negative

---

## Dashboard

Three tabs:

| Tab | Content |
|---|---|
| Overview | Signal card grid for all 20 tickers, sorted BUY → SELL → HOLD, with probability bars |
| Ticker Detail | Candlestick chart with BUY▲ / SELL▼ markers · confidence gauge · feature importance · signal history table |
| Model Performance | Full 8-model comparison table · key findings |

---

## Notes

- `ta` library is **not required** — manual fallbacks are implemented for all indicators
- Sentiment data only covers 2017–2021; the production model intentionally excludes sentiment (3 of 4 model families perform better without it at 10.7% coverage)
- The LSTM +69.5% backtest figure is a known selectivity artifact — it predicts HOLD 88% of the time. F1 macro is the reliable metric.

---

> Academic / research use only. Not intended for live trading without further validation.
