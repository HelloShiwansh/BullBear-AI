# BullBear AI – Data Preparation Pipeline

Produces a clean, ML-ready dataset of Buy/Sell/Hold signals for the top 20 NIFTY stocks.

## Quickstart

```bash
pip install -r requirements.txt
python pipeline.py
```

The final dataset lands at `data/processed/bullbear_dataset.csv`.

## File Map

| File | Responsibility |
|------|---------------|
| `data_fetch.py` | Download OHLCV from Yahoo Finance; cache to CSV |
| `preprocess.py` | Sort, validate, forward-fill small gaps |
| `feature_engineering.py` | Technical indicators + calendar/cyclical features |
| `target.py` | Next-day return → BUY / HOLD / SELL label |
| `pipeline.py` | Orchestrator; outputs final dataset |

## Features (21 total)

### Technical — Trend
| Feature | Description |
|---------|-------------|
| `ma10_ratio` | Close / 10-day MA |
| `ma30_ratio` | Close / 30-day MA |
| `rsi` | 14-period RSI (0–100) |

### Technical — MACD (12/26/9)
| Feature | Description |
|---------|-------------|
| `macd_line_ratio` | (EMA12 − EMA26) / Close — raw momentum divergence |
| `macd_signal_ratio` | EMA9(macd_line) / Close — smoothed trend direction |
| `macd_hist_ratio` | (line − signal) / Close — momentum acceleration; sign change = trend flip |

### Technical — Bollinger Bands (20-day, 2σ)
| Feature | Description |
|---------|-------------|
| `bb_width` | (upper − lower) / mid — volatility regime; squeeze precedes breakout |
| `bb_pct` | (Close − lower) / (upper − lower) — ≈0 oversold, ≈1 overbought |

### Technical — Return / Volume
| Feature | Description |
|---------|-------------|
| `daily_return` | Day-over-day % return |
| `volatility` | 10-day rolling std of returns |
| `log_volume` | log(1 + Volume) |

### Calendar / Seasonality
| Feature | Description |
|---------|-------------|
| `month` | 1–12 |
| `day_of_week` | 0=Mon … 4=Fri |
| `quarter` | 1–4 |
| `is_monsoon` | Jun–Sep flag |
| `is_winter` | Nov–Jan flag |
| `is_summer` | Mar–May flag |
| `month_sin/cos` | Cyclical month encoding |
| `day_of_week_sin/cos` | Cyclical weekday encoding |

## Target Labels

| Signal | Integer | Condition |
|--------|---------|-----------|
| SELL | 0 | next-day return < -1% |
| HOLD | 1 | -1% ≤ next-day return ≤ +1% |
| BUY | 2 | next-day return > +1% |

## Loading in your model script

```python
from pipeline import load_dataset
from feature_engineering import FEATURE_COLUMNS

df = load_dataset()
X = df[FEATURE_COLUMNS]
y = df["signal"]
```

## Scaling note
`log_volume`, RSI, MA ratios, and cyclical features are already bounded.
`daily_return` and `volatility` should be `StandardScaler`-transformed **inside**
an sklearn `Pipeline`, **after** your train/test split, to prevent leakage.
