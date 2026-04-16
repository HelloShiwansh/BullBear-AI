# BullBear AI – Stock Trend Predictor

An end-to-end machine learning system that predicts **Buy / Sell / Hold signals** for the top 20 NIFTY stocks by combining technical indicators, calendar seasonality, and news sentiment analysis.

## Features

- **Data Pipeline**: Automated OHLCV data fetching, preprocessing, feature engineering, and target labeling
- **Sentiment Analysis**: FinBERT-based sentiment scoring from financial news
- **Model Training**: Classical ML (Random Forest, XGBoost, LightGBM) and LSTM neural networks
- **Evaluation**: Time-series cross-validation, backtesting, and performance metrics
- **Live Prediction**: Real-time signal generation for all 20 NIFTY stocks
- **Interactive Dashboard**: Streamlit-based web interface for predictions and analysis

## Quick Start

### Installation

```bash
# Clone or download the project
cd "BullBear AI - Stock Trend Predictor"

# Create virtual environment
python -m venv bb_venv
# On Windows:
bb_venv\Scripts\activate
# On Unix/Mac:
# source bb_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Complete Pipeline

```bash
# Generate the ML-ready dataset
python pipeline.py

# Train all models (classical + LSTM)
python train_classical.py
python train_lstm.py

# Evaluate and compare all models
python evaluate.py

# Launch the prediction dashboard
streamlit run dashboard.py
```

## Project Structure

```
├── data/
│   ├── raw/           # Raw OHLCV data from Yahoo Finance
│   ├── processed/     # Final ML-ready dataset
│   ├── news/          # Financial news and sentiment scores
│   ├── models/        # Trained model files and configs
│   └── results/       # Evaluation metrics and backtest results
├── build_docs/        # Project documentation
├── dashboard.py       # Streamlit web interface
├── pipeline.py        # Data preparation orchestrator
├── train_classical.py # Classical ML training
├── train_lstm.py      # LSTM neural network training
├── evaluate.py        # Model evaluation and backtesting
├── predict.py         # Live prediction engine
├── sentiment.py       # News sentiment analysis
└── requirements.txt   # Python dependencies
```

## Data Pipeline

### 1. Data Fetching (`data_fetch.py`)
Downloads historical OHLCV data for the top 20 NIFTY stocks from Yahoo Finance.

**Stocks Covered:**
ASIANPAINT.NS, AXISBANK.NS, BAJFINANCE.NS, BHARTIARTL.NS, HCLTECH.NS, HDFCBANK.NS, HINDUNILVR.NS, ICICIBANK.NS, INFY.NS, ITC.NS, KOTAKBANK.NS, LT.NS, MARUTI.NS, RELIANCE.NS, SBIN.NS, SUNPHARMA.NS, TCS.NS, TITAN.NS, ULTRACEMCO.NS, WIPRO.NS

### 2. Preprocessing (`preprocess.py`)
- Data validation and cleaning
- Forward-fill small gaps
- Sort by date
- Remove invalid entries

### 3. Feature Engineering (`feature_engineering.py`)
**21 Technical + Calendar Features:**

#### Technical Indicators
- **Trend**: MA ratios (10-day, 30-day), RSI (14-period)
- **MACD**: Line ratio, signal ratio, histogram ratio
- **Bollinger Bands**: Width, position percentage
- **Volume/Returns**: Daily returns, volatility (10-day), log volume

#### Calendar/Seasonal Features
- Month, day of week, quarter
- Seasonal flags (monsoon, winter, summer)
- Cyclical encodings (sin/cos for month and weekday)

### 4. Target Creation (`target.py`)
**Buy/Sell/Hold Labels** based on next-day returns:
- **SELL (0)**: Return < -1%
- **HOLD (1)**: -1% ≤ Return ≤ +1%
- **BUY (2)**: Return > +1%

### 5. Sentiment Analysis (`sentiment.py`)
- FinBERT model for financial news sentiment
- Daily sentiment aggregation
- Optional feature addition (25 total features with sentiment)

## Model Training

### Classical ML Models
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting with regularization
- **LightGBM**: Microsoft's high-performance gradient boosting

### LSTM Neural Network
- **Sequence Length**: 30-day windows
- **Architecture**: 2 LSTM layers (128 hidden), dropout 0.3, FC layers (128→64→3)
- **Training**: Early stopping, Adam optimizer

### Training Configurations
Each model trained **twice** for ablation study:
1. **With Sentiment**: 25 features (full set)
2. **Without Sentiment**: 21 features (technical + calendar only)

## Evaluation & Backtesting

### Metrics
- F1 Macro Score (primary metric)
- Precision, Recall, Accuracy
- Confusion Matrix
- Classification Report

### Backtesting
- Simulated trading vs. buy-and-hold benchmark
- Time-series walk-forward validation
- Sharpe ratio, maximum drawdown, total return

### Results Summary
Models are evaluated and compared automatically. The best performing model (highest F1 macro) is selected as production model.

### Model Performance Comparison

| Model | Sentiment | F1 Macro | Accuracy | Strategy Return | Buy&Hold Return | Outperformance |
|-------|-----------|----------|----------|-----------------|-----------------|---------------|
| **Random Forest** | Without | **0.377** | **0.512** | **38.1%** | 5.0% | **33.1%** |
| Random Forest | With | 0.374 | 0.513 | 35.8% | 5.0% | 30.8% |
| XGBoost | Without | 0.375 | 0.479 | 32.4% | 5.0% | 27.4% |
| XGBoost | With | 0.369 | 0.482 | 29.7% | 5.0% | 24.7% |
| LightGBM | Without | 0.368 | 0.472 | 28.9% | 5.0% | 23.9% |
| LightGBM | With | 0.368 | 0.476 | 31.2% | 5.0% | 26.2% |
| LSTM | With | 0.329 | 0.589 | 15.6% | 5.0% | 10.6% |
| LSTM | Without | 0.320 | 0.550 | 12.3% | 5.0% | 7.3% |

**Best Model**: Random Forest without sentiment (F1 Macro: 0.377, Accuracy: 0.512, Outperformance: 33.1%)

## Live Prediction

### Usage
```bash
# Predict all 20 stocks
python predict.py

# Predict specific stock
python predict.py --ticker TCS.NS
```

### Features
- Fetches latest market data
- Generates real-time signals
- 30-day history for context
- Production Random Forest model

## Dashboard

### Launch
```bash
streamlit run dashboard.py
```

### Features
- Interactive predictions for all stocks
- Real-time signal visualization
- Historical performance charts
- Model confidence scores
- News sentiment integration

## Dependencies

Key packages:
- `yfinance`: Data fetching
- `pandas`, `numpy`: Data processing
- `ta`: Technical analysis
- `scikit-learn`: Classical ML
- `xgboost`, `lightgbm`: Gradient boosting
- `torch`: Deep learning
- `transformers`: FinBERT sentiment
- `streamlit`: Web dashboard

## Model Files

Trained models saved in `data/models/`:
- `best_model_classical.pkl`: Production Random Forest
- `scaler.pkl`: Fitted StandardScaler
- `lstm_with_sentiment.pt`: LSTM with sentiment
- `lstm_without_sentiment.pt`: LSTM without sentiment

## Results

Evaluation results in `data/results/`:
- Model performance metrics (JSON)
- Backtest results (JSON)
- Feature importance rankings (CSV)
- Training curves (PNG for LSTM)

## Usage as Library

```python
# Load dataset
from pipeline import load_dataset
from feature_engineering import FEATURE_COLUMNS

df = load_dataset()
X = df[FEATURE_COLUMNS]
y = df['signal']

# Load trained model
from predict import load_model
model, scaler = load_model()

# Make predictions
from predict import predict_all
signals, histories = predict_all()
```

## Scaling Notes

- Features like RSI, MA ratios, and cyclical encodings are pre-bounded
- `daily_return` and `volatility` require `StandardScaler` inside sklearn Pipeline
- **Important**: Fit scaler on training data only to prevent data leakage

## Contributing

This is a complete ML pipeline for stock trend prediction. For modifications:
1. Maintain time-series integrity (no future data leakage)
2. Use TimeSeriesSplit for cross-validation
3. Test on unseen data before deployment
4. Update documentation for any architectural changes

## License

Academic/research use only. Not for live trading without extensive validation.
