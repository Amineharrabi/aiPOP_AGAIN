# AION-EWS: Early Warning System for AI Sector Crashes

An autonomous research agent that collects free data (prices, Reddit, Google News RSS, Google Trends), builds 6-hour features, labels crash events for 1/3/7-day horizons, trains/calibrates models, and produces calibrated probabilities and backtest artifacts.

## Watch the Youtube Video :
https://www.youtube.com/watch?v=WbJNk43Xml4


<img width="640" height="420" alt="Brillo(4)" src="https://github.com/user-attachments/assets/3e22f3ab-86cb-4143-bbf0-aad6da8d0aaf" />

## Quickstart

1) Python 3.10+ recommended.
2) Install dependencies:

```
pip install -r requirements.txt
```

3) Run the full pipeline (defaults in `aion_ews/config.py`):

```
python run_pipeline.py
```

Artifacts are saved under `outputs/` and `data/intermediate/`. Final deliverables:


<img width="180" height="90" alt="2025-07" src="https://github.com/user-attachments/assets/a9cce175-8d99-4b06-996e-a1c1a1bcce35" />  <img width="180" height="90" alt="2024-02" src="https://github.com/user-attachments/assets/2220162a-cbe9-4e72-9729-93cf7f9aa2c8" />




- CSV: `outputs/predictions_latest.csv` (timestamp, features, labels, predicted probabilities)
- PNG: `outputs/backtest_latest.png`
- JSON: `outputs/run_latest.json` with shape {timestamp, crash_prob_1d, crash_prob_3d, crash_prob_7d, top_drivers, calibration}

## Config

You can create an optional `aion_ews_config.yaml` at the project root to override defaults. Example:

```
start_date: '2022-01-01'
end_date: null![Uploading 2025-07.png…]()

resample: '6H'
ai_tickers: [NVDA, MSFT, AMD, AVGO, GOOGL, META, TSLA, PLTR, SMCI, INTC]
baseline_tickers: [QQQ, SPY]
subreddits: [stocks, investing, wallstreetbets, technology, MachineLearning, ArtificialIntelligence]
crash:
  horizons_days: [1, 3, 7]
  drop_threshold: 0.07
  y_window_days: 3
model:
  cv_splits: 5
  calibrate_method: 'sigmoid'
  validation_frac: 0.2
```

## Notes
- Uses free data sources and libraries (yfinance, PRAW, Google News RSS, pytrends).
- Sentiment: tries a HuggingFace pipeline; if unavailable, falls back to VADER.
- All timestamps are normalized to UTC and aggregated to 6-hour bins.
- This is a research system; treat outputs as experimental and non-investment advice.
