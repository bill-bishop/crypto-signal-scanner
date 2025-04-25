# Crypto Signal Scanner

A modular framework for ingesting, visualizing, and analyzing cryptocurrency price signals — with baseline forecasting, backtesting, and directional correlation modeling across coins.

---

## Features

- Real-time price ingestion from CoinGecko
- Per-minute charting & trend detection (peaks, valleys, momentum)
- Signal-based model backtesting (MAE, hit rate)
- Directional correlation analysis (echo matrix)
- Predictive modeling with:
  - Linear regression
  - Rolling mean
  - Momentum-phase wave detection
- Model inversion testing for poor-signal detection

---

## Project Structure

```
project-root/
├── model/               # Forecasting models & signal logic
│   ├── baseline.py      # Simple predictors: last, mean, linear
│   ├── momentum.py      # Trend + wave-aware forecasting
│   ├── frequency.py     # Cycle period detection for trend shift timing
│   ├── projection.py    # Linear regression utility
│   └── data_health.py   # Timespan coverage diagnostics
│
├── runners/             # Executable scripts
│   ├── scan.py          # Collects per-minute price data
│   ├── load.py          # Visualizes live price + signals
│   ├── baseline_runner.py
│   ├── data_health_runner.py
│   └── backtest_runner.py
│
└── eval/
    └── backtest.py      # MAE & directional hit rate benchmarking
```

---

## 🚀 Quickstart

1. Clone the repo:
   ```bash
   git clone https://github.com/yourname/crypto-signal-scanner.git
   cd crypto-signal-scanner
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start price ingestion:
   ```bash
   python -m runners.scan
   ```

4. Visualize trends:
   ```bash
   python -m runners.load
   ```

5. Backtest a prediction strategy:
   ```bash
   python -m runners.backtest_runner --method linear --horizon 1h --hours 24
   ```

---

## Predictors Available

| Method         | Description                          |
|----------------|--------------------------------------|
| `last`         | No change from latest price          |
| `mean`         | Mean of past N prices                |
| `linear`       | Linear regression projection         |
| `momentum_wave`| Trend shifts + wave phase awareness  |

---

## Next Goals

- Add neural predictors / regressors
- Auto-tune parameters per coin
- Signal ensembling & reinforcement voting
- Trade simulator with slippage + fees

---

## 📜 License

MIT License
