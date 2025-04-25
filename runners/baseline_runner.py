# runner/baseline_runner.py

import json
import argparse
import pandas as pd
from datetime import datetime, timedelta
from model.baseline import predict_all

LOGFILE = "crypto_prices_log.jsonl"

def load_series_dict(logfile_path: str, hours: int) -> dict[str, pd.Series]:
    with open(logfile_path, 'r') as f:
        data = [json.loads(line) for line in f]

    df = pd.json_normalize(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    cutoff = datetime.utcnow() - timedelta(hours=hours)
    df = df[df.index >= cutoff]

    # Explode into separate series per coin
    series_dict = {}
    for col in df.columns:
        if col.startswith('prices.') and col.endswith('.usd'):
            coin = col.split('.')[1]
            series = df[col].dropna()
            series = series.resample('1min').mean().interpolate()
            series_dict[coin] = series

    return series_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=48, help="How many hours of history to use")
    args = parser.parse_args()

    print(f"📥 Loading last {args.hours}h of data...")
    series_dict = load_series_dict(LOGFILE, hours=args.hours)

    print("\n🔮 Predicting next-hour % change...")
    next_hour = predict_all(series_dict, horizon='1h', method='linear')

    print("\n📆 Predicting next-day % change...")
    next_day = predict_all(series_dict, horizon='1d', method='linear')

    print("\n=== Predicted Returns ===")
    print(f"{'Coin':<10} {'Next 1h':>10} {'Next 1d':>10}")
    print("-" * 34)
    for coin in sorted(series_dict.keys()):
        h1 = next_hour.get(coin, 0)
        d1 = next_day.get(coin, 0)
        print(f"{coin:<10} {h1:>9.2%} {d1:>9.2%}")

if __name__ == "__main__":
    main()
