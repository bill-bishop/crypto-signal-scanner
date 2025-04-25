# runner/backtest_runner.py

import argparse
import json
import pandas as pd
from model.baseline import predict_next
from eval.backtest import backtest_one

def load_series_dict(logfile_path: str, hours: int) -> dict[str, pd.Series]:
    with open(logfile_path, "r") as f:
        data = [json.loads(line) for line in f]

    df = pd.json_normalize(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.index = df.index.tz_localize("UTC")


    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    df = df[df.index >= cutoff]

    series_dict = {}
    for col in df.columns:
        if col.startswith("prices.") and col.endswith(".usd"):
            coin = col.split(".")[1]
            series = df[col].dropna()
            series = series.resample("1min").mean().interpolate()
            series_dict[coin] = series

    return series_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=72, help="How many hours of history to backtest")
    parser.add_argument("--horizon", choices=["1h", "1d"], default="1h", help="Prediction horizon")
    parser.add_argument("--method", choices=["last", "mean", "linear", "momentum_wave"], default="linear", help="Prediction method")
    args = parser.parse_args()

    print(f"📥 Loading last {args.hours} hours of data...")
    series_dict = load_series_dict("crypto_prices_log.jsonl", hours=args.hours)

    print(f"\n🔁 Backtesting {args.method} predictor with horizon = {args.horizon}...\n")

    results = []

    for coin, series in series_dict.items():
        df = backtest_one(series, horizon=args.horizon, method=args.method)
        if not df.empty:
            mae = df["error"].abs().mean()
            hit_rate = df["hit"].mean()
            results.append((coin, len(df), mae, hit_rate))

    results_df = pd.DataFrame(results, columns=["Coin", "Points", "MAE", "Hit Rate"])
    results_df = results_df.sort_values("Hit Rate", ascending=False)

    print(results_df.to_string(index=False, float_format="%.4f"))

if __name__ == "__main__":
    main()
