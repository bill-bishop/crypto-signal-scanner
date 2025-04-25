# runner/data_health_runner.py

import argparse
import json
import pandas as pd
from model.data_health import compute_health

def load_dataframe(logfile: str = "crypto_prices_log.jsonl") -> pd.DataFrame:
    with open(logfile, "r") as f:
        lines = [json.loads(line) for line in f]

    df = pd.json_normalize(lines)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24, help="How many hours back to check")
    args = parser.parse_args()

    df = load_dataframe()
    health_df = compute_health(df, hours=args.hours)

    print(f"\n📊 Data Health Report (last {args.hours} hours)\n")
    print(health_df.to_string(index=False))

if __name__ == "__main__":
    main()
