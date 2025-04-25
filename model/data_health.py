# model/data_health.py

import pandas as pd
from datetime import datetime, timedelta

def compute_health(df: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    """
    Evaluate data coverage for each coin in a given hour window.
    Returns a DataFrame with: coin, % filled, number of points, expected points
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    df = df[df.index >= cutoff]

    results = []
    expected_points = hours * 60  # 1-minute intervals

    for col in df.columns:
        if col.startswith("prices.") and col.endswith(".usd"):
            coin = col.split(".")[1]
            series = df[col].dropna()
            actual = len(series)
            fill_pct = actual / expected_points

            results.append({
                "coin": coin,
                "minutes_present": actual,
                "expected": expected_points,
                "fill_pct": round(fill_pct * 100, 2)
            })

    return pd.DataFrame(results).sort_values("fill_pct", ascending=False)
