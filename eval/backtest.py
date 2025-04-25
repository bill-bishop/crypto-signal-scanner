# eval/backtest.py

import pandas as pd
import numpy as np
from typing import Callable, Literal
from model.baseline import predict_next

Horizon = Literal["1h", "1d"]

def backtest_one(
    series: pd.Series,
    horizon: Horizon = "1h",
    method: str = "linear",
    step_minutes: int = 10,
    lookback_minutes: int = 60
) -> pd.DataFrame:
    """
    Backtest a single price series using a rolling prediction.

    Returns:
        DataFrame with columns: timestamp, prediction, actual, error, hit
    """
    freq = pd.Timedelta("1min")
    df = []

    timestamps = series.index
    delta = {"1h": pd.Timedelta("1h"), "1d": pd.Timedelta("1d")}[horizon]

    for i in range(lookback_minutes, len(series) - int(delta.total_seconds() / 60), step_minutes):
        window = series.iloc[i - lookback_minutes : i]
        current_time = window.index[-1]
        future_time = current_time + delta

        if future_time not in series.index:
            continue

        predicted_change = -1 * predict_next(window, horizon=horizon, method=method)
        future_price = series[future_time]
        current_price = window.iloc[-1]
        actual_change = (future_price - current_price) / current_price
        error = predicted_change - actual_change
        hit = int(np.sign(predicted_change) == np.sign(actual_change) and predicted_change != 0)

        df.append({
            "timestamp": current_time,
            "predicted": predicted_change,
            "actual": actual_change,
            "error": error,
            "hit": hit,
        })

    return pd.DataFrame(df)
