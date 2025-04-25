# model/baseline.py

import pandas as pd
import numpy as np
from typing import Literal
from model.momentum import momentum_wave_predict
from model.projection import linear_projection

Horizon = Literal["1h", "1d"]
Method = Literal["last", "mean", "linear"]

def predict_next(prices: pd.Series, horizon: Horizon = "1h", method: Method = "linear") -> float:
    """
    Predicts % change from current price for a given coin.

    Args:
        prices: pd.Series with datetime index and price values
        horizon: "1h" or "1d"
        method: prediction strategy to use

    Returns:
        Predicted % change (e.g. 0.02 = +2%)
    """
    freq = prices.index.freq or prices.index.inferred_freq
    if freq is None:
        raise ValueError("Time series has no frequency set or inferable.")

    step = {"1h": 60, "1d": 1440}[horizon]
    num_points = step // pd.Timedelta(freq).seconds * 60

    # Restrict to recent window
    window = prices[-num_points:]

    if len(window) < 2:
        return 0.0

    current_price = window.iloc[-1]

    if method == "last":
        return 0.0

    elif method == "mean":
        mean_price = window.mean()
        return (mean_price - current_price) / current_price

    elif method == "linear":
        return linear_projection(window, horizon=horizon)

    elif method == "momentum_wave":
        return momentum_wave_predict(window, horizon=horizon)

    else:
        raise ValueError(f"Unknown method: {method}")

def predict_all(series_dict: dict[str, pd.Series],
                horizon: Horizon = "1h",
                method: Method = "linear") -> dict[str, float]:
    """
    Predicts next % change for all coins in the dict.
    """
    return {
        coin: predict_next(prices, horizon=horizon, method=method)
        for coin, prices in series_dict.items()
        if not prices.empty
    }
