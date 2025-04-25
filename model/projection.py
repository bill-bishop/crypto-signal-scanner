import numpy as np
import pandas as pd

def linear_projection(window: pd.Series, horizon: str = "1h") -> float:
    """
    Simple linear extrapolation of a price series.
    Returns predicted % change over the given horizon.
    """
    if len(window) < 2:
        return 0.0

    x = np.arange(len(window))
    y = window.values
    coef = np.polyfit(x, y, 1)  # linear fit
    next_val = coef[0] * (x[-1] + 1) + coef[1]
    current_price = y[-1]
    return (next_val - current_price) / current_price
