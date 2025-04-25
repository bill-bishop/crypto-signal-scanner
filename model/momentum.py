import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from model.projection import linear_projection

def classify_trend(series: pd.Series, threshold=0.0) -> pd.Series:
    diff = series.diff()
    return diff.apply(lambda x: 'up' if x > threshold else 'down')

def find_peaks_and_valleys(series: pd.Series, distance: int = 10) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    peaks, _ = find_peaks(series.values, distance=distance)
    valleys, _ = find_peaks(-series.values, distance=distance)
    return list(series.index[peaks]), list(series.index[valleys])

def momentum_wave_predict(series: pd.Series, horizon: str = '1h') -> float:
    """
    Custom prediction logic:
    - Start with linear regression forecast
    - Adjust based on recent trend reversals and peak/valley proximity
    """
    series = series.dropna().resample('1min').mean().interpolate()
    if len(series) < 10:
        return 0.0

    base_pred = linear_projection(series, horizon)

    # Step 1: Classify trend
    trend = classify_trend(series)
    recent_trend = trend[-5:]  # Last 5 minutes

    # Step 2: Detect recent reversal
    reversals = recent_trend.ne(recent_trend.shift()).sum()
    last_trend = recent_trend.iloc[-1]

    # Step 3: Find distance to last peak/valley
    peaks, valleys = find_peaks_and_valleys(series)
    if not peaks or not valleys:
        return base_pred

    last_time = series.index[-1]
    last_peak = max([t for t in peaks if t <= last_time], default=None)
    last_valley = max([t for t in valleys if t <= last_time], default=None)

    time_since_turn = min(
        (last_time - last_peak).seconds / 60 if last_peak else float('inf'),
        (last_time - last_valley).seconds / 60 if last_valley else float('inf')
    )

    # Step 4: Adjust prediction based on wave phase
    boost = 1.0
    if reversals >= 2:
        if last_trend == 'up':
            boost += 0.5  # reinforce uptrend
        elif last_trend == 'down':
            boost -= 0.5  # suppress false recovery

    # Step 5: Proximity to peak/valley (wave compression)
    if time_since_turn < 5:
        boost *= 0.5  # we're near a turning point, reduce overconfidence

    return base_pred * boost