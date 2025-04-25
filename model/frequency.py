import pandas as pd
from scipy.signal import find_peaks
import numpy as np

def estimate_cycle_frequency(series: pd.Series, distance: int = 10) -> dict:
    """
    Estimate the average frequency of peaks and valleys in a time series.
    Returns the average interval in minutes between peaks and valleys.
    """
    series = series.dropna().resample('1min').mean().interpolate()

    peaks, _ = find_peaks(series.values, distance=distance)
    valleys, _ = find_peaks(-series.values, distance=distance)

    peak_times = series.index[peaks]
    valley_times = series.index[valleys]

    def average_diff(timestamps):
        if len(timestamps) < 2:
            return None
        diffs = np.diff(timestamps.astype(np.int64) // 10**9)  # seconds
        return np.mean(diffs) / 60  # convert to minutes

    return {
        "peak_period": average_diff(peak_times),
        "valley_period": average_diff(valley_times),
        "last_peak": peak_times[-1] if len(peak_times) > 0 else None,
        "last_valley": valley_times[-1] if len(valley_times) > 0 else None
    }

def predict_next_trend_shift(series: pd.Series, distance: int = 10) -> pd.Timestamp:
    """
    Predict the next trend shift time based on the average period between peaks/valleys.
    Chooses the most recent extrema and extrapolates forward using the average interval.
    """
    cycle_info = estimate_cycle_frequency(series, distance=distance)
    last_time = series.index[-1]

    next_peak = (cycle_info["last_peak"] + pd.Timedelta(minutes=cycle_info["peak_period"])) if cycle_info["last_peak"] and cycle_info["peak_period"] else None
    next_valley = (cycle_info["last_valley"] + pd.Timedelta(minutes=cycle_info["valley_period"])) if cycle_info["last_valley"] and cycle_info["valley_period"] else None

    candidates = [t for t in [next_peak, next_valley] if t and t > last_time]
    return min(candidates) if candidates else None