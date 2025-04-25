import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from datetime import timedelta
import matplotlib.ticker as tickr
import json

N = 3  # Hours of data to display
REFRESH_INTERVAL = 15  # Seconds between updates

def classify_trend(x, threshold=0):
    if x > threshold:
        return 'up'
    elif x < -threshold:
        return 'down'
    else:
        return 'flat'

last_seen_timestamp = None

plt.ion()
fig, ax = plt.subplots(figsize=(14, 6))

coins = [
    'bitcoin',
    'ethereum',
    'litecoin',
    'binancecoin',
    'cardano',
    'solana',
    'dogecoin',
    'shibainu',
    'polygon',
    'chainlink'
]

while True:
    with open('crypto_prices_log.jsonl') as f:
        data = [json.loads(line) for line in f]

    df = pd.json_normalize(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    if 'prices.bitcoin.usd' not in df.columns:
        print("No BTC data yet.")
        continue

    btc = df['prices.bitcoin.usd'].dropna()
    latest_timestamp = btc.index.max()

    if last_seen_timestamp == latest_timestamp:
        print(f"No new data — waiting... (last seen: {latest_timestamp})")
        plt.pause(REFRESH_INTERVAL)
        continue

    last_seen_timestamp = latest_timestamp

    coin_series = {}
    for coin in coins:
        col = f'prices.{coin}.usd'
        if col in df.columns:
            resampled = df[col].dropna().resample('1min').mean().interpolate()
            coin_series[coin] = resampled

    end_time = latest_timestamp
    start_time = end_time - timedelta(hours=N)

    sliced_coins = {}
    for coin, series in coin_series.items():
        sliced = series[start_time:end_time]
        if not sliced.empty:
            normed = (sliced / sliced.iloc[0]) - 1
            sliced_coins[coin] = normed

    btc_slice = sliced_coins['bitcoin']
    peaks, _ = find_peaks(btc_slice, distance=10)
    valleys, _ = find_peaks(-btc_slice, distance=10)

    velocity = btc_slice.diff()
    acceleration = velocity.diff()

    trend_df = pd.DataFrame({
        'price': btc_slice,
        'momentum': velocity,
        'acceleration': acceleration
    })
    trend_df['trend'] = trend_df['momentum'].apply(classify_trend)
    trend_df['trend_shift'] = trend_df['trend'].ne(trend_df['trend'].shift())

    # === Correlation Scoring ===
    btc_returns = btc_slice.diff().dropna()
    correlation_results = []

    for coin, series in sliced_coins.items():
        if coin == 'bitcoin':
            continue

        coin_returns = series.diff().dropna()
        combined = pd.DataFrame({'btc': btc_returns, 'alt': coin_returns}).dropna()

        same_direction = (combined['btc'] * combined['alt']) > 0
        agreement_rate = same_direction.sum() / len(combined)

        btc_std = combined['btc'][same_direction].std()
        alt_std = combined['alt'][same_direction].std()
        amplification = (alt_std / btc_std) if btc_std > 0 else 0

        score = agreement_rate * amplification

        correlation_results.append({
            'coin': coin,
            'agreement': round(agreement_rate * 100, 1),
            'amplification': round(amplification, 2),
            'score': round(score, 2),
        })

    correlation_results.sort(key=lambda x: x['score'], reverse=True)
    best_tracking_coin = correlation_results[0]['coin']

    print("\n🔍 BTC Echo Ranking:")
    for r in correlation_results:
        print(f"{r['coin'].capitalize():<12} | Agreement: {r['agreement']:>5.1f}% | Amplification: {r['amplification']:>4.2f}x | Score: {r['score']:>4.2f}")

    # === Plotting ===
    ax.clear()
    ax.yaxis.set_major_formatter(tickr.PercentFormatter(xmax=1.0))

    for coin, series in sliced_coins.items():
        if coin == 'bitcoin':
            continue

        style = {'linewidth': 1.2, 'alpha': 0.4}
        if coin == best_tracking_coin:
            style = {'linewidth': 2.5, 'alpha': 0.9, 'linestyle': '--'}

        ax.plot(series.index, series.values, label=coin.capitalize(), **style)

    # Plot BTC itself
    ax.plot(btc_slice.index, btc_slice.values, label='Bitcoin', color='blue', linewidth=2.5, zorder=5)
    ax.plot(btc_slice.rolling(5).mean(), label='Smoothed BTC', color='gray', linestyle='--')

    ax.plot(btc_slice.index[peaks], btc_slice.iloc[peaks], 'g^', label='BTC Peaks')
    ax.plot(btc_slice.index[valleys], btc_slice.iloc[valleys], 'rv', label='BTC Valleys')

    ax.scatter(trend_df[trend_df['trend'] == 'up'].index,
               trend_df[trend_df['trend'] == 'up']['price'],
               color='limegreen', s=30, label='BTC Uptrend', zorder=3)
    ax.scatter(trend_df[trend_df['trend'] == 'down'].index,
               trend_df[trend_df['trend'] == 'down']['price'],
               color='red', s=30, label='BTC Downtrend', zorder=3)
    ax.scatter(trend_df[trend_df['trend_shift']].index,
               trend_df[trend_df['trend_shift']]['price'],
               color='purple', s=50, marker='x', label='BTC Trend Shift')

    ax.set_title(f'Crypto % Change — Last {N} Hours (Updated: {end_time:%Y-%m-%d %H:%M:%S})')
    ax.legend(loc='upper left', ncol=2, fontsize='small')
    plt.tight_layout()
    plt.pause(REFRESH_INTERVAL)
