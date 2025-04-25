import requests
import time
import json
from datetime import datetime

API_URL = "https://api.coingecko.com/api/v3/simple/price"
PARAMS = {
    "ids": "bitcoin,ethereum,litecoin,binancecoin,cardano,solana,dogecoin,shibainu,polygon,chainlink",
    "vs_currencies": "usd"
}

LOG_FILE = "crypto_prices_log.jsonl"

while True:
    try:
        response = requests.get(API_URL, params=PARAMS, timeout=10)
        response.raise_for_status()
        prices = response.json()

        timestamp = datetime.utcnow().isoformat()
        data_point = {'timestamp': timestamp, 'prices': prices}

        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(data_point) + '\n')

        print(f"Logged at {timestamp}")
        print(prices)

    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"[ERROR] {datetime.utcnow().isoformat()} - {e}")
        print("Waiting 30 seconds before retrying...\n")
        time.sleep(30)
        continue

    time.sleep(60)
