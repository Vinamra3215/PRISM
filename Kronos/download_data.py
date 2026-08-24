from datetime import datetime
import pandas as pd
import requests

TICKER = "%5ENSEI"  # URL-encoded '^NSEI'
START_DATE = "2023-01-01"
END_DATE = "2026-08-01"
OUTPUT_FILE = "nifty_data.csv"

# Convert dates to Unix timestamps
period1 = int(datetime.strptime(START_DATE, "%Y-%m-%d").timestamp())
period2 = int(datetime.strptime(END_DATE, "%Y-%m-%d").timestamp())

url = f"https://query1.finance.yahoo.com/v8/finance/chart/{TICKER}?period1={period1}&period2={period2}&interval=1d"

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

print(f"Fetching ^NSEI data directly from Yahoo REST API...")

response = requests.get(url, headers=headers)

if response.status_code != 200:
    raise RuntimeError(
        f"API request failed with status code {response.status_code}: {response.text}"
    )

data = response.json()

try:
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, unit="s").date,
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        }
    )

    # Clean missing/null days (market holidays)
    df = df.dropna().reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(
        f"Successfully downloaded {len(df)} records. Saved to '{OUTPUT_FILE}'."
    )
    print("\nData Preview:")
    print(df.head())

except (KeyError, IndexError) as e:
    raise RuntimeError(
        f"Failed to parse API response structure: {e}\nFull Response: {data}"
    )