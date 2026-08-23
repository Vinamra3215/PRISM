import os
import pandas as pd
from curl_cffi import requests
import yfinance as yf

# 1. Target directory path
target_dir = os.path.expanduser("~/__shutupandbendover/devansh-swc-wala")
os.makedirs(target_dir, exist_ok=True)

# 2. Create an impersonating browser session with curl_cffi
session = requests.Session(impersonate="chrome")

# 3. Fetch data using yf.Ticker and the impersonated session
ticker_symbol = "RELIANCE.NS"
ticker = yf.Ticker(ticker_symbol, session=session)
df = ticker.history(start="2019-01-01", end="2020-01-01", auto_adjust=False)

# Fallback check
if df.empty:
    raise ValueError(f"Failed to fetch data for {ticker_symbol}. Check network/firewall.")

# 4. Standardize columns & reset date index
df.reset_index(inplace=True)
df.columns = [str(col).strip().lower() for col in df.columns]

# 5. Save to Parquet
output_path = os.path.join(target_dir, "reliance_2019.parquet")
df.to_parquet(output_path, engine="pyarrow", index=False)

print(f"Data successfully saved ({len(df)} rows) to: {output_path}")
print(df.head(3))