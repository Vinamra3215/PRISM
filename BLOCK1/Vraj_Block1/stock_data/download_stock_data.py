import yfinance as yf
import pandas as pd

# 1. Download 5 years of daily data for Reliance (NSE)
print("Downloading data from yfinance...")
df = yf.download(tickers="INFY.NS", period="5y", interval="1d")

# 2. Flatten MultiIndex columns (yfinance recently changed its output format)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 3. Filter only the requested OHLCV columns (dropping 'Adj Close')
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# 4. Reset the index so the Date/Datetime becomes a standard column
df = df.reset_index()

# Rename the first column (which yfinance names 'Date' or 'Datetime') to 'Timestamp'
df.rename(columns={df.columns[0]: 'timestamps'}, inplace=True)
df.columns = df.columns.str.lower()

# 5. Remove the NSE timezone information (tz-aware to tz-naive)
print("Formatting timestamps and removing timezone data...")
df['timestamps'] = pd.to_datetime(df['timestamps']).dt.tz_localize(None)

# Optional: Clean up column names to ensure they are standard strings
df.columns = [str(col).strip() for col in df.columns]

# 6. Save the resulting DataFrame to a Parquet file
output_file = "infosys_5y_1d.parquet"
df.to_parquet(output_file, engine="pyarrow")

print(f"Success! Data formatted to OHLCV and saved as {output_file}")
print("\nPreview of the data:")
print(df.head())