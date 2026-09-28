import yfinance as yf
from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TICKER = "^NSEI"

START_DATE = "2022-01-01"
END_DATE = None

OUTPUT_PATH = Path(__file__).resolve().parent / "nifty_ohlcv.parquet"


# ---------------------------------------------------------
# Download
# ---------------------------------------------------------

print(f"Downloading {TICKER} data...")
print(f"Start: {START_DATE}")
print(f"End: {END_DATE or 'latest'}")

data = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,
    progress=True,
)


# ---------------------------------------------------------
# Clean columns
# ---------------------------------------------------------

if data.empty:
    raise RuntimeError("No data was downloaded.")


# yfinance can return MultiIndex columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)


data.columns = [
    str(column).lower()
    for column in data.columns
]


# Keep only the OHLCV columns
required_columns = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]

missing = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}\n"
        f"Available columns: {list(data.columns)}"
    )


data = data[required_columns].copy()

data = data.dropna()

data.index = data.index.tz_localize(None)

data = data.sort_index()


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

data.to_parquet(OUTPUT_PATH)

print()
print("Download complete.")
print(f"Rows: {len(data)}")
print(f"From: {data.index.min()}")
print(f"To:   {data.index.max()}")
print(f"Saved to: {OUTPUT_PATH}")