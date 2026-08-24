import yfinance as yf
import os

symbol = "RELIANCE.NS"

df = yf.download(
    symbol,
    start="2019-01-01",
    end="2027-01-01",
    auto_adjust=False,
    progress=True
)

if df.empty:
    raise RuntimeError("No data downloaded.")

os.makedirs("data", exist_ok=True)

output = "data/RIL_19-26.parquet"
df.to_parquet(output)

print(f"\nSaved to: {output}")
print(f"Rows: {len(df)}")
print(f"Start: {df.index.min()}")
print(f"End:   {df.index.max()}")
print(df.head())
print(df.tail())