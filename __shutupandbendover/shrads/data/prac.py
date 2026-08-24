import pandas as pd

df = pd.read_parquet("NIFTY50_5Y_OHLCV.parquet")
print(df.head())

print(df.tail())

print(df.info())

print(df.describe())

print(df.columns)