import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet("/home/soq/NIFTY50_5Y_OHLCV.parquet")

print(df.head())

print(df.columns)

print(df.info())

plt.figure(figsize=(14,6))

plt.plot(df.index, df[("Close", "^NSEI")])

plt.title("NIFTY Close Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.grid(True)

plt.savefig("close_price.png")
plt.show()
print("Graph saved!")
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

plt.savefig(os.path.join(script_dir, "close_price.png"))


df["Log_Return"] = np.log(
    df[("Close", "^NSEI")] /
    df[("Close", "^NSEI")].shift(1)
)

plt.figure(figsize=(15,6))

plt.plot(df.index, df["Log_Return"], color="blue")

plt.title("NIFTY Log Return")
plt.xlabel("Time")
plt.ylabel("Log Return")

plt.grid(True)

plt.savefig("/home/soq/bhoomika/log_return.png")

print("Log Return graph saved!")


import plotly.graph_objects as go

# Reliance data
stock = pd.read_parquet("/home/soq/RELIANCE_5Y_OHLCV.parquet")

# NIFTY 50 data
nifty = pd.read_parquet("/home/soq/NIFTY50_5Y_OHLCV.parquet")

# Close prices
stock_close = stock[("Close", "RELIANCE")]
nifty_close = nifty[("Close", "^NSEI")]

# Log returns
stock_return = np.log(stock_close / stock_close.shift(1))
nifty_return = np.log(nifty_close / nifty_close.shift(1))

# Combine and align dates
data = pd.DataFrame({
    "Stock_Return": stock_return,
    "NIFTY_Return": nifty_return
}).dropna()

# 60-day rolling beta
data["Beta"] = (
    data["Stock_Return"].rolling(60).cov(data["NIFTY_Return"])
    /
    data["NIFTY_Return"].rolling(60).var()
)