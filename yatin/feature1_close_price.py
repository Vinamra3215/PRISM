import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_parquet("../NIFTY50_5Y_OHLCV.parquet")

# Display available columns
print(df.columns)

# Plot Close Price
plt.figure(figsize=(12,6))

plt.plot(df[("Close","^NSEI")])

plt.title("NIFTY50 Close Price")

plt.xlabel("Date")

plt.ylabel("Close Price")

plt.grid(True)

plt.savefig("plots/nifty_close_price.png")

plt.show()