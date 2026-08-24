import pandas as pd
import matplotlib.pyplot as plt

# ===========================
# Read Dataset
# ===========================
df = pd.read_parquet("RELIANCE_5Y_OHLCVdata.parquet")

# Convert index to datetime
df.index = pd.to_datetime(df.index)

# ===========================
# Select Date Range
# ===========================
start_date = "2024-01-01"
end_date = "2024-06-30"

df = df.loc[start_date:end_date]

# ===========================
# Calculate Daily Return
# ===========================
df["Return"] = df["Close"].pct_change()

# ===========================
# 10-Day Volatility
# ===========================
df["Volatility10"] = df["Return"].rolling(window=10).std()

# ===========================
# 20-Day Average Volatility
# ===========================
df["AvgVolatility20"] = df["Volatility10"].rolling(window=20).mean()

# ===========================
# Volatility Expansion Score
# ===========================
df["VES"] = df["Volatility10"] / df["AvgVolatility20"]

# Remove NaN values
df = df.dropna()

# ===========================
# Plot Graph
# ===========================
plt.figure(figsize=(12,6))

plt.plot(df.index, df["VES"], linewidth=2)

# Normal volatility line
plt.axhline(y=1, color="red", linestyle="--")

plt.title("Volatility Expansion Score (VES)")
plt.xlabel("Date")
plt.ylabel("VES")
plt.grid(True)

plt.savefig("plots/featureE_ves.png")
plt.show()

# ===========================
# Print Last Values
# ===========================
print(df[[
    "Close",
    "Return",
    "Volatility10",
    "AvgVolatility20",
    "VES"
]].tail())