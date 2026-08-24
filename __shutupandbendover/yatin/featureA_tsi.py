import pandas as pd
import matplotlib.pyplot as plt

# ============================
# Read Dataset
# ============================

df = pd.read_parquet("RELIANCE_5Y_OHLCVdata.parquet")

# Convert index to datetime
df.index = pd.to_datetime(df.index)

# ============================
# Feature Engineering
# ============================

# Daily Return
df["Return"] = df["Close"].pct_change()

# 10-Day Momentum (Percentage)
df["Momentum"] = (
    df["Close"] - df["Close"].shift(10)
) / df["Close"].shift(10)

# 10-Day Volatility
df["Volatility"] = df["Return"].rolling(window=10).std()

# Trend Stability Index (TSI)
df["TSI"] = df["Momentum"] / df["Volatility"]

# ============================
# Select Date Range
# ============================

start_date = "2024-01-01"
end_date = "2024-06-30"

selected_data = df.loc[start_date:end_date]

# ============================
# Plot Graph
# ============================

plt.figure(figsize=(12,6))

plt.plot(
    selected_data.index,
    selected_data["TSI"],
    linewidth=2
)

plt.title("Trend Stability Index (TSI)\nRELIANCE (2024-01-01 to 2024-06-30)")

plt.xlabel("Date")

plt.ylabel("TSI")

plt.grid(True)

plt.savefig("plots/featureA_tsi.png")

plt.show()

# ============================
# Display Last Few Values
# ============================

print(selected_data[["Close","Momentum","Volatility","TSI"]].tail())