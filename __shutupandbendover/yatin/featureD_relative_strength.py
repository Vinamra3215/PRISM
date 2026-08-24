import pandas as pd
import matplotlib.pyplot as plt

# ===========================
# Read RELIANCE Data
# ===========================

rel = pd.read_parquet("RELIANCE_5Y_OHLCVdata.parquet")
rel.index = pd.to_datetime(rel.index)

# ===========================
# Read NIFTY Data
# ===========================

nifty = pd.read_parquet("../NIFTY50_5Y_OHLCV.parquet")
nifty.index = pd.to_datetime(nifty.index)

# Extract Close Price
nifty_close = nifty[("Close", "^NSEI")]

# Create DataFrame
data = pd.DataFrame()
data["Reliance_Close"] = rel["Close"]
data["Nifty_Close"] = nifty_close

# Relative Strength
data["RS"] = data["Reliance_Close"] / data["Nifty_Close"]

# Select Date Range
start_date = "2024-01-01"
end_date = "2024-06-30"

selected = data.loc[start_date:end_date]

# Plot
plt.figure(figsize=(12,6))

plt.plot(selected.index, selected["RS"], linewidth=2)

plt.title("Relative Strength (RELIANCE / NIFTY)")
plt.xlabel("Date")
plt.ylabel("RS")

plt.grid(True)

plt.savefig("plots/featureD_relative_strength.png")

plt.show()

# Print Last Values
print(selected[[
    "Reliance_Close",
    "Nifty_Close",
    "RS"
]].tail())