import pandas as pd
import matplotlib.pyplot as plt

# ====================================
# Read RELIANCE Data
# ====================================

rel = pd.read_parquet("RELIANCE_5Y_OHLCVdata.parquet")

rel.index = pd.to_datetime(rel.index)

# ====================================
# Read NIFTY Data
# ====================================

nifty = pd.read_parquet("../NIFTY50_5Y_OHLCV.parquet")

nifty.index = pd.to_datetime(nifty.index)

# Extract Close Price
nifty_close = nifty[("Close", "^NSEI")]

# ====================================
# Create DataFrame
# ====================================

data = pd.DataFrame()

data["Reliance_Close"] = rel["Close"]

data["Nifty_Close"] = nifty_close

# ====================================
# Daily Returns
# ====================================

data["Reliance_Return"] = data["Reliance_Close"].pct_change()

data["Nifty_Return"] = data["Nifty_Close"].pct_change()

# ====================================
# Rolling Beta (20 Days)
# ====================================

rolling_cov = (
    data["Reliance_Return"]
    .rolling(20)
    .cov(data["Nifty_Return"])
)

rolling_var = (
    data["Nifty_Return"]
    .rolling(20)
    .var()
)

data["Rolling_Beta"] = rolling_cov / rolling_var

# ====================================
# Select Date Range
# ====================================

start_date = "2024-01-01"
end_date = "2024-06-30"

selected = data.loc[start_date:end_date]

# ====================================
# Plot
# ====================================

plt.figure(figsize=(12,6))

plt.plot(
    selected.index,
    selected["Rolling_Beta"],
    linewidth=2
)

plt.title("Rolling Beta (20-Day)\nRELIANCE vs NIFTY")

plt.xlabel("Date")

plt.ylabel("Beta")

plt.grid(True)

plt.savefig("plots/featureC_rolling_beta.png")

plt.show()

# ====================================
# Print Last Values
# ====================================

print(selected[[
    "Reliance_Close",
    "Nifty_Close",
    "Rolling_Beta"
]].tail())