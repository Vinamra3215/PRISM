import pandas as pd

df = pd.read_parquet("RELIANCE_5Y_OHLCVdata.parquet")

print(df.columns)
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# Read Dataset
# ============================

df = pd.read_parquet("RELIANCE_5Y_OHLCVdata.parquet")

# Extract Close and Volume columns
close = df[("Close", "RELIANCE.NS")]
volume = df[("Volume", "RELIANCE.NS")]

# Create a clean dataframe
data = pd.DataFrame({
    "Close": close,
    "Volume": volume
})

# Convert index to datetime
data.index = pd.to_datetime(data.index)

# ============================
# Calculate Features
# ============================

# Daily Return
data["Return"] = data["Close"].pct_change()

# 10-Day Momentum
data["Momentum"] = (
    data["Close"] - data["Close"].shift(10)
) / data["Close"].shift(10)

# 10-Day Average Volume
data["AvgVolume"] = data["Volume"].rolling(10).mean()

# Volume Ratio
data["VolumeRatio"] = (
    data["Volume"] / data["AvgVolume"]
)

# Volume Weighted Momentum
data["VWM"] = (
    data["Momentum"] * data["VolumeRatio"]
)

# ============================
# Select Date Range
# ============================

start_date = "2024-01-01"
end_date = "2024-06-30"

selected = data.loc[start_date:end_date]

# ============================
# Plot
# ============================

plt.figure(figsize=(12,6))

plt.plot(
    selected.index,
    selected["VWM"],
    linewidth=2
)

plt.title("Volume Weighted Momentum (VWM)")
plt.xlabel("Date")
plt.ylabel("VWM")

plt.grid(True)

plt.savefig("plots/featureB_vwm.png")

plt.show()

# ============================
# Print Last Values
# ============================

print(
    selected[
        [
            "Close",
            "Volume",
            "Momentum",
            "VolumeRatio",
            "VWM"
        ]
    ].tail()
)