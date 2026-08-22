import os
import pandas as pd
import plotly.graph_objects as go

# ==========================
# Read Data
# ==========================

df = pd.read_parquet("data/parquet_data/RELIANCE-EQ_1min.parquet")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# ==========================
# Select One Trading Day
# ==========================

selected_day = "2024-02-13"

day_df = df[df["timestamp"].dt.strftime("%Y-%m-%d") == selected_day].copy()

print("Rows:", len(day_df))

# ==========================
# Calculate VWAP
# ==========================

day_df["TypicalPrice"] = (
    day_df["high"] +
    day_df["low"] +
    day_df["close"]
) / 3

day_df["TPxV"] = day_df["TypicalPrice"] * day_df["volume"]

day_df["CumTPxV"] = day_df["TPxV"].cumsum()

day_df["CumVolume"] = day_df["volume"].cumsum()

day_df["VWAP"] = day_df["CumTPxV"] / day_df["CumVolume"]

# ==========================
# Plot
# ==========================

fig = go.Figure()

# Close Price
fig.add_trace(
    go.Scatter(
        x=day_df["timestamp"],
        y=day_df["close"],
        mode="lines",
        name="Close Price"
    )
)

# VWAP
fig.add_trace(
    go.Scatter(
        x=day_df["timestamp"],
        y=day_df["VWAP"],
        mode="lines",
        name="VWAP"
    )
)

fig.update_layout(
    title=f"Feature F : VWAP ({selected_day})",
    xaxis_title="Time",
    yaxis_title="Price",
    template="plotly_white",
    width=1200,
    height=700
)

# ==========================
# Save
# ==========================

os.makedirs("plots", exist_ok=True)

fig.write_html("plots/featureF_vwap.html")

fig.show()

print("Saved : plots/featureF_vwap.html")