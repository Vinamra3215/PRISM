import os
import pandas as pd
import plotly.graph_objects as go

# ==========================
# Read Data
# ==========================

df = pd.read_parquet("data/parquet_data/RELIANCE-EQ_1min.parquet")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# ==========================
# Choose one trading day
# ==========================

selected_day = "2024-02-13"

day_df = df[df["timestamp"].dt.strftime("%Y-%m-%d") == selected_day].copy()

print("Rows:", len(day_df))

# ==========================
# Calculate VWM
# ==========================

day_df["price_change"] = day_df["close"].diff()

day_df["VWM"] = day_df["price_change"] * day_df["volume"]

# ==========================
# Plot
# ==========================

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=day_df["timestamp"],
        y=day_df["VWM"],
        mode="lines",
        name="Volume Weighted Momentum"
    )
)

fig.add_hline(
    y=0,
    line_dash="dash",
    line_color="red"
)

fig.update_layout(
    title=f"Feature B: Volume Weighted Momentum ({selected_day})",
    xaxis_title="Time",
    yaxis_title="VWM",
    template="plotly_white",
    width=1200,
    height=700
)

# ==========================
# Save
# ==========================

os.makedirs("plots", exist_ok=True)

fig.write_html("plots/featureB_vwm.html")

fig.show()

print("Saved: plots/featureB_vwm.html")
