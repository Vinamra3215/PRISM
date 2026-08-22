import pandas as pd
import plotly.graph_objects as go

# Read data
df = pd.read_parquet("data/parquet_data/RELIANCE-EQ_1min.parquet")

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# -----------------------------
# Choose one random trading day
# -----------------------------
random_day = "2024-02-13"

day_df = df[df["timestamp"].dt.strftime("%Y-%m-%d") == random_day].copy()

print("Rows:", len(day_df))

# -----------------------------
# Calculate TSI
# -----------------------------
price_change = day_df["close"].diff()

double_smoothed_pc = (
    price_change.ewm(span=25, adjust=False).mean()
    .ewm(span=13, adjust=False).mean()
)

double_smoothed_abs = (
    price_change.abs().ewm(span=25, adjust=False).mean()
    .ewm(span=13, adjust=False).mean()
)

day_df["TSI"] = 100 * double_smoothed_pc / double_smoothed_abs

# -----------------------------
# Plot
# -----------------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=day_df["timestamp"],
        y=day_df["TSI"],
        mode="lines",
        name="TSI"
    )
)

fig.add_hline(
    y=0,
    line_dash="dash",
    line_color="red"
)

fig.update_layout(
    title=f"TSI for RELIANCE ({random_day})",
    xaxis_title="Time",
    yaxis_title="TSI",
    template="plotly_white"
)

# Save
fig.write_html("plots/featureA_tsi_day.html")

print("Saved successfully!")