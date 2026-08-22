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
# Calculate Relative Strength
# ==========================

day_df["MA20"] = day_df["close"].rolling(window=20).mean()

day_df["RS"] = day_df["close"] / day_df["MA20"]

# ==========================
# Plot
# ==========================

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=day_df["timestamp"],
        y=day_df["RS"],
        mode="lines",
        name="Relative Strength"
    )
)

fig.add_hline(
    y=1,
    line_dash="dash",
    line_color="red"
)

fig.update_layout(
    title=f"Feature D: Relative Strength ({selected_day})",
    xaxis_title="Time",
    yaxis_title="Relative Strength",
    template="plotly_white",
    width=1200,
    height=700
)

# ==========================
# Save
# ==========================

os.makedirs("plots", exist_ok=True)

fig.write_html("plots/featureD_relative_strength.html")

fig.show()

print("Saved: plots/featureD_relative_strength.html")