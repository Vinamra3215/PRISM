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
# Calculate Volume Explosion Score
# ==========================

day_df["AvgVolume20"] = day_df["volume"].rolling(20).mean()

day_df["VES"] = day_df["volume"] / day_df["AvgVolume20"]

# ==========================
# Plot
# ==========================

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=day_df["timestamp"],
        y=day_df["VES"],
        mode="lines",
        name="Volume Explosion Score"
    )
)

fig.add_hline(
    y=1,
    line_dash="dash",
    line_color="green",
    annotation_text="Normal Volume"
)

fig.add_hline(
    y=2,
    line_dash="dash",
    line_color="red",
    annotation_text="High Volume"
)

fig.update_layout(
    title=f"Feature E: Volume Explosion Score ({selected_day})",
    xaxis_title="Time",
    yaxis_title="VES",
    template="plotly_white",
    width=1200,
    height=700
)

# ==========================
# Save
# ==========================

os.makedirs("plots", exist_ok=True)

fig.write_html("plots/featureE_volume_explosion.html")

fig.show()

print("Saved: plots/featureE_volume_explosion.html")
