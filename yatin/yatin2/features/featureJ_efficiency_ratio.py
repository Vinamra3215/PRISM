import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================================
# READ DATA
# ==========================================================

df = pd.read_parquet(
    "data/parquet_data/RELIANCE-EQ_1min.parquet"
)

df["timestamp"] = pd.to_datetime(df["timestamp"])

# ==========================================================
# SELECT ONE TRADING DAY
# ==========================================================

selected_day = "2024-02-13"

day_df = df[
    df["timestamp"].dt.strftime("%Y-%m-%d") == selected_day
].copy()

# ==========================================================
# CALCULATE KAUFMAN EFFICIENCY RATIO (ER)
# ==========================================================

window = 20

# Net Change
day_df["net_change"] = (
    day_df["close"] -
    day_df["close"].shift(window)
).abs()

# Total Change
day_df["total_change"] = (
    day_df["close"]
    .diff()
    .abs()
    .rolling(window=window)
    .sum()
)

# Efficiency Ratio
day_df["ER"] = (
    day_df["net_change"] /
    day_df["total_change"]
)

day_df["ER"] = day_df["ER"].fillna(0)

# ==========================================================
# FEATURE MEAN
# ==========================================================

feature_mean = day_df["ER"].mean()

# ==========================================================
# CREATE PLOT
# ==========================================================

fig = make_subplots(
    specs=[[{"secondary_y": True}]]
)

# -----------------------------
# Close Price
# -----------------------------

fig.add_trace(

    go.Scatter(

        x=day_df["timestamp"],
        y=day_df["close"],

        mode="lines",

        name="Close Price",

        line=dict(
            color="royalblue",
            width=2
        )

    ),

    secondary_y=False

)

# -----------------------------
# Efficiency Ratio
# -----------------------------

fig.add_trace(

    go.Scatter(

        x=day_df["timestamp"],
        y=day_df["ER"],

        mode="lines",

        name="Efficiency Ratio",

        line=dict(
            color="darkorange",
            width=2
        )

    ),

    secondary_y=True

)

# -----------------------------
# Mean Line
# -----------------------------

fig.add_trace(

    go.Scatter(

        x=[
            day_df["timestamp"].iloc[0],
            day_df["timestamp"].iloc[-1]
        ],

        y=[
            feature_mean,
            feature_mean
        ],

        mode="lines",

        name=f"Mean ER ({feature_mean:.3f})",

        line=dict(
            color="red",
            dash="dash",
            width=2
        )

    ),

    secondary_y=True

)

# ==========================================================
# AXES
# ==========================================================

fig.update_xaxes(
    title="Time"
)

fig.update_yaxes(
    title_text="Close Price (₹)",
    secondary_y=False
)

fig.update_yaxes(
    title_text="Efficiency Ratio",
    secondary_y=True,
    range=[0, 1]
)

# ==========================================================
# LAYOUT
# ==========================================================

fig.update_layout(

    title=f"RELIANCE : Kaufman Efficiency Ratio ({selected_day})",

    template="plotly_white",

    width=1300,
    height=700,

    hovermode="x unified"

)

# ==========================================================
# SAVE
# ==========================================================

os.makedirs("plots", exist_ok=True)

fig.write_html(
    "plots/featureJ_efficiency_ratio.html"
)

fig.show()

print("Rows :", len(day_df))
print("Mean ER :", round(feature_mean,4))
print("Saved : plots/featureJ_efficiency_ratio.html")