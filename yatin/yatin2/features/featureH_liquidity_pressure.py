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
# NORMALIZED LIQUIDITY PRESSURE INDEX (NLPI)
# ==========================================================

# Close Return
day_df["return"] = day_df["close"].pct_change()

# 20-period EMA of Volume
day_df["ema_volume"] = (
    day_df["volume"]
    .ewm(span=20, adjust=False)
    .mean()
)

# Relative Volume
day_df["relative_volume"] = (
    day_df["volume"] /
    day_df["ema_volume"]
)

# Normalized Liquidity Pressure Index
day_df["NLPI"] = (
    day_df["return"] *
    day_df["relative_volume"] *
    100
)

day_df["NLPI"] = day_df["NLPI"].fillna(0)

feature_mean = day_df["NLPI"].mean()

# ==========================================================
# PLOT
# ==========================================================

fig = make_subplots(
    specs=[[{"secondary_y": True}]]
)

# Close Price
fig.add_trace(

    go.Scatter(

        x=day_df["timestamp"],
        y=day_df["close"],

        name="Close Price",

        line=dict(
            color="royalblue",
            width=2
        )

    ),

    secondary_y=False

)

# NLPI
fig.add_trace(

    go.Scatter(

        x=day_df["timestamp"],
        y=day_df["NLPI"],

        name="NLPI",

        line=dict(
            color="darkorange",
            width=2
        )

    ),

    secondary_y=True

)

# Mean Line
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

        name=f"Mean NLPI ({feature_mean:.3f})",

        line=dict(
            color="red",
            dash="dash",
            width=2
        )

    ),

    secondary_y=True

)

# ==========================================================
# LAYOUT
# ==========================================================

fig.update_layout(

    title=f"RELIANCE : Normalized Liquidity Pressure Index ({selected_day})",

    template="plotly_white",

    width=1300,

    height=700,

    hovermode="x unified"

)

fig.update_xaxes(title="Time")

fig.update_yaxes(
    title_text="Close Price (₹)",
    secondary_y=False
)

fig.update_yaxes(
    title_text="NLPI",
    secondary_y=True
)

# ==========================================================
# SAVE
# ==========================================================

os.makedirs("plots", exist_ok=True)

fig.write_html(
    "plots/featureH_NLPI.html"
)

fig.show()

print("Rows :", len(day_df))
print("Mean NLPI :", round(feature_mean, 4))
print("Saved : plots/featureH_NLPI.html")