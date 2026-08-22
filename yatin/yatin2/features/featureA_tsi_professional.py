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

print("Rows :", len(day_df))

# ==========================================================
# CALCULATE TSI
# ==========================================================

price_change = day_df["close"].diff()

ema1 = price_change.ewm(span=25, adjust=False).mean()
ema2 = ema1.ewm(span=13, adjust=False).mean()

abs_change = price_change.abs()

abs_ema1 = abs_change.ewm(span=25, adjust=False).mean()
abs_ema2 = abs_ema1.ewm(span=13, adjust=False).mean()

day_df["TSI"] = 100 * (ema2 / abs_ema2)

# ==========================================================
# TSI DAILY MEAN
# ==========================================================

tsi_mean = day_df["TSI"].mean()

# ==========================================================
# PLOT
# ==========================================================

fig = make_subplots(
    specs=[[{"secondary_y": True}]]
)

# -----------------------
# Close Price
# -----------------------

fig.add_trace(

    go.Scatter(

        x=day_df["timestamp"],

        y=day_df["close"],

        mode="lines",

        name="Close Price",

        line=dict(color="#0f5c8c", width=2)

    ),

    secondary_y=False

)

# -----------------------
# TSI
# -----------------------

fig.add_trace(

    go.Scatter(

        x=day_df["timestamp"],

        y=day_df["TSI"],

        mode="lines",

        name="TSI",

        line=dict(color="#d97706", width=2)

    ),

    secondary_y=True

)

# -----------------------
# Mean Line
# -----------------------

fig.add_trace(

    go.Scatter(

        x=[
            day_df["timestamp"].iloc[0],
            day_df["timestamp"].iloc[-1]
        ],

        y=[tsi_mean, tsi_mean],

        mode="lines",

        name=f"TSI Mean ({tsi_mean:.2f})",

        line=dict(
            color="red",
            width=2,
            dash="dash"
        )

    ),

    secondary_y=True

)

# ==========================================================
# AXIS
# ==========================================================

fig.update_xaxes(

    title_text="Time"

)

fig.update_yaxes(

    title_text="Close Price",

    secondary_y=False

)

fig.update_yaxes(

    title_text="TSI",

    secondary_y=True

)

# ==========================================================
# LAYOUT
# ==========================================================

fig.update_layout(

    title=f"RELIANCE : True Strength Index (TSI) ({selected_day})",

    template="plotly_white",

    height=650,

    legend=dict(

        orientation="h",

        y=1.08,

        x=0

    )

)

# ==========================================================
# SAVE
# ==========================================================

os.makedirs("plots", exist_ok=True)

fig.write_html(

    "plots/featureA_tsi_professional.html"

)

fig.show()

print("Saved : plots/featureA_tsi_professional.html")