import numpy as np
import pandas as pd
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# PATHS
# ============================================================

CSV_PATH = Path("lora_predictions.csv")

METRICS_PATH = Path("lora_metrics_same_as_baseline.json")

HTML_PATH = Path("lora_forecast_same_metrics.html")


# ============================================================
# LOAD CSV
# ============================================================

results = pd.read_csv(CSV_PATH)

results["date"] = pd.to_datetime(results["date"])

results = results.sort_values("date")
results = results.set_index("date")


# ============================================================
# METRICS
# EXACTLY THE SAME AS BASELINE SCRIPT
# ============================================================

actual = results["actual_close"].to_numpy(dtype=float)

predicted = results["predicted_close"].to_numpy(dtype=float)

error = actual - predicted


# ------------------------------------------------------------
# RMSE
# ------------------------------------------------------------

rmse = np.sqrt(
    np.mean(np.square(error))
)


# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------

mae = np.mean(
    np.abs(error)
)


# ------------------------------------------------------------
# MAPE
# ------------------------------------------------------------

mape = np.mean(
    np.abs(error)
    / np.maximum(np.abs(actual), 1e-8)
) * 100


# ------------------------------------------------------------
# R²
# ------------------------------------------------------------

ss_res = np.sum(
    np.square(error)
)

ss_tot = np.sum(
    np.square(actual - actual.mean())
)

r2 = (
    1 - ss_res / ss_tot
    if ss_tot != 0
    else 0
)


# ------------------------------------------------------------
# DIRECTION ACCURACY
# ------------------------------------------------------------

actual_change = np.diff(actual)

predicted_change = (
    predicted[1:]
    - actual[:-1]
)

actual_direction = np.sign(
    actual_change
)

predicted_direction = np.sign(
    predicted_change
)

direction_accuracy = (
    np.mean(
        actual_direction == predicted_direction
    ) * 100
)


# ------------------------------------------------------------
# STRATEGY RETURN
# ------------------------------------------------------------

actual_returns = (
    np.diff(actual)
    / np.maximum(
        np.abs(actual[:-1]),
        1e-8
    )
)

strategy_positions = np.sign(
    predicted_change
)

strategy_returns = (
    strategy_positions
    * actual_returns
)

strategy_growth = np.prod(
    1 + strategy_returns
)

strategy_return = (
    strategy_growth - 1
) * 100


# ------------------------------------------------------------
# PEARSON
# ------------------------------------------------------------

if (
    len(actual) > 1
    and np.std(actual) > 0
    and np.std(predicted) > 0
):

    pearson = np.corrcoef(
        actual,
        predicted
    )[0, 1]

else:

    pearson = 0.0


# ============================================================
# METRICS DICTIONARY
# ============================================================

metrics = {
    "RMSE": float(rmse),
    "MAE": float(mae),
    "MAPE": float(mape),
    "R2": float(r2),
    "Direction Accuracy": float(direction_accuracy),
    "Return (% change)": float(strategy_return),
    "Pearson Correlation": float(pearson),
}


# ============================================================
# SAVE METRICS
# ============================================================

with open(METRICS_PATH, "w") as f:
    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# PRINT METRICS
# ============================================================

print("=" * 70)
print("METRICS")
print("=" * 70)

print(f"RMSE                  : {rmse:,.2f}")
print(f"MAE                   : {mae:,.2f}")
print(f"MAPE                  : {mape:.2f}%")
print(f"R2                    : {r2:.4f}")
print(f"Direction Accuracy    : {direction_accuracy:.2f}%")
print(f"Return (% change)     : {strategy_return:.2f}%")
print(f"Pearson Correlation   : {pearson:.4f}")


# ============================================================
# CREATE PLOT
# ============================================================

figure = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.40, 0.40, 0.20],
    subplot_titles=(
        "Actual Daily OHLC",
        "Kronos + LoRA Predicted Daily OHLC",
        "Close Price Difference (Actual - Predicted)",
    ),
)


# ============================================================
# ACTUAL CANDLES
# ============================================================

figure.add_trace(
    go.Candlestick(
        x=results.index,
        open=results["actual_open"],
        high=results["actual_high"],
        low=results["actual_low"],
        close=results["actual_close"],
        name="Actual",
    ),
    row=1,
    col=1,
)


# ============================================================
# PREDICTED CANDLES
# ============================================================

figure.add_trace(
    go.Candlestick(
        x=results.index,
        open=results["predicted_open"],
        high=results["predicted_high"],
        low=results["predicted_low"],
        close=results["predicted_close"],
        name="Predicted",
    ),
    row=2,
    col=1,
)


# ============================================================
# DIFFERENCE
# ============================================================

difference = (
    results["actual_close"]
    - results["predicted_close"]
)

figure.add_trace(
    go.Scatter(
        x=results.index,
        y=difference,
        mode="lines",
        name="Actual - Predicted",
    ),
    row=3,
    col=1,
)


# Zero line

figure.add_hline(
    y=0,
    row=3,
    col=1,
    line_dash="dash",
)


# ============================================================
# METRICS HEADER
# ============================================================

metrics_text = (
    f"<b>Metrics</b>&nbsp;&nbsp;&nbsp;"
    f"RMSE: {rmse:,.2f}"
    f"&nbsp;&nbsp;|&nbsp;&nbsp;"
    f"MAE: {mae:,.2f}"
    f"&nbsp;&nbsp;|&nbsp;&nbsp;"
    f"MAPE: {mape:.2f}%"
    f"&nbsp;&nbsp;|&nbsp;&nbsp;"
    f"R²: {r2:.4f}"
    f"&nbsp;&nbsp;|&nbsp;&nbsp;"
    f"Direction Accuracy: {direction_accuracy:.2f}%"
    f"&nbsp;&nbsp;|&nbsp;&nbsp;"
    f"Return: {strategy_return:.2f}%"
    f"&nbsp;&nbsp;|&nbsp;&nbsp;"
    f"Pearson: {pearson:.4f}"
)


figure.add_annotation(
    text=metrics_text,
    xref="paper",
    yref="paper",
    x=0.5,
    y=1.045,
    xanchor="center",
    yanchor="bottom",
    showarrow=False,
    align="center",
    bgcolor="white",
    bordercolor="gray",
    borderwidth=1,
    font={"size": 12},
)


# ============================================================
# LAYOUT
# ============================================================

figure.update_layout(
    title=(
        "Kronos + LoRA: "
        "40-Day Rolling Daily OHLC Forecast"
    ),
    height=1100,
    template="plotly_white",
    hovermode="x unified",

    xaxis_rangeslider_visible=False,
    xaxis2_rangeslider_visible=False,
    xaxis3_rangeslider_visible=False,

    margin=dict(
        t=150,
        b=70,
        l=80,
        r=40,
    ),
)


# ============================================================
# AXIS LABELS
# ============================================================

figure.update_yaxes(
    title_text="Actual Price",
    row=1,
    col=1,
)

figure.update_yaxes(
    title_text="Predicted Price",
    row=2,
    col=1,
)

figure.update_yaxes(
    title_text="Difference",
    row=3,
    col=1,
)

figure.update_xaxes(
    title_text="Trading Day",
    row=3,
    col=1,
)


# ============================================================
# SAVE HTML
# ============================================================

figure.write_html(
    str(HTML_PATH),
    include_plotlyjs="cdn",
)

print("\nFiles generated:")
print(f"Metrics : {METRICS_PATH}")
print(f"HTML    : {HTML_PATH}")