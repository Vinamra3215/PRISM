import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
KRONOS_DIR = Path("/home/soq/Kronos")

DATA_PATH = PROJECT_DIR / "data" / "nifty_ohlcv.parquet"

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = RESULTS_DIR / "baseline_predictions.csv"
METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"
PLOT_PATH = RESULTS_DIR / "baseline_forecast.html"


# ============================================================
# CONFIG
# ============================================================

START_DATE = "2022-01-01"
CONTEXT_LENGTH = 40

MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

PRICE_COLUMNS = ["open", "high", "low", "close"]

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
    else "cpu"
)


# ============================================================
# KRONOS IMPORT
# ============================================================

sys.path.insert(0, str(KRONOS_DIR))

from model import Kronos, KronosPredictor, KronosTokenizer


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("KRONOS ZERO-SHOT BASELINE")
print("=" * 70)

print(f"\nDevice: {DEVICE}")
print(f"Kronos: {KRONOS_DIR}")
print(f"Data:   {DATA_PATH}")

data = pd.read_parquet(DATA_PATH)

data.index = pd.to_datetime(data.index)
data = data.sort_index()

required_columns = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]

missing = [
    column for column in required_columns
    if column not in data.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

data = data[required_columns].astype(float).dropna()

# Only start the experiment from 1 Jan 2022.
data = data[data.index >= START_DATE].copy()

if len(data) <= CONTEXT_LENGTH:
    raise ValueError(
        f"Not enough data. Need more than {CONTEXT_LENGTH} rows."
    )

print(f"\nRows available from {START_DATE}: {len(data)}")
print(f"Date range: {data.index[0].date()} -> {data.index[-1].date()}")


# ============================================================
# LOAD KRONOS
# ============================================================

print("\nLoading Kronos tokenizer...")

tokenizer = KronosTokenizer.from_pretrained(
    TOKENIZER_NAME
)

print("Loading Kronos model...")

model = Kronos.from_pretrained(
    MODEL_NAME
)

model = model.to(DEVICE)
model.eval()
tokenizer.eval()

print("Kronos loaded successfully.")


# ============================================================
# PREDICTOR
# ============================================================

predictor = KronosPredictor(
    model,
    tokenizer,
    device=DEVICE,
    max_context=512,
)


# ============================================================
# WALK-FORWARD PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("STARTING 40-DAY ROLLING WALK-FORWARD FORECAST")
print("=" * 70)

predictions = []

# First prediction:
#
# data[0:40] -> predict data[40]
#
# Then:
#
# data[0:41] -> predict data[41]
#
# etc.
#
# IMPORTANT:
# The target candle is NEVER included in its own context.

total_predictions = len(data) - CONTEXT_LENGTH

for i in range(CONTEXT_LENGTH, len(data)):

    context = data.iloc[i - CONTEXT_LENGTH:i].copy()

    actual = data.iloc[i]
    target_date = data.index[i]

    context_timestamps = pd.Series(context.index)
    target_timestamps = pd.Series([target_date])

    predicted = predictor.predict(
        context,
        context_timestamps,
        target_timestamps,
        pred_len=1,
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
    )

    predicted_candle = predicted.iloc[0]

    predictions.append(
        {
            "date": target_date,

            "actual_open": actual["open"],
            "actual_high": actual["high"],
            "actual_low": actual["low"],
            "actual_close": actual["close"],
            "actual_volume": actual["volume"],

            "predicted_open": predicted_candle["open"],
            "predicted_high": predicted_candle["high"],
            "predicted_low": predicted_candle["low"],
            "predicted_close": predicted_candle["close"],
            "predicted_volume": predicted_candle["volume"],
        }
    )

    completed = i - CONTEXT_LENGTH + 1

    if (
        completed == 1
        or completed % 25 == 0
        or completed == total_predictions
    ):
        print(
            f"Progress: {completed}/{total_predictions} "
            f"| Target: {target_date.date()}"
        )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

results = pd.DataFrame(predictions)
results["date"] = pd.to_datetime(results["date"])
results = results.set_index("date")

results.to_csv(PREDICTIONS_PATH)

print("\nPredictions saved:")
print(PREDICTIONS_PATH)


# ============================================================
# METRICS
# ============================================================

actual = results["actual_close"].to_numpy(dtype=float)
predicted = results["predicted_close"].to_numpy(dtype=float)

error = actual - predicted


# RMSE
rmse = np.sqrt(
    np.mean(np.square(error))
)


# MAE
mae = np.mean(
    np.abs(error)
)


# MAPE
mape = np.mean(
    np.abs(error) / np.maximum(np.abs(actual), 1e-8)
) * 100


# R²
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


# Direction Accuracy
#
# We compare the direction of today's actual
# movement with the direction implied by
# today's predicted close versus yesterday's actual close.

actual_change = np.diff(actual)

predicted_change = predicted[1:] - actual[:-1]

actual_direction = np.sign(actual_change)
predicted_direction = np.sign(predicted_change)

direction_accuracy = (
    np.mean(actual_direction == predicted_direction) * 100
)


# Strategy Return
#
# Position:
# +1 if model predicts price will rise
# -1 if model predicts price will fall
#
# Then apply that position to the ACTUAL next-day return.

actual_returns = (
    np.diff(actual)
    / np.maximum(np.abs(actual[:-1]), 1e-8)
)

strategy_positions = np.sign(predicted_change)

strategy_returns = (
    strategy_positions * actual_returns
)

strategy_growth = np.prod(
    1 + strategy_returns
)

strategy_return = (
    strategy_growth - 1
) * 100


# Pearson Correlation
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
# PRINT METRICS
# ============================================================

print("\n" + "=" * 70)
print("BASELINE METRICS")
print("=" * 70)

for name, value in metrics.items():

    if name in {
        "MAPE",
        "Direction Accuracy",
        "Return (% change)",
    }:
        print(f"{name:<25}: {value:.2f}%")

    elif name in {
        "R2",
        "Pearson Correlation",
    }:
        print(f"{name:<25}: {value:.4f}")

    else:
        print(f"{name:<25}: {value:,.2f}")


# Save metrics
import json

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\nMetrics saved: {METRICS_PATH}")


# ============================================================
# PLOTLY CANDLESTICK GRAPH
# ============================================================

print("\nCreating Plotly graph...")


figure = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.40, 0.40, 0.20],
    subplot_titles=(
        "Actual Daily OHLC",
        "Kronos Zero-Shot Predicted Daily OHLC",
        "Close Price Difference (Actual - Predicted)",
    ),
)


# ------------------------------------------------------------
# Actual candles
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Predicted candles
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Difference
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Metrics annotation
# ------------------------------------------------------------

# ------------------------------------------------------------
# Metrics annotation
# ------------------------------------------------------------

metrics_text = (
    f"<b>Metrics</b>&nbsp;&nbsp;&nbsp;"
    f"RMSE: {rmse:,.2f}&nbsp;&nbsp;|&nbsp;&nbsp;"
    f"MAE: {mae:,.2f}&nbsp;&nbsp;|&nbsp;&nbsp;"
    f"MAPE: {mape:.2f}%&nbsp;&nbsp;|&nbsp;&nbsp;"
    f"R²: {r2:.4f}&nbsp;&nbsp;|&nbsp;&nbsp;"
    f"Direction Accuracy: {direction_accuracy:.2f}%&nbsp;&nbsp;|&nbsp;&nbsp;"
    f"Return: {strategy_return:.2f}%&nbsp;&nbsp;|&nbsp;&nbsp;"
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


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

figure.update_layout(
    title=(
        "Kronos Zero-Shot Baseline: "
        "40-Day Rolling Daily OHLC Forecast"
    ),
    height=1100,
    template="plotly_white",
    hovermode="x unified",
    xaxis_rangeslider_visible=False,
    xaxis2_rangeslider_visible=False,
    xaxis3_rangeslider_visible=False,
    
    margin = dict(
        t = 150,
        b = 70,
        l = 80,
        r = 40,
    )
)


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


# Save interactive HTML
figure.write_html(
    str(PLOT_PATH),
    include_plotlyjs="cdn",
)

print(f"Plotly graph saved: {PLOT_PATH}")


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("BASELINE EXPERIMENT COMPLETE")
print("=" * 70)

print(f"""
Predictions : {PREDICTIONS_PATH}
Metrics     : {METRICS_PATH}
Plotly      : {PLOT_PATH}

Total predictions: {len(results)}
First target:      {results.index[0].date()}
Last target:       {results.index[-1].date()}
""")