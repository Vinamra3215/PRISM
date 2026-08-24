import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "HCLTECH_clean.parquet"
)

FORECAST_FILE = os.path.join(
    BASE_DIR,
    "results",
    "lora_full_forecast.parquet"
)

HISTORY_FILE = os.path.join(
    BASE_DIR,
    "results",
    "training_history.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "plots",
    "HCLTECH_training_prediction.html"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

CUTOFF_DATE = pd.Timestamp("2022-01-01")

CONTEXT_LENGTH = 400


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


# ============================================================
# LOAD ORIGINAL CLEAN DATA
# ============================================================

print("=" * 70)
print("LOADING HCLTECH DATA")
print("=" * 70)

data = pd.read_parquet(DATA_FILE)

data["Date"] = pd.to_datetime(
    data["Date"],
    errors="coerce"
)

data = data.dropna(
    subset=["Date", "Close"]
)

data = data.sort_values(
    "Date"
)

data = data.drop_duplicates(
    subset=["Date"],
    keep="last"
)

print("Total clean rows:", len(data))
print("Data start:", data["Date"].min())
print("Data end:", data["Date"].max())


# ============================================================
# LAST 400 TRADING DAYS BEFORE 2022-01-01
# ============================================================

before_cutoff = data[
    data["Date"] < CUTOFF_DATE
].copy()

if len(before_cutoff) < CONTEXT_LENGTH:
    raise ValueError(
        f"Only {len(before_cutoff)} rows available before "
        f"{CUTOFF_DATE.date()}, but {CONTEXT_LENGTH} are required."
    )

training = before_cutoff.tail(
    CONTEXT_LENGTH
).copy()

training_start = training["Date"].min()
training_end = training["Date"].max()

print()
print("=" * 70)
print("TRAINING / CONTEXT PERIOD")
print("=" * 70)

print("Number of trading days:", len(training))
print("Training start:", training_start)
print("Training end:", training_end)


# ============================================================
# LOAD KRONOS + LoRA FORECAST
# ============================================================

print()
print("=" * 70)
print("LOADING KRONOS + LoRA FORECAST")
print("=" * 70)

forecast = pd.read_parquet(
    FORECAST_FILE
)

# The forecast file uses Date as its index.
forecast = forecast.reset_index()

# Make sure the date column is called Date.
if "Date" not in forecast.columns:

    # Sometimes reset_index creates "index".
    if "index" in forecast.columns:
        forecast = forecast.rename(
            columns={"index": "Date"}
        )

    else:
        raise ValueError(
            "Could not find Date in forecast file."
        )


forecast["Date"] = pd.to_datetime(
    forecast["Date"],
    errors="coerce"
)

forecast = forecast.dropna(
    subset=["Date"]
)

forecast = forecast.sort_values(
    "Date"
)

forecast = forecast.drop_duplicates(
    subset=["Date"],
    keep="last"
)


print("Forecast rows:", len(forecast))
print("Forecast start:", forecast["Date"].min())
print("Forecast end:", forecast["Date"].max())


# ============================================================
# KEEP ONLY REQUIRED PREDICTION PERIOD
#
# 1-Jan-2022 -> latest available forecast date
# ============================================================

prediction = forecast[
    forecast["Date"] >= CUTOFF_DATE
].copy()

if len(prediction) == 0:
    raise ValueError(
        "No prediction rows found from 2022-01-01 onward."
    )


print()
print("=" * 70)
print("PREDICTION PERIOD")
print("=" * 70)

print("Prediction rows:", len(prediction))
print("Prediction start:", prediction["Date"].min())
print("Prediction end:", prediction["Date"].max())


# ============================================================
# ACTUAL CLOSE
# ============================================================

prediction["Actual"] = pd.to_numeric(
    prediction["actual_close"],
    errors="coerce"
)

prediction["Prediction"] = pd.to_numeric(
    prediction["predicted_close"],
    errors="coerce"
)


# ============================================================
# BASELINE
#
# Naive baseline:
# predict every future close as the LAST observed
# close of the 400-day training/context period.
# ============================================================

last_training_close = float(
    training["Close"].iloc[-1]
)

prediction["Baseline"] = last_training_close


print()
print("Last training close:", last_training_close)


# ============================================================
# DIVERGENCE
#
# Actual - Kronos + LoRA prediction
# ============================================================

prediction["Divergence"] = (
    prediction["Actual"]
    - prediction["Prediction"]
)


# ============================================================
# LOAD TRAINING HISTORY
# ============================================================

history = pd.read_csv(
    HISTORY_FILE
)

required_history_columns = [
    "epoch",
    "training_loss",
    "s1_loss",
    "s2_loss"
]

for col in required_history_columns:

    if col not in history.columns:
        raise ValueError(
            f"Missing column '{col}' in training_history.csv"
        )

history["epoch"] = pd.to_numeric(
    history["epoch"],
    errors="coerce"
)

history["training_loss"] = pd.to_numeric(
    history["training_loss"],
    errors="coerce"
)

history["s1_loss"] = pd.to_numeric(
    history["s1_loss"],
    errors="coerce"
)

history["s2_loss"] = pd.to_numeric(
    history["s2_loss"],
    errors="coerce"
)

history = history.dropna(
    subset=["epoch"]
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL EXPERIMENT SUMMARY")
print("=" * 70)

print()
print("TRAINING / CONTEXT")
print("------------------")
print("Days:", len(training))
print("Start:", training_start.date())
print("End:", training_end.date())

print()
print("PREDICTION")
print("----------")
print("Start:", prediction["Date"].min().date())
print("End:", prediction["Date"].max().date())
print("Rows:", len(prediction))

print()
print("TRAINING LOSS")
print("-------------")
print(
    "First:",
    history["training_loss"].iloc[0]
)

print(
    "Last:",
    history["training_loss"].iloc[-1]
)


# ============================================================
# CREATE ONE INTERACTIVE FIGURE
#
# ROW 1:
#   400 training days
#   Actual future
#   Kronos + LoRA prediction
#   Baseline
#
# ROW 2:
#   Divergence
#
# ROW 3:
#   Training loss
# ============================================================

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=False,
    vertical_spacing=0.08,
    row_heights=[
        0.60,
        0.20,
        0.20
    ],
    subplot_titles=[
        "HCLTECH — 400-Day Training Context + 2022→Latest Prediction",
        "Prediction Divergence: Actual − Kronos + LoRA",
        "LoRA Training Loss"
    ]
)


# ============================================================
# ROW 1 — TRAINING PERIOD
# ============================================================

fig.add_trace(
    go.Scatter(
        x=training["Date"],
        y=training["Close"],
        mode="lines",
        name="Training Close (400 days)",
        line=dict(
            width=2
        ),
        hovertemplate=(
            "<b>Training</b><br>"
            "Date: %{x|%Y-%m-%d}<br>"
            "Close: %{y:.2f}"
            "<extra></extra>"
        )
    ),
    row=1,
    col=1
)


# ============================================================
# ROW 1 — ACTUAL FUTURE
# ============================================================

fig.add_trace(
    go.Scatter(
        x=prediction["Date"],
        y=prediction["Actual"],
        mode="lines",
        name="Actual Close",
        line=dict(
            width=2
        ),
        connectgaps=False,
        hovertemplate=(
            "<b>Actual</b><br>"
            "Date: %{x|%Y-%m-%d}<br>"
            "Close: %{y:.2f}"
            "<extra></extra>"
        )
    ),
    row=1,
    col=1
)


# ============================================================
# ROW 1 — KRONOS + LoRA
# ============================================================

fig.add_trace(
    go.Scatter(
        x=prediction["Date"],
        y=prediction["Prediction"],
        mode="lines",
        name="Kronos + LoRA",
        line=dict(
            width=2,
            dash="dash"
        ),
        hovertemplate=(
            "<b>Kronos + LoRA</b><br>"
            "Date: %{x|%Y-%m-%d}<br>"
            "Prediction: %{y:.2f}"
            "<extra></extra>"
        )
    ),
    row=1,
    col=1
)


# ============================================================
# ROW 1 — BASELINE
# ============================================================

fig.add_trace(
    go.Scatter(
        x=prediction["Date"],
        y=prediction["Baseline"],
        mode="lines",
        name="Baseline",
        line=dict(
            width=1.5,
            dash="dot"
        ),
        hovertemplate=(
            "<b>Baseline</b><br>"
            "Date: %{x|%Y-%m-%d}<br>"
            "Value: %{y:.2f}"
            "<extra></extra>"
        )
    ),
    row=1,
    col=1
)


# ============================================================
# TRAIN / PREDICTION SEPARATOR
# ============================================================

fig.add_vline(
    x=CUTOFF_DATE.timestamp() * 1000,
    line_width=2,
    line_dash="dash",
    row=1,
    col=1
)

fig.add_annotation(
    x=CUTOFF_DATE,
    y=1,
    yref="paper",
    text="Prediction starts: 1-Jan-2022",
    showarrow=False,
    xanchor="left",
    yanchor="top"
)


# ============================================================
# ROW 2 — DIVERGENCE
# ============================================================

fig.add_trace(
    go.Scatter(
        x=prediction["Date"],
        y=prediction["Divergence"],
        mode="lines",
        name="Divergence",
        line=dict(
            width=1.5
        ),
        hovertemplate=(
            "<b>Divergence</b><br>"
            "Date: %{x|%Y-%m-%d}<br>"
            "Actual − Prediction: %{y:.2f}"
            "<extra></extra>"
        )
    ),
    row=2,
    col=1
)


# Zero line for divergence

fig.add_hline(
    y=0,
    line_width=1,
    line_dash="dot",
    row=2,
    col=1
)


# ============================================================
# ROW 3 — TRAINING LOSS
# ============================================================

fig.add_trace(
    go.Scatter(
        x=history["epoch"],
        y=history["training_loss"],
        mode="lines+markers",
        name="Training Loss",
        line=dict(
            width=2
        ),
        hovertemplate=(
            "<b>Training Loss</b><br>"
            "Epoch: %{x}<br>"
            "Loss: %{y:.4f}"
            "<extra></extra>"
        )
    ),
    row=3,
    col=1
)


# ============================================================
# S1 LOSS
# ============================================================

fig.add_trace(
    go.Scatter(
        x=history["epoch"],
        y=history["s1_loss"],
        mode="lines+markers",
        name="S1 Loss",
        line=dict(
            width=1.5,
            dash="dash"
        ),
        hovertemplate=(
            "<b>S1 Loss</b><br>"
            "Epoch: %{x}<br>"
            "Loss: %{y:.4f}"
            "<extra></extra>"
        )
    ),
    row=3,
    col=1
)


# ============================================================
# S2 LOSS
# ============================================================

fig.add_trace(
    go.Scatter(
        x=history["epoch"],
        y=history["s2_loss"],
        mode="lines+markers",
        name="S2 Loss",
        line=dict(
            width=1.5,
            dash="dot"
        ),
        hovertemplate=(
            "<b>S2 Loss</b><br>"
            "Epoch: %{x}<br>"
            "Loss: %{y:.4f}"
            "<extra></extra>"
        )
    ),
    row=3,
    col=1
)


# ============================================================
# AXIS LABELS
# ============================================================

fig.update_yaxes(
    title_text="HCLTECH Price",
    row=1,
    col=1
)

fig.update_yaxes(
    title_text="Divergence",
    row=2,
    col=1
)

fig.update_yaxes(
    title_text="Loss",
    row=3,
    col=1
)

fig.update_xaxes(
    title_text="Date",
    row=1,
    col=1
)

fig.update_xaxes(
    title_text="Date",
    row=2,
    col=1
)

fig.update_xaxes(
    title_text="Epoch",
    row=3,
    col=1
)


# ============================================================
# RANGE SLIDER
# ============================================================

fig.update_xaxes(
    rangeslider=dict(
        visible=True
    ),
    row=1,
    col=1
)


# ============================================================
# LAYOUT
# ============================================================

fig.update_layout(
    title=(
        "HCLTECH — Kronos + LoRA<br>"
        "<sup>"
        "400 Trading Days Training Context → "
        "1-Jan-2022 to Latest Prediction"
        "</sup>"
    ),

    height=1100,

    hovermode="x unified",

    template="plotly_white",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),

    margin=dict(
        l=70,
        r=40,
        t=130,
        b=60
    )
)


# ============================================================
# SAVE
# ============================================================

fig.write_html(
    OUTPUT_FILE,
    include_plotlyjs=True
)

print()
print("=" * 70)
print("PLOT CREATED SUCCESSFULLY")
print("=" * 70)

print()
print("FILE:")
print(OUTPUT_FILE)

print()
print("Open this HTML file in your browser.")