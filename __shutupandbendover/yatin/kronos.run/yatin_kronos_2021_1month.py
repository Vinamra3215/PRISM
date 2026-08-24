import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import torch

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model import Kronos, KronosTokenizer, KronosPredictor


# ============================================================
# 1. EXPERIMENT SETTINGS
# ============================================================

TICKER = "RELIANCE.NS"

# Your selected starting point
FORECAST_START = "2021-01-01"

# Maximum lookback requested by mentor
LOOKBACK = 400

# Predict the next 400 trading periods
PRED_LEN = 400

# Download enough history before and after 2021
DATA_START = "2019-01-01"
DATA_END = "2024-12-31"

MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

# Kronos sampling
TEMPERATURE = 1.0
TOP_P = 0.9
SAMPLE_COUNT = 1

# Divergence detection
ROLLING_WINDOW = 20
DIVERGENCE_MULTIPLIER = 1.5
PERSISTENCE_WINDOW = 10
PERSISTENCE_REQUIRED = 8


# ============================================================
# 2. OUTPUT FILES
# ============================================================

OUTPUT_DIR = "_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "yatin-kronos-2021-400-results.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "yatin-kronos-2021-400-report.txt"
)

PLOT_FILE = os.path.join(
    OUTPUT_DIR,
    "yatin-kronos-2021-400-interactive.html"
)


# ============================================================
# 3. START
# ============================================================

print()
print("=" * 80)
print("KRONOS 2021 -> 400 TRADING PERIOD EXPERIMENT")
print("=" * 80)

print()
print("Ticker            :", TICKER)
print("Forecast start    :", FORECAST_START)
print("Lookback          :", LOOKBACK)
print("Forecast horizon  :", PRED_LEN)
print("Model             :", MODEL_NAME)
print()


# ============================================================
# 4. DOWNLOAD DATA
# ============================================================

print("=" * 80)
print("STEP 1: DOWNLOADING DATA")
print("=" * 80)

df = yf.download(
    TICKER,
    start=DATA_START,
    end=DATA_END,
    interval="1d",
    auto_adjust=False,
    progress=False
)

if df.empty:
    raise RuntimeError("Yahoo Finance returned no data.")


# ============================================================
# 5. FIX YFINANCE MULTI-INDEX
# ============================================================

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)


df = df.reset_index()


# ============================================================
# 6. RENAME COLUMNS
# ============================================================

if "Date" in df.columns:
    df.rename(
        columns={"Date": "timestamps"},
        inplace=True
    )


required_columns = [
    "timestamps",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing columns: {missing}"
    )


df = df[required_columns].copy()


df.rename(
    columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    },
    inplace=True
)


# ============================================================
# 7. CLEAN DATA
# ============================================================

df["timestamps"] = pd.to_datetime(
    df["timestamps"]
)

for col in [
    "open",
    "high",
    "low",
    "close",
    "volume"
]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df = df.dropna()

df = df.sort_values(
    "timestamps"
)

df = df.drop_duplicates(
    "timestamps"
)

df = df.reset_index(
    drop=True
)


print()
print(
    "Downloaded rows:",
    len(df)
)

print(
    "Data begins:",
    df["timestamps"].iloc[0].date()
)

print(
    "Data ends:",
    df["timestamps"].iloc[-1].date()
)


# ============================================================
# 8. FIND FORECAST START
# ============================================================

print()
print("=" * 80)
print("STEP 2: BUILDING 400 + 400 EXPERIMENT")
print("=" * 80)

forecast_date = pd.Timestamp(
    FORECAST_START
)

positions = np.where(
    df["timestamps"] >= forecast_date
)[0]

if len(positions) == 0:
    raise RuntimeError(
        "Forecast start date is not available."
    )


forecast_start_index = positions[0]


# ============================================================
# 9. BUILD 400-DAY CONTEXT
# ============================================================

context_start_index = (
    forecast_start_index - LOOKBACK
)

if context_start_index < 0:
    raise RuntimeError(
        "Not enough historical data before 2021 "
        "for a 400-period context."
    )


context_df = df.iloc[
    context_start_index:
    forecast_start_index
].copy()


# ============================================================
# 10. BUILD NEXT 400 PERIOD FORECAST
# ============================================================

forecast_end_index = (
    forecast_start_index + PRED_LEN
)

if forecast_end_index > len(df):
    raise RuntimeError(
        "Not enough data after 2021 for "
        "400 trading periods."
    )


actual_df = df.iloc[
    forecast_start_index:
    forecast_end_index
].copy()


context_df = context_df.reset_index(
    drop=True
)

actual_df = actual_df.reset_index(
    drop=True
)


# ============================================================
# 11. VERIFY EXPERIMENT
# ============================================================

if len(context_df) != LOOKBACK:
    raise RuntimeError(
        f"Context should contain {LOOKBACK} rows, "
        f"but contains {len(context_df)}."
    )


if len(actual_df) != PRED_LEN:
    raise RuntimeError(
        f"Forecast should contain {PRED_LEN} rows, "
        f"but contains {len(actual_df)}."
    )


print()
print("EXPERIMENT WINDOW")
print("-" * 80)

print(
    "400-period context:"
)

print(
    context_df["timestamps"].iloc[0].date(),
    "to",
    context_df["timestamps"].iloc[-1].date()
)

print()

print(
    "400-period forecast:"
)

print(
    actual_df["timestamps"].iloc[0].date(),
    "to",
    actual_df["timestamps"].iloc[-1].date()
)

print()

print("Context rows :", len(context_df))
print("Forecast rows:", len(actual_df))


# ============================================================
# 12. PREPARE KRONOS INPUT
# ============================================================

print()
print("=" * 80)
print("STEP 3: PREPARING KRONOS INPUT")
print("=" * 80)


x_df = context_df[
    [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]
].copy()


x_timestamp = context_df[
    "timestamps"
]

y_timestamp = actual_df[
    "timestamps"
]


# ============================================================
# 13. LOAD KRONOS TOKENIZER
# ============================================================

print()
print("Loading Kronos tokenizer...")

tokenizer = KronosTokenizer.from_pretrained(
    TOKENIZER_NAME
)

print("Tokenizer loaded.")


# ============================================================
# 14. LOAD KRONOS MODEL
# ============================================================

print()
print("Loading Kronos-base...")

model = Kronos.from_pretrained(
    MODEL_NAME
)

print("Kronos-base loaded.")


# ============================================================
# 15. DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

print()
print("Device:", DEVICE)


# ============================================================
# 16. CREATE PREDICTOR
# ============================================================

predictor = KronosPredictor(
    model,
    tokenizer,
    device=DEVICE,
    max_context=512
)


# ============================================================
# 17. RUN KRONOS
# ============================================================

print()
print("=" * 80)
print("STEP 4: RUNNING KRONOS")
print("=" * 80)

print()
print(
    f"Predicting {PRED_LEN} trading periods..."
)

pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=PRED_LEN,
    T=TEMPERATURE,
    top_p=TOP_P,
    sample_count=SAMPLE_COUNT,
    verbose=True
)


print()
print("Kronos prediction completed.")


# ============================================================
# 18. CHECK PREDICTION
# ============================================================

if len(pred_df) != PRED_LEN:
    raise RuntimeError(
        f"Kronos returned {len(pred_df)} predictions "
        f"instead of {PRED_LEN}."
    )


for col in [
    "open",
    "high",
    "low",
    "close"
]:
    if col not in pred_df.columns:
        raise RuntimeError(
            f"Prediction missing column: {col}"
        )


pred_df = pred_df.reset_index(
    drop=True
)


# ============================================================
# 19. CREATE RESULTS DATAFRAME
# ============================================================

print()
print("=" * 80)
print("STEP 5: ACTUAL VS PREDICTED")
print("=" * 80)


results = pd.DataFrame()

results["timestamps"] = (
    actual_df["timestamps"].values
)


# Actual
results["actual_open"] = (
    actual_df["open"].values
)

results["actual_high"] = (
    actual_df["high"].values
)

results["actual_low"] = (
    actual_df["low"].values
)

results["actual_close"] = (
    actual_df["close"].values
)


# Predicted
results["pred_open"] = (
    pred_df["open"].values
)

results["pred_high"] = (
    pred_df["high"].values
)

results["pred_low"] = (
    pred_df["low"].values
)

results["pred_close"] = (
    pred_df["close"].values
)


# ============================================================
# 20. CALCULATE ERRORS
# ============================================================

results["close_error"] = (
    results["pred_close"]
    -
    results["actual_close"]
)

results["absolute_error"] = (
    results["close_error"].abs()
)

results["percentage_error"] = (
    results["absolute_error"]
    /
    results["actual_close"].abs()
) * 100


results["open_error"] = (
    results["pred_open"]
    -
    results["actual_open"]
)

results["high_error"] = (
    results["pred_high"]
    -
    results["actual_high"]
)

results["low_error"] = (
    results["pred_low"]
    -
    results["actual_low"]
)


# ============================================================
# 21. FORECAST HORIZON
# ============================================================

results["horizon"] = np.arange(
    1,
    len(results) + 1
)


# ============================================================
# 22. ROLLING ERROR
# ============================================================

results["rolling_mae"] = (
    results["absolute_error"]
    .rolling(
        ROLLING_WINDOW
    )
    .mean()
)


results["normalized_error"] = (
    results["absolute_error"]
    /
    results["actual_close"].abs()
)


# ============================================================
# 23. OVERALL METRICS
# ============================================================

mae = results[
    "absolute_error"
].mean()


rmse = np.sqrt(
    np.mean(
        results["close_error"] ** 2
    )
)


mape = results[
    "percentage_error"
].mean()


print()
print("Overall MAE :", round(mae, 4))
print("Overall RMSE:", round(rmse, 4))
print("Overall MAPE:", round(mape, 2), "%")


# ============================================================
# 24. FIND SUSTAINED DIVERGENCE
# ============================================================

print()
print("=" * 80)
print("STEP 6: DIVERGENCE ANALYSIS")
print("=" * 80)


valid = results[
    results["rolling_mae"].notna()
].copy()


# First 50 forecast periods = baseline
baseline_points = min(
    50,
    len(valid)
)


baseline_mae = valid[
    "rolling_mae"
].iloc[
    :baseline_points
].median()


threshold = (
    baseline_mae
    *
    DIVERGENCE_MULTIPLIER
)


valid["above_threshold"] = (
    valid["rolling_mae"]
    >
    threshold
)


# Require the error to stay high
# rather than treating one bad candle
# as divergence.

valid["persistent"] = (
    valid["above_threshold"]
    .rolling(
        PERSISTENCE_WINDOW
    )
    .sum()
    >=
    PERSISTENCE_REQUIRED
)


candidates = valid[
    valid["persistent"]
]


if len(candidates) > 0:

    first = candidates.iloc[0]

    divergence_found = True

    divergence_horizon = int(
        first["horizon"]
    )

    divergence_date = (
        first["timestamps"]
    )

    divergence_mae = (
        first["rolling_mae"]
    )

else:

    divergence_found = False

    divergence_horizon = None

    divergence_date = None

    divergence_mae = None


print()
print(
    "Initial rolling MAE:",
    round(baseline_mae, 4)
)

print(
    "Divergence threshold:",
    round(threshold, 4)
)


if divergence_found:

    print()
    print(">>> SUSTAINED DIVERGENCE FOUND <<<")

    print(
        "Forecast horizon:",
        divergence_horizon,
        "trading periods"
    )

    print(
        "Date:",
        divergence_date.date()
    )

    print(
        "Rolling MAE:",
        round(divergence_mae, 4)
    )

else:

    print()
    print(
        "No sustained divergence automatically detected."
    )


# ============================================================
# 25. SAVE CSV
# ============================================================

results.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# 26. CREATE REPORT
# ============================================================

report = []

report.append(
    "=" * 80
)

report.append(
    "KRONOS BASELINE DIVERGENCE REPORT"
)

report.append(
    "=" * 80
)

report.append("")

report.append(
    f"Ticker              : {TICKER}"
)

report.append(
    f"Forecast start      : {FORECAST_START}"
)

report.append(
    f"Lookback            : {LOOKBACK} trading periods"
)

report.append(
    f"Forecast horizon    : {PRED_LEN} trading periods"
)

report.append(
    f"Model               : {MODEL_NAME}"
)

report.append(
    f"Device              : {DEVICE}"
)

report.append("")

report.append(
    "-" * 80
)

report.append(
    "DATA WINDOW"
)

report.append(
    "-" * 80
)

report.append(
    f"Context start       : "
    f"{context_df['timestamps'].iloc[0].date()}"
)

report.append(
    f"Context end         : "
    f"{context_df['timestamps'].iloc[-1].date()}"
)

report.append(
    f"Forecast start      : "
    f"{actual_df['timestamps'].iloc[0].date()}"
)

report.append(
    f"Forecast end        : "
    f"{actual_df['timestamps'].iloc[-1].date()}"
)

report.append("")

report.append(
    "-" * 80
)

report.append(
    "MODEL PERFORMANCE"
)

report.append(
    "-" * 80
)

report.append(
    f"MAE                 : {mae:.4f}"
)

report.append(
    f"RMSE                : {rmse:.4f}"
)

report.append(
    f"MAPE                : {mape:.2f}%"
)

report.append("")

report.append(
    "-" * 80
)

report.append(
    "DIVERGENCE ANALYSIS"
)

report.append(
    "-" * 80
)

report.append(
    f"Rolling window      : {ROLLING_WINDOW}"
)

report.append(
    f"Baseline MAE        : {baseline_mae:.4f}"
)

report.append(
    f"Threshold            : {threshold:.4f}"
)

report.append(
    f"Persistence window  : {PERSISTENCE_WINDOW}"
)

report.append(
    f"Required persistence: {PERSISTENCE_REQUIRED}"
)

report.append("")


if divergence_found:

    report.append(
        "DIVERGENCE DETECTED : YES"
    )

    report.append(
        f"Divergence horizon  : "
        f"{divergence_horizon} trading periods"
    )

    report.append(
        f"Divergence date     : "
        f"{divergence_date.date()}"
    )

    report.append(
        f"Divergence MAE      : "
        f"{divergence_mae:.4f}"
    )

    report.append("")

    report.append(
        "INTERPRETATION:"
    )

    report.append(
        "Kronos begins showing sustained "
        "prediction error around this horizon."
    )

    report.append(
        "This duration is the candidate "
        "horizon for the next training experiment."
    )

else:

    report.append(
        "DIVERGENCE DETECTED : NO"
    )

    report.append(
        "No sustained divergence was detected "
        "using the automatic threshold."
    )

report.append("")

report.append(
    "-" * 80
)

report.append(
    "NEXT STEP"
)

report.append(
    "-" * 80
)

report.append(
    "First inspect the interactive Plotly graph."
)

report.append(
    "Then use the observed divergence duration "
    "to determine the next training/fine-tuning experiment."
)

report.append("")

report.append(
    "=" * 80
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report)
    )


# ============================================================
# 27. CREATE INTERACTIVE PLOTLY GRAPH
# ============================================================

print()
print("=" * 80)
print("STEP 7: CREATING INTERACTIVE PLOTLY GRAPH")
print("=" * 80)


fig = make_subplots(

    rows=3,

    cols=1,

    shared_xaxes=True,

    vertical_spacing=0.06,

    row_heights=[
        0.60,
        0.20,
        0.20
    ],

    subplot_titles=[
        "Actual vs Kronos Predicted Candlesticks",
        "Absolute Close Prediction Error",
        "Rolling MAE / Divergence Threshold"
    ]
)


# ============================================================
# 28. ACTUAL CANDLESTICKS
# ============================================================

fig.add_trace(

    go.Candlestick(

        x=results["timestamps"],

        open=results["actual_open"],

        high=results["actual_high"],

        low=results["actual_low"],

        close=results["actual_close"],

        name="Actual"
    ),

    row=1,
    col=1
)


# ============================================================
# 29. PREDICTED CANDLESTICKS
# ============================================================

fig.add_trace(

    go.Candlestick(

        x=results["timestamps"],

        open=results["pred_open"],

        high=results["pred_high"],

        low=results["pred_low"],

        close=results["pred_close"],

        name="Kronos Prediction",

        opacity=0.55
    ),

    row=1,
    col=1
)


# ============================================================
# 30. ABSOLUTE ERROR
# ============================================================

fig.add_trace(

    go.Scatter(

        x=results["timestamps"],

        y=results["absolute_error"],

        mode="lines",

        name="Absolute Close Error"
    ),

    row=2,
    col=1
)


# ============================================================
# 31. ROLLING MAE
# ============================================================

fig.add_trace(

    go.Scatter(

        x=results["timestamps"],

        y=results["rolling_mae"],

        mode="lines",

        name="20-Period Rolling MAE"
    ),

    row=3,
    col=1
)


# ============================================================
# 32. DIVERGENCE THRESHOLD
# ============================================================

fig.add_hline(

    y=threshold,

    line_dash="dash",

    line_width=2,

    annotation_text="Divergence threshold",

    row=3,

    col=1
)


# ============================================================
# 33. MARK DIVERGENCE DATE
# ============================================================

if divergence_found:

    fig.add_vline(

        x=divergence_date,

        line_dash="dash",

        line_width=3,

        annotation_text=(
            f"Divergence ≈ "
            f"{divergence_horizon} periods"
        ),

        row=1,
        col=1
    )

    fig.add_vline(

        x=divergence_date,

        line_dash="dash",

        line_width=2,

        row=2,
        col=1
    )

    fig.add_vline(

        x=divergence_date,

        line_dash="dash",

        line_width=2,

        row=3,
        col=1
    )


# ============================================================
# 34. GRAPH SETTINGS
# ============================================================

fig.update_layout(

    title=(
        f"{TICKER} — Kronos 400-Period "
        f"Baseline / Divergence Analysis"
    ),

    template="plotly_white",

    height=1200,

    hovermode="x unified",

    xaxis_rangeslider_visible=False,

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)


fig.update_yaxes(
    title_text="Price (₹)",
    row=1,
    col=1
)

fig.update_yaxes(
    title_text="Error (₹)",
    row=2,
    col=1
)

fig.update_yaxes(
    title_text="Rolling MAE (₹)",
    row=3,
    col=1
)

fig.update_xaxes(
    title_text="Date",
    row=3,
    col=1
)


# ============================================================
# 35. SAVE HTML
# ============================================================

fig.write_html(
    PLOT_FILE,
    include_plotlyjs=True
)


# ============================================================
# 36. FINAL MESSAGE
# ============================================================

print()
print("=" * 80)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 80)

print()

print(
    "CSV   :",
    RESULTS_FILE
)

print(
    "REPORT:",
    REPORT_FILE
)

print(
    "GRAPH :",
    PLOT_FILE
)

print()

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"MAPE : {mape:.2f}%"
)

print()

if divergence_found:

    print(
        "DIVERGENCE HORIZON:",
        divergence_horizon,
        "trading periods"
    )

    print(
        "DIVERGENCE DATE:",
        divergence_date.date()
    )

else:

    print(
        "No automatic sustained divergence found."
    )

print()
print(
    "Open the HTML file to inspect the "
    "interactive Plotly candlestick graph."
)

print("=" * 80)
