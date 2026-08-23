import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

# ============================================================
# 1. KRONOS IMPORT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from model import Kronos, KronosTokenizer, KronosPredictor


# ============================================================
# 2. CONFIG
# ============================================================

DATA_FILE = os.path.expanduser(
    "~/Kronos/het/data/NIFTY50_5Y_OHLCV.parquet"
)

TRAIN_START = "2022-01-01"
TRAIN_END = "2022-12-31"

MODEL_NAME = "NeoQuasar/Kronos-small"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

CSV_OUTPUT = "nifty50_kronos_2022_forecast.csv"
HTML_OUTPUT = "nifty50_kronos_2022_forecast.html"


# ============================================================
# 3. DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = "cuda:0"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


print("=" * 70)
print("KRONOS NIFTY 50 FORECAST")
print("=" * 70)

print(f"Device: {DEVICE}")


# ============================================================
# 4. LOAD EXISTING NIFTY 50 PARQUET
# ============================================================

print("\nLoading NIFTY 50 data...")

print(f"File: {DATA_FILE}")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"\nNIFTY data not found at:\n{DATA_FILE}"
    )

df = pd.read_parquet(DATA_FILE)

print("\nOriginal columns:")
print(df.columns)


# ============================================================
# 5. FIX PARQUET COLUMNS
# ============================================================

# Your file has columns like:
#
# ('Close', '^NSEI')
# ('High', '^NSEI')
# ('Low', '^NSEI')
# ('Open', '^NSEI')
# ('Volume', '^NSEI')
# Date
#
# Convert them to:
#
# close
# high
# low
# open
# volume
# timestamps

if isinstance(df.columns, pd.MultiIndex):

    new_columns = []

    for col in df.columns:

        # Example:
        # ('Close', '^NSEI')
        #
        # We only need "Close"

        new_columns.append(
            str(col[0]).lower()
        )

    df.columns = new_columns

else:

    df.columns = [
        str(col).lower()
        for col in df.columns
    ]


# ============================================================
# 6. HANDLE DATE
# ============================================================

if "date" in df.columns:

    df.rename(
        columns={
            "date": "timestamps"
        },
        inplace=True
    )

elif "datetime" in df.columns:

    df.rename(
        columns={
            "datetime": "timestamps"
        },
        inplace=True
    )


# If Date is the index instead of a column
if "timestamps" not in df.columns:

    df = df.reset_index()

    if "date" in df.columns:

        df.rename(
            columns={
                "date": "timestamps"
            },
            inplace=True
        )

    elif "Date" in df.columns:

        df.rename(
            columns={
                "Date": "timestamps"
            },
            inplace=True
        )

    elif "index" in df.columns:

        df.rename(
            columns={
                "index": "timestamps"
            },
            inplace=True
        )


# ============================================================
# 7. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "timestamps",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        "\nMissing required columns: "
        + str(missing_columns)
        + "\n\nAvailable columns:\n"
        + str(df.columns.tolist())
    )


# ============================================================
# 8. CLEAN DATA
# ============================================================

df = df[
    required_columns
].copy()

df["timestamps"] = pd.to_datetime(
    df["timestamps"]
)

# Remove timezone if present
if df["timestamps"].dt.tz is not None:

    df["timestamps"] = (
        df["timestamps"]
        .dt.tz_localize(None)
    )


df.dropna(
    inplace=True
)

df.sort_values(
    "timestamps",
    inplace=True
)

df.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# 9. DISPLAY DATA RANGE
# ============================================================

print("\nLoaded dataset:")

print(
    f"Start : {df['timestamps'].iloc[0]}"
)

print(
    f"End   : {df['timestamps'].iloc[-1]}"
)

print(
    f"Rows  : {len(df)}"
)


# ============================================================
# 10. SPLIT DATA
# ============================================================

# IMPORTANT:
#
# TRAIN DATA:
# 01/01/2022 → 31/12/2022
#
# FUTURE DATA:
# 01/01/2023 → end of dataset
#
# Only TRAIN DATA is passed to Kronos.
#
# FUTURE DATA is used ONLY:
# - to provide future timestamps
# - to compare predictions with actual prices
# - to plot the result
#
# Actual future OHLCV values are NEVER passed to Kronos.

train_df = df[
    (df["timestamps"] >= TRAIN_START)
    &
    (df["timestamps"] <= TRAIN_END)
].copy()

future_df = df[
    df["timestamps"] > TRAIN_END
].copy()


train_df.reset_index(
    drop=True,
    inplace=True
)

future_df.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# 11. CHECK SPLIT
# ============================================================

if train_df.empty:

    raise RuntimeError(
        "No training data found between "
        f"{TRAIN_START} and {TRAIN_END}."
    )


if future_df.empty:

    raise RuntimeError(
        "No future data exists after "
        f"{TRAIN_END}."
    )


print("\n" + "=" * 70)

print("MODEL INPUT")
print(
    f"{train_df['timestamps'].iloc[0].date()}"
    f" → "
    f"{train_df['timestamps'].iloc[-1].date()}"
)

print(
    f"Training rows: {len(train_df)}"
)

print("\nFORECAST HORIZON")

print(
    f"{future_df['timestamps'].iloc[0].date()}"
    f" → "
    f"{future_df['timestamps'].iloc[-1].date()}"
)

print(
    f"Forecast rows: {len(future_df)}"
)

print("=" * 70)


# ============================================================
# 12. LOAD KRONOS TOKENIZER
# ============================================================

print("\nLoading Kronos tokenizer...")

tokenizer = KronosTokenizer.from_pretrained(
    TOKENIZER_NAME
)


# ============================================================
# 13. LOAD KRONOS MODEL
# ============================================================

print("Loading Kronos model...")

model = Kronos.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 14. CREATE PREDICTOR
# ============================================================

predictor = KronosPredictor(
    model,
    tokenizer,
    device=DEVICE,
    max_context=512
)

print("Kronos loaded successfully.")


# ============================================================
# 15. PREPARE 2022 INPUT
# ============================================================

# THIS IS THE ONLY PRICE DATA GIVEN TO KRONOS.

x_df = train_df[
    [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
].copy()


x_timestamp = train_df[
    "timestamps"
].copy()


# ============================================================
# 16. FUTURE TIMESTAMPS
# ============================================================

# We give Kronos the timestamps corresponding to the
# dates we want it to forecast.
#
# We DO NOT give it the actual prices for these dates.

y_timestamp = future_df[
    "timestamps"
].copy()


y_timestamp = future_df[
    "timestamps"
].copy()

PRED_LEN = min(512, len(y_timestamp))

y_timestamp = y_timestamp.iloc[:PRED_LEN].copy()
future_df = future_df.iloc[:PRED_LEN].copy()

print(
    f"\nForecasting {PRED_LEN} trading days..."
)
   


print(
    f"\nForecasting {PRED_LEN} trading days..."
)

print(
    f"From {y_timestamp.iloc[0].date()}"
    f" → "
    f"{y_timestamp.iloc[-1].date()}"
)


# ============================================================
# 17. SINGLE KRONOS FORECAST
# ============================================================

print("\nRunning Kronos...")
print(
    "The model will receive ONLY the 2022 data."
)

print(
    "No rolling window will be used."
)

print(
    "No future actual prices will be fed back."
)

print("\n" + "-" * 70)


pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=PRED_LEN,

    T=1.0,
    top_p=0.9,
    sample_count=1,
)


print("-" * 70)
print("Kronos forecast complete.")


# ============================================================
# 18. PREPARE PREDICTION DATAFRAME
# ============================================================

pred_df = pred_df.copy()


pred_df["timestamp"] = (
    y_timestamp.values
)


pred_df.set_index(
    "timestamp",
    inplace=True
)


# ============================================================
# 19. RENAME PREDICTION COLUMNS
# ============================================================

pred_df.rename(
    columns={
        "open": "pred_open",
        "high": "pred_high",
        "low": "pred_low",
        "close": "pred_close",
        "volume": "pred_volume",
    },
    inplace=True
)


# ============================================================
# 20. ADD ACTUAL FUTURE DATA
# ============================================================

# IMPORTANT:
#
# These values are NOT used by Kronos.
#
# They are only added AFTER prediction so that we can
# compare the forecast against reality.

pred_df["actual_open"] = (
    future_df["open"].values
)

pred_df["actual_high"] = (
    future_df["high"].values
)

pred_df["actual_low"] = (
    future_df["low"].values
)

pred_df["actual_close"] = (
    future_df["close"].values
)


# ============================================================
# 21. CALCULATE CLOSE ERRORS
# ============================================================

pred_df["close_error"] = (
    pred_df["pred_close"]
    - pred_df["actual_close"]
)

pred_df["absolute_error"] = (
    pred_df["close_error"]
    .abs()
)

pred_df["squared_error"] = (
    pred_df["close_error"]
    ** 2
)


# ============================================================
# 22. MAE / RMSE
# ============================================================

mae = (
    pred_df["absolute_error"]
    .mean()
)


rmse = np.sqrt(
    pred_df["squared_error"]
    .mean()
)


# ============================================================
# 23. DIRECTIONAL ACCURACY
# ============================================================

actual_direction = (
    pred_df["actual_close"].diff()
    > 0
)

pred_direction = (
    pred_df["pred_close"].diff()
    > 0
)


direction_matches = (
    actual_direction.iloc[1:]
    ==
    pred_direction.iloc[1:]
)


directional_accuracy = (
    direction_matches.mean()
    * 100
)


# ============================================================
# 24. PRINT METRICS
# ============================================================

print("\n")
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(
    f"Training period:"
    f" {train_df['timestamps'].iloc[0].date()}"
    f" → "
    f"{train_df['timestamps'].iloc[-1].date()}"
)

print(
    f"Forecast period:"
    f" {future_df['timestamps'].iloc[0].date()}"
    f" → "
    f"{future_df['timestamps'].iloc[-1].date()}"
)

print(
    f"\nForecast days: {PRED_LEN}"
)

print(
    f"\nMAE:  {mae:.4f}"
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"Directional Accuracy:"
    f" {directional_accuracy:.2f}%"
)


# ============================================================
# 25. SHOW FIRST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 FORECASTS")
print("=" * 70)

print(
    pred_df[
        [
            "pred_open",
            "pred_high",
            "pred_low",
            "pred_close",
            "actual_close",
        ]
    ].head(10)
)


# ============================================================
# 26. SHOW LAST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("LAST 10 FORECASTS")
print("=" * 70)

print(
    pred_df[
        [
            "pred_open",
            "pred_high",
            "pred_low",
            "pred_close",
            "actual_close",
        ]
    ].tail(10)
)


# ============================================================
# 27. SAVE CSV
# ============================================================

pred_df.to_csv(
    CSV_OUTPUT
)

print(
    f"\nCSV saved:"
    f"\n{os.path.abspath(CSV_OUTPUT)}"
)


# ============================================================
# 28. CREATE PLOTLY FIGURE
# ============================================================

fig = go.Figure()


# ------------------------------------------------------------
# 2022 ACTUAL
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=train_df["timestamps"],
        y=train_df["close"],

        mode="lines",

        name="2022 Actual",

        line=dict(
            width=2
        ),

        hovertemplate=(
            "<b>2022 Actual</b>"
            "<br>Date: %{x|%d %b %Y}"
            "<br>NIFTY: %{y:.2f}"
            "<extra></extra>"
        ),
    )
)


# ------------------------------------------------------------
# ACTUAL FUTURE
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=future_df["timestamps"],
        y=future_df["close"],

        mode="lines",

        name="Actual Future",

        line=dict(
            width=2
        ),

        hovertemplate=(
            "<b>Actual</b>"
            "<br>Date: %{x|%d %b %Y}"
            "<br>NIFTY: %{y:.2f}"
            "<extra></extra>"
        ),
    )
)


# ------------------------------------------------------------
# KRONOS FORECAST
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=pred_df.index,
        y=pred_df["pred_close"],

        mode="lines",

        name="Kronos Prediction",

        line=dict(
            width=2
        ),

        hovertemplate=(
            "<b>Kronos Prediction</b>"
            "<br>Date: %{x|%d %b %Y}"
            "<br>Predicted NIFTY: %{y:.2f}"
            "<extra></extra>"
        ),
    )
)


# ============================================================
# 29. FORECAST START MARKER
# ============================================================

fig.add_vline(
    x=pd.Timestamp(TRAIN_END),

    line_dash="dash",

    annotation_text="Forecast starts",

    annotation_position="top",
)


# ============================================================
# 30. PLOTLY LAYOUT
# ============================================================

fig.update_layout(

    title={
        "text": (
            "NIFTY 50 - Kronos Long-Horizon Forecast"
        ),

        "x": 0.5,
    },

    xaxis_title="Date",

    yaxis_title="NIFTY 50",

    template="plotly_white",

    height=750,

    hovermode="x unified",

    legend=dict(
        orientation="h",

        yanchor="bottom",

        y=1.02,

        xanchor="center",

        x=0.5,
    ),

    margin=dict(
        l=70,
        r=40,
        t=100,
        b=70,
    ),

    xaxis=dict(

        type="date",

        rangeslider=dict(
            visible=True
        ),

        rangeselector=dict(

            buttons=[

                dict(
                    count=1,
                    label="1Y",
                    step="year",
                    stepmode="backward",
                ),

                dict(
                    count=2,
                    label="2Y",
                    step="year",
                    stepmode="backward",
                ),

                dict(
                    count=3,
                    label="3Y",
                    step="year",
                    stepmode="backward",
                ),

                dict(
                    step="all",
                    label="ALL",
                ),
            ]
        ),
    ),
)


# ============================================================
# 31. SAVE INTERACTIVE HTML
# ============================================================

fig.write_html(
    HTML_OUTPUT,

    include_plotlyjs=True,

    full_html=True,
)


print(
    f"\nInteractive HTML saved:"
    f"\n{os.path.abspath(HTML_OUTPUT)}"
)


# ============================================================
# 32. DISPLAY
# ============================================================

fig.show()


# ============================================================
# 33. DONE
# ============================================================

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)