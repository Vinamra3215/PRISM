import os
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YATIN6_DIR = os.path.join(
    os.path.dirname(BASE_DIR),
    "yatin6"
)

DATA_PATH = os.path.join(
    YATIN6_DIR,
    "data",
    "HCLTECH_clean.parquet"
)

# Prediction result created by your yatin6 rolling prediction
RESULT_PATH = os.path.join(
    YATIN6_DIR,
    "results",
    "HCLTECH_rolling_predictions.parquet"
)

PLOTS_DIR = os.path.join(
    BASE_DIR,
    "plots"
)

os.makedirs(PLOTS_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    PLOTS_DIR,
    "HCLTECH_rolling_prediction_less_noise.html"
)


# ============================================================
# SETTINGS
# ============================================================

SMOOTHING_WINDOW = 5

START_DATE = "2022-01-01"
END_DATE = "2026-12-31"


# ============================================================
# LOAD ORIGINAL DATA
# ============================================================

print("=" * 70)
print("HCLTECH - LESS NOISY ROLLING PREDICTION")
print("=" * 70)

print("\nLoading original data:")
print(DATA_PATH)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\nOriginal data not found:\n{DATA_PATH}"
    )

data = pd.read_parquet(DATA_PATH)

print("\nOriginal data shape:")
print(data.shape)

print("\nOriginal data columns:")
print(data.columns.tolist())


# ============================================================
# FIND DATE COLUMN IN ORIGINAL DATA
# ============================================================

date_candidates = [
    "Date",
    "date",
    "Datetime",
    "datetime",
    "Timestamp",
    "timestamp"
]

date_col = None

for col in date_candidates:
    if col in data.columns:
        date_col = col
        break


# If not found, check index
if date_col is None:

    if isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index()
        date_col = data.columns[0]

    else:
        raise ValueError(
            "\nCould not find a date column in original data.\n"
            f"Available columns:\n{data.columns.tolist()}"
        )


# ============================================================
# PREPARE DATES
# ============================================================

data[date_col] = pd.to_datetime(
    data[date_col],
    errors="coerce"
)

data = data.dropna(
    subset=[date_col]
)

data = data.sort_values(
    date_col
).reset_index(drop=True)


# ============================================================
# LIMIT ORIGINAL DATA TO 2022-2026
# ============================================================

data_period = data[
    (data[date_col] >= START_DATE) &
    (data[date_col] <= END_DATE)
].copy()

data_period = data_period.reset_index(drop=True)

print("\nDates available from 2022-2026:")
print(len(data_period))

print(
    "Date range:",
    data_period[date_col].min(),
    "to",
    data_period[date_col].max()
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("\nLoading prediction results:")
print(RESULT_PATH)

if not os.path.exists(RESULT_PATH):
    raise FileNotFoundError(
        f"\nPrediction file not found:\n{RESULT_PATH}"
    )

pred = pd.read_parquet(
    RESULT_PATH
)

print("\nPrediction shape:")
print(pred.shape)

print("\nPrediction columns:")
print(pred.columns.tolist())


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "actual_close",
    "predicted_close"
]

for col in required_columns:

    if col not in pred.columns:
        raise ValueError(
            f"\nRequired column '{col}' not found.\n"
            f"Available columns:\n{pred.columns.tolist()}"
        )


# ============================================================
# CONVERT CLOSE VALUES TO NUMERIC
# ============================================================

pred["actual_close"] = pd.to_numeric(
    pred["actual_close"],
    errors="coerce"
)

pred["predicted_close"] = pd.to_numeric(
    pred["predicted_close"],
    errors="coerce"
)

pred = pred.reset_index(drop=True)


# ============================================================
# ALIGN DATES WITH PREDICTIONS
# ============================================================

prediction_length = len(pred)
date_length = len(data_period)

print("\nPrediction rows:", prediction_length)
print("Available dates:", date_length)


if date_length < prediction_length:

    raise ValueError(
        "\nNot enough dates in the 2022-2026 data to match "
        "the prediction rows.\n"
        f"Predictions: {prediction_length}\n"
        f"Dates: {date_length}"
    )


# Take the LAST N dates if prediction file contains
# the latest N prediction observations.

dates = data_period[date_col].iloc[
    -prediction_length:
].reset_index(drop=True)


# ============================================================
# CREATE FINAL DATAFRAME
# ============================================================

df = pd.DataFrame()

df["Date"] = dates

df["Actual Close"] = pred[
    "actual_close"
].values

df["Original Prediction"] = pred[
    "predicted_close"
].values


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

df = df.dropna(
    subset=[
        "Date",
        "Actual Close",
        "Original Prediction"
    ]
).reset_index(drop=True)


# ============================================================
# 5-DAY SMOOTHING
# ============================================================

df["Smoothed Prediction"] = (
    df["Original Prediction"]
    .rolling(
        window=SMOOTHING_WINDOW,
        center=True,
        min_periods=1
    )
    .mean()
)


# ============================================================
# PRINT INFORMATION
# ============================================================

print("\nFinal plotting data:")
print(df.shape)

print(
    "Plot date range:",
    df["Date"].min(),
    "to",
    df["Date"].max()
)

print(
    "\nSmoothing window:",
    SMOOTHING_WINDOW,
    "trading days"
)


# ============================================================
# CREATE GRAPH
# ============================================================

fig = go.Figure()


# ------------------------------------------------------------
# ACTUAL CLOSE
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Actual Close"],
        mode="lines",
        name="Actual Close",
        line=dict(
            width=2
        )
    )
)


# ------------------------------------------------------------
# ORIGINAL NOISY PREDICTION
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Original Prediction"],
        mode="lines",
        name="Original Prediction",
        line=dict(
            width=1
        ),
        opacity=0.20
    )
)


# ------------------------------------------------------------
# SMOOTHED PREDICTION
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Smoothed Prediction"],
        mode="lines",
        name="5-Day Smoothed Prediction",
        line=dict(
            width=3
        )
    )
)


# ============================================================
# LAYOUT
# ============================================================

fig.update_layout(

    title=(
        "HCLTECH — Kronos + LoRA | "
        "Rolling Prediction with Reduced Noise"
    ),

    xaxis=dict(
        title="Date",
        range=[
            START_DATE,
            END_DATE
        ]
    ),

    yaxis=dict(
        title="HCLTECH Price"
    ),

    template="plotly_white",

    showlegend=True,

    hovermode=False,

    margin=dict(
        l=70,
        r=40,
        t=90,
        b=60
    )
)


# ============================================================
# NO ZOOM / NO PAN / NO MODEBAR
# ============================================================

fig.write_html(
    OUTPUT_PATH,
    config={
        "staticPlot": True,
        "displayModeBar": False
    }
)


# ============================================================
# SAVE
# ============================================================

print("\n" + "=" * 70)
print("GRAPH CREATED SUCCESSFULLY")
print("=" * 70)

print("\nOutput:")
print(OUTPUT_PATH)

print("\nThe graph contains:")
print("1. Actual Close")
print("2. Original prediction (faint)")
print("3. 5-day smoothed prediction")

print("\nNo zoom / pan controls.")

print("=" * 70)
