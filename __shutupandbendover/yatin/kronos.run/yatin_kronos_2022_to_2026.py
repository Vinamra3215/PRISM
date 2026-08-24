# ============================================================
# KRONOS LONG-RANGE FORECAST
#
# Context       : 400 trading days before 2022-01-01
# Forecast      : 2022-01-01 to 2026-08-14
# Ticker        : RELIANCE.NS
#
# Graph:
#   1. 400-day historical/context data
#   2. Actual price
#   3. Kronos predicted price
#   4. Baseline
#   5. Divergence points + vertical divergence lines
#
# Output:
#   _results/yatin-kronos-2022-2026-interactive.html
#   _results/yatin-kronos-2022-2026-results.csv
#   _results/yatin-kronos-2022-2026-report.txt
# ============================================================


import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from model import Kronos, KronosTokenizer, KronosPredictor


# ============================================================
# 1. SETTINGS
# ============================================================

TICKER = "RELIANCE.NS"

DATA_FILE = "/home/soq/yatin/RELIANCE_5Y_OHLCVdata.parquet"

LOOKBACK = 400

START_DATE = "2022-01-01"
END_DATE = "2026-08-14"

# Forecast in chunks.
# 22 trading days is approximately one month.
PRED_LEN = 22

# Kronos parameters
TEMPERATURE = 1.0
TOP_P = 0.9
SAMPLE_COUNT = 1


# ============================================================
# 2. MODEL SETTINGS
# ============================================================

TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_NAME = "NeoQuasar/Kronos-base"

# Existing Kronos context limit
MAX_CONTEXT = 512


# ============================================================
# 3. OUTPUT DIRECTORY
# ============================================================

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "_results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

GRAPH_FILE = os.path.join(
    RESULTS_DIR,
    "yatin-kronos-2022-2026-interactive.html"
)

CSV_FILE = os.path.join(
    RESULTS_DIR,
    "yatin-kronos-2022-2026-results.csv"
)

REPORT_FILE = os.path.join(
    RESULTS_DIR,
    "yatin-kronos-2022-2026-report.txt"
)


# ============================================================
# 4. CHECK DATA FILE
# ============================================================

if not os.path.exists(DATA_FILE):

    raise FileNotFoundError(
        f"\nData file not found:\n{DATA_FILE}\n"
    )


# ============================================================
# 5. LOAD DATA
# ============================================================

print()
print("=" * 80)
print("KRONOS 400-DAY CONTEXT + 2022-2026 FORECAST")
print("=" * 80)

print()
print("Loading data...")

raw = pd.read_parquet(DATA_FILE)

print("Original shape:", raw.shape)


# ============================================================
# 6. NORMALIZE MULTIINDEX COLUMNS
# ============================================================

def normalize_column_name(col):

    if isinstance(col, tuple):

        parts = []

        for x in col:

            if x is not None:

                parts.append(str(x))

        return "_".join(parts).lower()

    return str(col).lower()


raw.columns = [
    normalize_column_name(c)
    for c in raw.columns
]


# ============================================================
# 7. FIND OHLCV COLUMNS
# ============================================================

def find_col(name):

    name = name.lower()

    for col in raw.columns:

        c = str(col).lower()

        if c == name:
            return col

        if c.startswith(name + "_"):
            return col

        if "_" + name in c:
            return col

    return None


OPEN_COL = find_col("open")
HIGH_COL = find_col("high")
LOW_COL = find_col("low")
CLOSE_COL = find_col("close")
VOLUME_COL = find_col("volume")


print()
print("Detected columns:")

print("Open   :", OPEN_COL)
print("High   :", HIGH_COL)
print("Low    :", LOW_COL)
print("Close  :", CLOSE_COL)
print("Volume :", VOLUME_COL)


if any(
    x is None
    for x in [
        OPEN_COL,
        HIGH_COL,
        LOW_COL,
        CLOSE_COL,
        VOLUME_COL
    ]
):

    print()
    print("Available columns:")
    print(raw.columns.tolist())

    raise ValueError(
        "\nCould not correctly identify OHLCV columns."
    )


# ============================================================
# 8. FIND DATE / TIMESTAMP INDEX
# ============================================================

if isinstance(raw.index, pd.DatetimeIndex):

    raw.index = pd.to_datetime(
        raw.index,
        errors="coerce"
    )

    raw = raw.reset_index()

    DATE_COL = raw.columns[0]

else:

    possible_dates = [
        "date",
        "datetime",
        "timestamp",
        "time"
    ]

    DATE_COL = None

    for c in raw.columns:

        if str(c).lower() in possible_dates:

            DATE_COL = c
            break

    if DATE_COL is None:

        raise ValueError(
            "No date/timestamp column found."
        )


raw[DATE_COL] = pd.to_datetime(
    raw[DATE_COL],
    errors="coerce"
)


# ============================================================
# 9. BUILD CLEAN DATAFRAME
# ============================================================

df = pd.DataFrame({

    "timestamp": raw[DATE_COL],

    "open": pd.to_numeric(
        raw[OPEN_COL],
        errors="coerce"
    ),

    "high": pd.to_numeric(
        raw[HIGH_COL],
        errors="coerce"
    ),

    "low": pd.to_numeric(
        raw[LOW_COL],
        errors="coerce"
    ),

    "close": pd.to_numeric(
        raw[CLOSE_COL],
        errors="coerce"
    ),

    "volume": pd.to_numeric(
        raw[VOLUME_COL],
        errors="coerce"
    )
})


df = df.dropna(
    subset=[
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]
)


df = (
    df
    .sort_values("timestamp")
    .drop_duplicates("timestamp")
    .reset_index(drop=True)
)


# ============================================================
# 10. REMOVE TIMEZONE IF PRESENT
# ============================================================

if hasattr(
    df["timestamp"].dt,
    "tz"
):

    if df["timestamp"].dt.tz is not None:

        df["timestamp"] = (
            df["timestamp"]
            .dt.tz_localize(None)
        )


# ============================================================
# 11. DATE RANGES
# ============================================================

forecast_start = pd.Timestamp(
    START_DATE
)

forecast_end = pd.Timestamp(
    END_DATE
)


# We need exactly 400 observations before
# 2022-01-01.

before_start = df[
    df["timestamp"] < forecast_start
].copy()


if len(before_start) < LOOKBACK:

    raise ValueError(
        f"\nOnly {len(before_start)} rows are available "
        f"before {START_DATE}.\n"
        f"Need at least {LOOKBACK} rows."
    )


context_df = before_start.tail(
    LOOKBACK
).copy()


# ============================================================
# 12. ACTUAL FORECAST-PERIOD DATA
# ============================================================

actual_df = df[
    (
        df["timestamp"] >= forecast_start
    )
    &
    (
        df["timestamp"] <= forecast_end
    )
].copy()


if len(actual_df) == 0:

    raise ValueError(
        "\nNo data exists between "
        f"{START_DATE} and {END_DATE}."
    )


# ============================================================
# 13. CHECK DATA LENGTH
# ============================================================

print()
print("Context period:")
print(
    context_df["timestamp"].iloc[0],
    "to",
    context_df["timestamp"].iloc[-1]
)

print(
    "Context rows:",
    len(context_df)
)

print()
print("Forecast period:")
print(
    actual_df["timestamp"].iloc[0],
    "to",
    actual_df["timestamp"].iloc[-1]
)

print(
    "Forecast rows:",
    len(actual_df)
)


# ============================================================
# 14. LOAD KRONOS MODEL
# ============================================================

print()
print("=" * 80)
print("LOADING KRONOS")
print("=" * 80)

print()
print("Tokenizer:", TOKENIZER_NAME)
print("Model    :", MODEL_NAME)


tokenizer = KronosTokenizer.from_pretrained(
    TOKENIZER_NAME
)


model = Kronos.from_pretrained(
    MODEL_NAME
)


predictor = KronosPredictor(
    model,
    tokenizer,
    max_context=MAX_CONTEXT
)


print()
print("Kronos loaded successfully.")


# ============================================================
# 15. PREPARE INITIAL CONTEXT
# ============================================================

current_context = context_df[
    [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]
].copy()


# ============================================================
# 16. FORECAST CONTAINER
# ============================================================

all_predictions = []


# ============================================================
# 17. LONG-RANGE ROLLING FORECAST
# ============================================================

print()
print("=" * 80)
print("STARTING LONG-RANGE FORECAST")
print("=" * 80)

remaining_actual = actual_df.copy()

total_rows = len(
    remaining_actual
)

completed = 0


while completed < total_rows:

    # --------------------------------------------------------
    # Number of periods for this iteration
    # --------------------------------------------------------

    remaining = total_rows - completed

    this_pred_len = min(
        PRED_LEN,
        remaining
    )


    # --------------------------------------------------------
    # Current context
    # --------------------------------------------------------

    model_context = current_context.tail(
        MAX_CONTEXT
    ).copy()


    # --------------------------------------------------------
    # Future timestamps
    # --------------------------------------------------------

    future_rows = remaining_actual.iloc[
        completed:
        completed + this_pred_len
    ].copy()


    x_timestamp = model_context[
        "timestamp"
    ]

    y_timestamp = future_rows[
        "timestamp"
    ]


    # --------------------------------------------------------
    # Kronos prediction
    # --------------------------------------------------------

    print(
        f"\nForecasting "
        f"{completed + 1}"
        f"-"
        f"{completed + this_pred_len}"
        f" / {total_rows}"
    )


    pred_df = predictor.predict(

        df=model_context[
            [
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        ],

        x_timestamp=x_timestamp,

        y_timestamp=y_timestamp,

        pred_len=this_pred_len,

        T=TEMPERATURE,

        top_p=TOP_P,

        sample_count=SAMPLE_COUNT,

        verbose=False
    )


    # --------------------------------------------------------
    # Normalize prediction output
    # --------------------------------------------------------

    pred_df = pred_df.copy()

    pred_df.index = pd.to_datetime(
        pred_df.index
    )


    # Make sure columns are lowercase
    pred_df.columns = [
        str(c).lower()
        for c in pred_df.columns
    ]


    # --------------------------------------------------------
    # Handle possible timestamp column
    # --------------------------------------------------------

    if "timestamp" in pred_df.columns:

        pred_df["timestamp"] = pd.to_datetime(
            pred_df["timestamp"]
        )

    else:

        pred_df["timestamp"] = (
            future_rows["timestamp"]
            .values[:len(pred_df)]
        )


    # --------------------------------------------------------
    # Store prediction
    # --------------------------------------------------------

    all_predictions.append(
        pred_df[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        ].copy()
    )


    # --------------------------------------------------------
    # Feed predictions back into context
    #
    # This creates a recursive long-range forecast.
    # --------------------------------------------------------

    predicted_context = pd.DataFrame({

        "timestamp":
            pred_df["timestamp"].values,

        "open":
            pred_df["open"].values,

        "high":
            pred_df["high"].values,

        "low":
            pred_df["low"].values,

        "close":
            pred_df["close"].values,

        "volume":
            pred_df["volume"].values
    })


    current_context = pd.concat(
        [
            current_context,
            predicted_context
        ],
        ignore_index=True
    )


    # --------------------------------------------------------
    # Move forward
    # --------------------------------------------------------

    completed += this_pred_len


# ============================================================
# 18. COMBINE PREDICTIONS
# ============================================================

pred_df = pd.concat(
    all_predictions,
    ignore_index=True
)


pred_df = (
    pred_df
    .sort_values("timestamp")
    .drop_duplicates("timestamp")
    .reset_index(drop=True)
)


# ============================================================
# 19. MERGE ACTUAL + PREDICTED
# ============================================================

results = actual_df[
    [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]
].copy()


results = results.rename(
    columns={
        "open": "actual_open",
        "high": "actual_high",
        "low": "actual_low",
        "close": "actual_close",
        "volume": "actual_volume"
    }
)


results = results.merge(

    pred_df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ].rename(
        columns={
            "open": "predicted_open",
            "high": "predicted_high",
            "low": "predicted_low",
            "close": "predicted_close",
            "volume": "predicted_volume"
        }
    ),

    on="timestamp",

    how="left"
)


# ============================================================
# 20. BASELINE
#
# Naive baseline:
# previous actual close
#
# For first forecast point, use final
# context close.
# ============================================================

previous_close = context_df[
    "close"
].iloc[-1]


baseline_values = []

last_value = previous_close


for value in results["actual_close"]:

    baseline_values.append(
        last_value
    )

    last_value = value


results["baseline"] = baseline_values


# ============================================================
# 21. PREDICTION ERROR
# ============================================================

results["error"] = (
    results["predicted_close"]
    - results["actual_close"]
)


results["absolute_error"] = (
    results["error"].abs()
)


# ============================================================
# 22. BASELINE ERROR
# ============================================================

results["baseline_error"] = (
    results["baseline"]
    - results["actual_close"]
)


results["baseline_absolute_error"] = (
    results["baseline_error"].abs()
)


# ============================================================
# 23. METRICS
# ============================================================

valid = results.dropna(
    subset=[
        "actual_close",
        "predicted_close"
    ]
).copy()


mae = np.mean(
    np.abs(
        valid["predicted_close"]
        - valid["actual_close"]
    )
)


rmse = np.sqrt(
    np.mean(
        (
            valid["predicted_close"]
            - valid["actual_close"]
        ) ** 2
    )
)


mape = np.mean(
    np.abs(
        (
            valid["predicted_close"]
            - valid["actual_close"]
        )
        /
        valid["actual_close"]
    )
) * 100


baseline_mae = np.mean(
    np.abs(
        valid["baseline"]
        - valid["actual_close"]
    )
)


# ============================================================
# 24. DIVERGENCE THRESHOLD
#
# Use rolling error statistics.
# ============================================================

rolling_mae = (
    results["absolute_error"]
    .rolling(
        window=22,
        min_periods=5
    )
    .mean()
)


results["rolling_mae"] = rolling_mae


results["divergence_threshold"] = (
    rolling_mae * 2
)


results["divergence"] = (
    results["absolute_error"]
    >
    results["divergence_threshold"]
)


# ============================================================
# 25. SUSTAINED DIVERGENCE
#
# Require 3 consecutive divergent observations.
# ============================================================

results["divergence_group"] = (
    results["divergence"]
    .astype(int)
    .rolling(
        window=3,
        min_periods=3
    )
    .sum()
)


results["sustained_divergence"] = (
    results["divergence_group"] >= 3
)


divergence_points = results[
    results["sustained_divergence"]
].copy()


# ============================================================
# 26. SAVE CSV
# ============================================================

results.to_csv(
    CSV_FILE,
    index=False
)


# ============================================================
# 27. CREATE INTERACTIVE GRAPH
# ============================================================

print()
print("=" * 80)
print("CREATING INTERACTIVE GRAPH")
print("=" * 80)


fig = go.Figure()


# ============================================================
# 28. 400-DAY TRAINING / CONTEXT DATA
# ============================================================

fig.add_trace(
    go.Candlestick(

        x=context_df["timestamp"],

        open=context_df["open"],

        high=context_df["high"],

        low=context_df["low"],

        close=context_df["close"],

        name="400-Day Training Context"
    )
)


# ============================================================
# 29. ACTUAL FORECAST-PERIOD PRICE
# ============================================================

fig.add_trace(
    go.Scatter(

        x=results["timestamp"],

        y=results["actual_close"],

        mode="lines",

        name="Actual Close",

        line=dict(
            width=2
        )
    )
)


# ============================================================
# 30. KRONOS PREDICTION
# ============================================================

fig.add_trace(
    go.Scatter(

        x=results["timestamp"],

        y=results["predicted_close"],

        mode="lines",

        name="Kronos Prediction",

        line=dict(
            width=2,
            dash="dash"
        )
    )
)


# ============================================================
# 31. BASELINE
# ============================================================

fig.add_trace(
    go.Scatter(

        x=results["timestamp"],

        y=results["baseline"],

        mode="lines",

        name="Naive Baseline",

        line=dict(
            width=1,
            dash="dot"
        )
    )
)


# ============================================================
# 32. START OF FORECAST LINE
# ============================================================

fig.add_vline(

    x=forecast_start,

    line_width=2,

    line_dash="dash"
)


fig.add_annotation(

    x=forecast_start,

    y=1,

    yref="paper",

    text="Forecast starts",

    showarrow=False,

    yanchor="bottom"
)


# ============================================================
# 33. DIVERGENCE LINES
# ============================================================

# Avoid drawing hundreds of overlapping lines.
# Draw the first point of each sustained divergence episode.

divergence_dates = []

if len(divergence_points) > 0:

    previous_date = None

    for date in divergence_points[
        "timestamp"
    ]:

        if (
            previous_date is None
            or
            (date - previous_date).days > 5
        ):

            divergence_dates.append(
                date
            )

        previous_date = date


for date in divergence_dates:

    row = results[
        results["timestamp"] == date
    ]

    if len(row) == 0:
        continue

    price = row[
        "actual_close"
    ].iloc[0]


    fig.add_vline(

        x=date,

        line_width=2,

        line_dash="dot"
    )


    fig.add_annotation(

        x=date,

        y=price,

        text="Divergence",

        showarrow=True,

        arrowhead=2
    )


# ============================================================
# 34. GRAPH LAYOUT
# ============================================================

fig.update_layout(

    title=(
        "RELIANCE.NS — Kronos "
        "400-Day Context → "
        "2022-01-01 to 2026-08-14"
    ),

    xaxis_title="Date / Time",

    yaxis_title="Price",

    hovermode="x unified",

    height=900,

    template="plotly_white",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
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
                    label="1M",
                    step="month",
                    stepmode="backward"
                ),

                dict(
                    count=3,
                    label="3M",
                    step="month",
                    stepmode="backward"
                ),

                dict(
                    count=6,
                    label="6M",
                    step="month",
                    stepmode="backward"
                ),

                dict(
                    count=1,
                    label="1Y",
                    step="year",
                    stepmode="backward"
                ),

                dict(
                    step="all",
                    label="ALL"
                )
            ]
        )
    )
)


# ============================================================
# 35. SAVE HTML
# ============================================================

fig.write_html(

    GRAPH_FILE,

    include_plotlyjs=True,

    full_html=True
)


# ============================================================
# 36. REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w"
) as f:

    f.write(
        "KRONOS LONG-RANGE FORECAST REPORT\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Ticker              : {TICKER}\n"
    )

    f.write(
        f"Context periods     : {LOOKBACK}\n"
    )

    f.write(
        f"Context start       : "
        f"{context_df['timestamp'].iloc[0]}\n"
    )

    f.write(
        f"Context end         : "
        f"{context_df['timestamp'].iloc[-1]}\n"
    )

    f.write(
        f"Forecast start      : {START_DATE}\n"
    )

    f.write(
        f"Forecast end        : {END_DATE}\n"
    )

    f.write(
        f"Forecast rows       : "
        f"{len(results)}\n\n"
    )

    f.write(
        f"MAE                 : "
        f"{mae:.4f}\n"
    )

    f.write(
        f"RMSE                : "
        f"{rmse:.4f}\n"
    )

    f.write(
        f"MAPE                : "
        f"{mape:.4f}%\n"
    )

    f.write(
        f"Baseline MAE        : "
        f"{baseline_mae:.4f}\n\n"
    )

    f.write(
        f"Sustained divergence episodes : "
        f"{len(divergence_dates)}\n\n"
    )

    f.write(
        "Divergence dates:\n"
    )

    for date in divergence_dates:

        f.write(
            f"{date}\n"
        )


# ============================================================
# 37. FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("FORECAST COMPLETE")
print("=" * 80)

print()
print("Context:")
print(
    context_df["timestamp"].iloc[0],
    "to",
    context_df["timestamp"].iloc[-1]
)

print()
print("Forecast:")
print(
    results["timestamp"].iloc[0],
    "to",
    results["timestamp"].iloc[-1]
)

print()
print("Rows predicted:", len(results))

print()
print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("MAPE:", round(mape, 4), "%")

print()
print("Baseline MAE:", round(baseline_mae, 4))

print()
print("Divergence episodes:")
print(len(divergence_dates))

print()
print("GRAPH:")
print(GRAPH_FILE)

print()
print("CSV:")
print(CSV_FILE)

print()
print("REPORT:")
print(REPORT_FILE)

print()
print("=" * 80)
