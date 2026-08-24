import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = os.path.expanduser(
    "~/Kronos/het/data/NIFTY50_5Y_OHLCV.parquet"
)

# Existing CSV produced by the completed rolling experiment.
PREDICTION_CSV = (
    "nifty50_rolling_lora_predictions.csv"
)

# Updated CSV containing the calculated prediction error.
OUTPUT_CSV = (
    "nifty50_2022_rolling_lora_predictions_with_difference.csv"
)

# Final Plotly HTML.
OUTPUT_HTML = (
    "nifty50_2022_rolling_lora_predictions_updated.html"
)

WINDOW_SIZE = 40


# ============================================================
# LOAD ORIGINAL MARKET DATA
# ============================================================

print("=" * 70)
print("REBUILDING ROLLING FORECAST PLOT")
print("=" * 70)

print("\nLoading original NIFTY 50 data...")
print(DATA_PATH)

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"Original data file not found:\n{DATA_PATH}"
    )

df = pd.read_parquet(
    DATA_PATH
)


# ============================================================
# NORMALIZE ORIGINAL DATA COLUMNS
# ============================================================

if isinstance(
    df.columns,
    pd.MultiIndex
):

    df.columns = [
        str(column[0]).lower()
        for column in df.columns
    ]

else:

    df.columns = [
        str(column).lower()
        for column in df.columns
    ]


# ============================================================
# FIND TIMESTAMP COLUMN
# ============================================================

if "timestamps" not in df.columns:

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

    else:

        # Timestamp may be stored in the index.
        df = df.reset_index()

        df.columns = [
            str(column).lower()
            for column in df.columns
        ]

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

        elif "index" in df.columns:

            df.rename(
                columns={
                    "index": "timestamps"
                },
                inplace=True
            )


if "timestamps" not in df.columns:

    raise RuntimeError(
        "Could not identify timestamp column.\n"
        f"Available columns: {df.columns.tolist()}"
    )


# ============================================================
# CLEAN ORIGINAL DATA
# ============================================================

df["timestamps"] = pd.to_datetime(
    df["timestamps"]
)

if df["timestamps"].dt.tz is not None:

    df["timestamps"] = (
        df["timestamps"]
        .dt.tz_localize(None)
    )


required_columns = [
    "timestamps",
    "close",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:

    raise RuntimeError(
        "Missing required columns in original data: "
        + str(missing)
    )


df = df[
    required_columns
].copy()


df.dropna(
    inplace=True
)


df.sort_values(
    "timestamps",
    inplace=True
)


df.drop_duplicates(
    subset="timestamps",
    keep="last",
    inplace=True
)


df.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# LOAD EXISTING PREDICTIONS
# ============================================================

print("\nLoading existing predictions...")
print(PREDICTION_CSV)

if not os.path.exists(
    PREDICTION_CSV
):

    raise FileNotFoundError(
        f"\nPrediction CSV not found:\n"
        f"{os.path.abspath(PREDICTION_CSV)}\n\n"
        "Use the filename of the CSV produced by "
        "your completed rolling experiment."
    )


results = pd.read_csv(
    PREDICTION_CSV
)


# ============================================================
# CLEAN PREDICTION TIMESTAMPS
# ============================================================

if "timestamp" not in results.columns:

    raise RuntimeError(
        "Prediction CSV does not contain 'timestamp'.\n"
        f"Available columns: {results.columns.tolist()}"
    )


results["timestamp"] = pd.to_datetime(
    results["timestamp"]
)


results.sort_values(
    "timestamp",
    inplace=True
)


results.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# CHECK REQUIRED PREDICTION COLUMNS
# ============================================================

required_prediction_columns = [
    "actual_close",
    "pred_close",
]

missing = [
    column
    for column in required_prediction_columns
    if column not in results.columns
]

if missing:

    raise RuntimeError(
        "Prediction CSV is missing required columns: "
        + str(missing)
    )


# ============================================================
# CHECK THAT PREDICTIONS EXIST
# ============================================================

if len(results) == 0:

    raise RuntimeError(
        "Prediction CSV contains no prediction rows."
    )


# ============================================================
# CALCULATE PREDICTION ERROR
# ============================================================

# Error convention:
#
#     Error = Predicted - Actual
#
# Positive error:
#     Model predicted HIGHER than actual.
#
# Negative error:
#     Model predicted LOWER than actual.

results["close_difference"] = (
    results["pred_close"]
    - results["actual_close"]
)


results["absolute_close_error"] = (
    results["close_difference"]
    .abs()
)


# ============================================================
# SAVE UPDATED CSV
# ============================================================

results.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# PRINT ACTUAL VS PREDICTED
# ============================================================

print("\n")
print("=" * 100)
print("ACTUAL VS PREDICTED CLOSE")
print("=" * 100)

print(
    f"{'Date':<15}"
    f"{'Actual':>15}"
    f"{'Predicted':>15}"
    f"{'Error':>15}"
)

print("-" * 70)


for _, row in results.iterrows():

    print(
        f"{row['timestamp'].date()!s:<15}"
        f"{row['actual_close']:>15.4f}"
        f"{row['pred_close']:>15.4f}"
        f"{row['close_difference']:>+15.4f}"
    )


# ============================================================
# ERROR SUMMARY
# ============================================================

mae = (
    results["absolute_close_error"]
    .mean()
)


rmse = np.sqrt(
    np.mean(
        results["close_difference"] ** 2
    )
)


mean_difference = (
    results["close_difference"]
    .mean()
)


print("\n")
print("=" * 70)
print("CLOSE ERROR SUMMARY")
print("=" * 70)

print(
    f"Predictions:        {len(results):,}"
)

print(
    f"Mean Difference:    {mean_difference:.6f}"
)

print(
    f"Close MAE:          {mae:.6f}"
)

print(
    f"Close RMSE:         {rmse:.6f}"
)


# ============================================================
# FIRST PREDICTION
# ============================================================

first_prediction_date = (
    results["timestamp"].iloc[0]
)

last_prediction_date = (
    results["timestamp"].iloc[-1]
)


print(
    f"\nFirst prediction:   "
    f"{first_prediction_date.date()}"
)

print(
    f"Last prediction:    "
    f"{last_prediction_date.date()}"
)


# ============================================================
# DETERMINE PLOT START
# ============================================================

# We explicitly include the 40 trading days immediately
# preceding the first prediction.
#
# If the first prediction is the 41st trading day,
# this means the graph starts at the first day of its
# 40-day context window.
#
# We find the actual row corresponding to the first
# prediction and take WINDOW_SIZE rows before it.

prediction_positions = (
    df.index[
        df["timestamps"]
        < first_prediction_date
    ]
)


if len(prediction_positions) < WINDOW_SIZE:

    raise RuntimeError(
        "Original NIFTY data does not contain enough "
        "historical rows before the first prediction "
        f"to show the requested {WINDOW_SIZE}-day context."
    )


first_prediction_position = (
    prediction_positions[-1]
    + 1
)


plot_start_position = (
    first_prediction_position
    - WINDOW_SIZE
)


if plot_start_position < 0:

    plot_start_position = 0


# ============================================================
# GET ACTUAL DATA FOR PLOT
# ============================================================

# Include:
#
#   40 historical context days
#   +
#   all actual days for which predictions exist
#
# This gives the complete actual curve from the beginning
# of the first rolling window through the final prediction.

plot_df = df.iloc[
    plot_start_position:
].copy()


plot_df = plot_df[
    plot_df["timestamps"]
    <= last_prediction_date
].copy()


if len(plot_df) == 0:

    raise RuntimeError(
        "No actual market data available for plotting."
    )


print(
    f"\nPlot actual data: "
    f"{plot_df['timestamps'].iloc[0].date()}"
    f" → "
    f"{plot_df['timestamps'].iloc[-1].date()}"
)

print(
    f"Actual points plotted: "
    f"{len(plot_df):,}"
)


# ============================================================
# BUILD FIGURE
# ============================================================

fig = go.Figure()


# ============================================================
# ACTUAL CLOSE
# ============================================================

fig.add_trace(
    go.Scatter(
        x=plot_df["timestamps"],
        y=plot_df["close"],
        mode="lines",
        name="Actual Close",
        line=dict(
            width=1.5
        ),

        # Do not create a separate hover box for
        # the actual trace. The prediction trace below
        # contains the three values we want.
        hoverinfo="skip",
    )
)


# ============================================================
# PREDICTED CLOSE
# ============================================================

fig.add_trace(
    go.Scatter(
        x=results["timestamp"],
        y=results["pred_close"],
        mode="lines",
        name="LoRA Predicted Close",

        line=dict(
            width=1.3
        ),

        # Only pass the two values needed in the hover:
        #
        # customdata[0] = actual
        # customdata[1] = error
        customdata=np.column_stack(
            [
                results["actual_close"].values,
                results["close_difference"].values,
            ]
        ),

        # EXACTLY THREE THINGS:
        #
        # Actual
        # Predicted
        # Error
        #
        # No date.
        # No absolute error.
        # No extra information.
        hovertemplate=(
            "<b>Actual:</b> "
            "%{customdata[0]:.2f}<br>"
            "<b>Predicted:</b> "
            "%{y:.2f}<br>"
            "<b>Error:</b> "
            "%{customdata[1]:+.2f}"
            "<extra></extra>"
        ),
    )
)


# ============================================================
# MARK FIRST PREDICTION
# ============================================================

fig.add_vline(
    x=first_prediction_date,
    line_dash="dash",
    annotation_text="First prediction",
    annotation_position="top"
)


# ============================================================
# LAYOUT
# ============================================================

fig.update_layout(

    title=(
        "NIFTY 50 - Rolling LoRA "
        "One-Day-Ahead Forecast"
    ),

    xaxis_title="Date",

    yaxis_title="NIFTY 50",

    # IMPORTANT:
    #
    # Do NOT use "x unified".
    #
    # "x unified" would create a combined hover box
    # containing information from multiple traces.
    #
    # "closest" ensures the prediction trace displays
    # only its three-value hover template.
    hovermode="closest",

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
        t=100,
        b=70
    )
)


# ============================================================
# SAVE HTML
# ============================================================

fig.write_html(
    OUTPUT_HTML,
    include_plotlyjs=True,
    full_html=True
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Updated CSV:\n"
    f"{os.path.abspath(OUTPUT_CSV)}"
)

print(
    f"\nUpdated HTML:\n"
    f"{os.path.abspath(OUTPUT_HTML)}"
)

print(
    "\nNo model was loaded."
)

print(
    "No training was performed."
)

print(
    "Existing predictions were reused."
)

print(
    "\nGraph contains:"
)

print(
    "  - First 40 historical trading days"
)

print(
    "  - Actual Close"
)

print(
    "  - LoRA Predicted Close"
)

print(
    "  - First prediction marker"
)

print(
    "  - No prediction-difference trace"
)

print(
    "  - Hover: Actual / Predicted / Error only"
)

print("=" * 70)